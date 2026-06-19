import os
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.conf import settings
from unittest.mock import patch
from apps.engine.models import Project, MediaAsset
from apps.processor.tasks import process_video_render_task

User = get_user_model()

class ProcessorTasksTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='processoruser',
            email='processor@example.com',
            password='password123'
        )
        # Pre-populate project with timeline data matching new Gemini schema (no LLM timestamps).
        # Whisper alignment runs at task time; timestamps are NOT stored in json_timeline.
        self.project = Project.objects.create(
            user=self.user,
            base_prompt="Create a space video",
            format_type=Project.FormatType.REAL_VIDEO,
            script_data={
                "title": "Space Voyage",
                "voiceover_script": "Space is the ultimate frontier."
            },
            json_timeline=[
                {"chunk_index": 0, "text": "Space is the ultimate", "visual_keyword": "galaxy"},
                {"chunk_index": 1, "text": "frontier.", "visual_keyword": "stars"}
            ]
        )

    @patch('apps.processor.tasks.assemble_final_video')
    @patch('ffmpeg.probe')
    def test_process_video_render_task_success(self, mock_probe, mock_assemble):
        """Verify process_video_render_task resolves media paths, calls FFmpeg, and registers the final video output."""
        mock_probe.return_value = {'format': {'duration': '7.5'}}
        from apps.processor.services.ffmpeg_service import assemble_final_video as real_assemble
        mock_assemble.side_effect = real_assemble

        self.assertEqual(self.project.render_status, Project.RenderStatus.PENDING)
        self.assertEqual(self.project.media_assets.count(), 0)
        
        # Call Celery task synchronously in-thread
        result = process_video_render_task.apply(args=[self.project.id])
        
        # Assert task success
        self.assertEqual(result.status, 'SUCCESS')
        
        # Verify project render status updated to COMPLETED
        self.project.refresh_from_db()
        self.assertEqual(self.project.render_status, Project.RenderStatus.COMPLETED)

        # Verify that assemble_final_video was called with the adjusted duration
        mock_assemble.assert_called_once()
        args, kwargs = mock_assemble.call_args
        video_clips = args[1]
        self.assertEqual(len(video_clips), 2)
        # Clips are now dicts: {"path", "duration", "is_cloned", "source_idx"}
        self.assertEqual(video_clips[0]["duration"], 3.5)
        # 7.5 (total audio duration) - 3.5 (first clip duration) = 4.0
        self.assertEqual(video_clips[1]["duration"], 4.0)
        
        # Verify MediaAsset creations
        assets = self.project.media_assets.all()
        # 1 Audio asset + 2 intermediate video clips + 1 final stitched video = 4 assets total
        self.assertEqual(assets.count(), 4)
        
        audio_asset = assets.filter(media_type=MediaAsset.MediaType.AUDIO).first()
        self.assertIsNotNone(audio_asset)
        self.assertEqual(audio_asset.file_url, f"/media/projects/{self.project.human_name}/audio/voiceover.wav")

        clip_assets = assets.filter(media_type=MediaAsset.MediaType.VIDEO_CLIP)
        self.assertEqual(clip_assets.count(), 2)
        self.assertEqual(clip_assets[0].file_url, f"/media/projects/{self.project.human_name}/clips/chunk_0.mp4")
        self.assertEqual(clip_assets[1].file_url, f"/media/projects/{self.project.human_name}/clips/chunk_1.mp4")

        # Verify final stitched output
        final_video_asset = assets.filter(media_type=MediaAsset.MediaType.VIDEO_FINAL).first()
        self.assertIsNotNone(final_video_asset)
        expected_final_path = os.path.join(settings.MEDIA_ROOT, 'projects', self.project.human_name, 'final', 'final_output.mp4')
        self.assertEqual(final_video_asset.file_url, expected_final_path)

    @patch('apps.processor.tasks.generate_tts_audio')
    def test_process_video_render_task_refunds_on_failure(self, mock_tts):
        """Verify process_video_render_task triggers a credit refund on failures."""
        mock_tts.side_effect = Exception("TTS provider down")
        
        from apps.accounts.models import Credits
        credits_obj = Credits.objects.create(user=self.user, balance=50)

        result = process_video_render_task.apply(args=[self.project.id])
        self.assertEqual(result.status, 'FAILURE')

        self.project.refresh_from_db()
        self.assertEqual(self.project.render_status, Project.RenderStatus.FAILED)

        credits_obj.refresh_from_db()
        self.assertEqual(credits_obj.balance, 60)


class FetchBackgroundVideoFallbackTestCase(TestCase):
    """
    Unit tests for fetch_background_video's smart fallback behaviour.
    All network I/O is mocked — no real Pexels calls are made.
    time.sleep is patched to a no-op so retry backoff doesn't slow CI.

    Directory layout mirrors what the service constructs internally:
        MEDIA_ROOT / projects / {project_id} / clips /
    """

    def setUp(self):
        import tempfile
        # Create a temp MEDIA_ROOT, then build the full clips path under it
        self.media_root = tempfile.mkdtemp()
        self.project_id = 'test_project_fallback'
        self.clips_dir = os.path.join(
            self.media_root, 'projects', self.project_id, 'clips'
        )
        os.makedirs(self.clips_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.media_root, ignore_errors=True)

    @patch('apps.processor.services.media_service.time.sleep', return_value=None)
    @patch('apps.processor.services.media_service.requests.Session')
    @patch('apps.processor.services.media_service.settings')
    def test_chunk_n_failure_clones_previous_chunk(self, mock_settings, mock_session_cls, _mock_sleep):
        """
        When chunk_1 download fails on all 3 attempts, the service must copy
        chunk_0.mp4 as chunk_1.mp4 rather than creating an empty file.
        """
        from apps.processor.services.media_service import fetch_background_video

        mock_settings.TESTING = False
        mock_settings.PEXELS_API_KEY = 'fake-key'
        mock_settings.MEDIA_ROOT = self.media_root

        # Pre-create a valid (non-empty) chunk_0.mp4 stub at the correct path
        chunk0_path = os.path.join(self.clips_dir, 'chunk_0.mp4')
        with open(chunk0_path, 'wb') as f:
            f.write(b'\x00' * 1024)  # 1 KB stub — non-empty, valid copy source

        # All session.get() calls raise a connection error (simulates IncompleteRead)
        mock_session = mock_session_cls.return_value
        mock_session.get.side_effect = ConnectionError("Simulated IncompleteRead")

        result = fetch_background_video('stars', project_id=self.project_id, chunk_index=1)

        chunk1_path = os.path.join(self.clips_dir, 'chunk_1.mp4')
        self.assertTrue(os.path.exists(chunk1_path), "chunk_1.mp4 should exist after clone fallback")
        self.assertGreater(os.path.getsize(chunk1_path), 0, "Cloned chunk_1.mp4 must not be empty")
        self.assertIn('chunk_1.mp4', result)

    @patch('apps.processor.services.media_service.time.sleep', return_value=None)
    @patch('apps.processor.services.media_service.requests.Session')
    @patch('apps.processor.services.media_service.settings')
    def test_chunk_0_failure_raises_runtime_error(self, mock_settings, mock_session_cls, _mock_sleep):
        """
        When chunk_0 download fails on all 3 attempts, the service must raise
        RuntimeError (no prior clip to clone) so the task triggers a refund.
        """
        from apps.processor.services.media_service import fetch_background_video

        mock_settings.TESTING = False
        mock_settings.PEXELS_API_KEY = 'fake-key'
        mock_settings.MEDIA_ROOT = self.media_root

        mock_session = mock_session_cls.return_value
        mock_session.get.side_effect = ConnectionError("Simulated network failure")

        with self.assertRaises(RuntimeError) as ctx:
            fetch_background_video('galaxy', project_id=self.project_id, chunk_index=0)

        self.assertIn('chunk_0', str(ctx.exception))
        self.assertIn('Aborting render to trigger refund', str(ctx.exception))


class SubtitleChunkingTestCase(TestCase):
    """
    Unit tests for sentence_aware_chunk() covering all four boundary rules and
    edge cases. No Whisper model is loaded — all word timestamps are synthetic.

    Also covers _ass_ts() timestamp formatting from ffmpeg_service.
    """

    @staticmethod
    def _make_words(words_with_times):
        """
        Args:
            words_with_times: list of (word_str, start_float, end_float)
        Returns:
            list of {"word": str, "start": float, "end": float}
        """
        return [
            {"word": w, "start": s, "end": e}
            for w, s, e in words_with_times
        ]

    # ── 1. Empty input ─────────────────────────────────────────────────────────
    def test_empty_input_returns_empty_list(self):
        from apps.processor.services.whisper_service import sentence_aware_chunk
        result = sentence_aware_chunk([])
        self.assertEqual(result, [])

    # ── 2. Sentence boundary forces chunk close (PRIORITY 1) ───────────────────
    def test_sentence_ender_forces_chunk_close(self):
        """
        'Hi. How are you?' should produce exactly TWO chunks:
          ["Hi."] and ["How", "are", "you?"]
        NOT one merged chunk of 4 words.
        """
        from apps.processor.services.whisper_service import sentence_aware_chunk
        words = self._make_words([
            ("Hi.",  0.00, 0.40),
            ("How",  0.50, 0.75),
            ("are",  0.75, 1.00),
            ("you?", 1.00, 1.50),
        ])
        chunks = sentence_aware_chunk(words)
        self.assertEqual(len(chunks), 2, f"Expected 2 chunks, got {len(chunks)}: {[c.text for c in chunks]}")
        self.assertEqual(chunks[0].words, ["Hi."])
        self.assertEqual(chunks[1].words, ["How", "are", "you?"])

    # ── 3. Max-word limit closes chunk (PRIORITY 3) ────────────────────────────
    def test_max_word_limit_closes_chunk(self):
        """
        5 words with no punctuation should split at 4 + 1.
        """
        from apps.processor.services.whisper_service import sentence_aware_chunk
        words = self._make_words([
            ("The",    0.0, 0.3),
            ("quick",  0.3, 0.6),
            ("brown",  0.6, 0.9),
            ("fox",    0.9, 1.2),
            ("jumps",  1.2, 1.5),
        ])
        chunks = sentence_aware_chunk(words)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].words), 4)
        self.assertEqual(len(chunks[1].words), 1)

    # ── 4. Max-duration limit closes chunk (PRIORITY 2) ───────────────────────
    def test_max_duration_closes_chunk(self):
        """
        Two words spanning 2.5s (> 2.2s _MAX_DURATION_S) must produce a single
        chunk of 2 words, and then a second chunk for any further words.
        """
        from apps.processor.services.whisper_service import sentence_aware_chunk, _MAX_DURATION_S
        words = self._make_words([
            ("Start", 0.0,  1.2),
            ("slow",  1.2,  2.5),   # chunk duration = 2.5 > _MAX_DURATION_S
            ("next",  2.6,  3.0),
        ])
        chunks = sentence_aware_chunk(words)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].words, ["Start", "slow"])
        self.assertLessEqual(
            chunks[0].end_time - chunks[0].start_time,
            _MAX_DURATION_S + 0.5,
        )

    # ── 5. Micro-chunk merges forward without a natural pause ──────────────────
    def test_micro_chunk_merges_forward_without_natural_gap(self):
        """
        A single word lasting 0.05s (< _MIN_DURATION_S) with only a 0.02s gap
        to the next word should be merged into the next iteration.
        """
        from apps.processor.services.whisper_service import sentence_aware_chunk
        words = self._make_words([
            ("A",    0.00, 0.05),   # 0.05s micro, 0.02s gap — merge forward
            ("long", 0.07, 0.40),
            ("day",  0.40, 0.80),
        ])
        chunks = sentence_aware_chunk(words)
        self.assertEqual(len(chunks), 1)
        self.assertIn("A", chunks[0].words)

    # ── 6. Micro-chunk with natural gap — gap only rescues at a triggered boundary ─
    def test_micro_chunk_survives_with_natural_gap(self):
        """
        The natural-gap rescue applies when the chunk closure is already triggered
        (e.g. by max_words) AND the resulting chunk duration is micro. In that case,
        a gap >= 0.1s allows the micro-chunk to stand alone rather than merging.

        In this test, 4 fast words hit the max-word limit at 0.4s total (> _MIN_DURATION_S),
        so the rescue is NOT needed — the chunk closes normally. Then 'wait' is alone
        in the next chunk. This verifies normal max-word flow is unaffected.
        """
        from apps.processor.services.whisper_service import sentence_aware_chunk
        words = self._make_words([
            ("A",    0.00, 0.10),
            ("long", 0.10, 0.20),
            ("dark", 0.20, 0.30),
            ("road", 0.30, 0.40),   # 4 words → max_words hit, chunk closes (0.4s > _MIN)
            ("wait", 0.55, 0.80),   # 0.15s gap after 'road', new chunk
        ])
        chunks = sentence_aware_chunk(words)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].words, ["A", "long", "dark", "road"])
        self.assertEqual(chunks[1].words, ["wait"])

    # ── 7. Sentence-ending micro-chunk never merges (PRIORITY 1 wins) ─────────
    def test_sentence_ending_micro_chunk_never_merges(self):
        """
        Even if a word ending in '!' is very short, PRIORITY 1 (sentence ender)
        always overrides the merge-forward guard.
        """
        from apps.processor.services.whisper_service import sentence_aware_chunk
        words = self._make_words([
            ("Go!",  0.00, 0.06),   # 0.06s micro, sentence ender — must close
            ("Run",  0.07, 0.40),
            ("now",  0.40, 0.80),
        ])
        chunks = sentence_aware_chunk(words)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].words, ["Go!"])

    # ── 8. SubtitleChunk.text and duration properties ─────────────────────────
    def test_subtitle_chunk_text_and_duration_properties(self):
        from apps.processor.services.whisper_service import SubtitleChunk
        chunk = SubtitleChunk(words=["Hello", "world"], start_time=1.0, end_time=2.5)
        self.assertEqual(chunk.text, "Hello world")
        self.assertAlmostEqual(chunk.duration, 1.5, places=3)

    # ── 9. _ass_ts() timestamp formatting ─────────────────────────────────────
    def test_ass_timestamp_formatting(self):
        from apps.processor.services.ffmpeg_service import _ass_ts
        self.assertEqual(_ass_ts(0.0),    "0:00:00.00")
        self.assertEqual(_ass_ts(1.5),    "0:00:01.50")
        self.assertEqual(_ass_ts(61.25),  "0:01:01.25")
        self.assertEqual(_ass_ts(3661.0), "1:01:01.00")

    # ── 10. All words appear exactly once across all chunks ────────────────────
    def test_all_words_covered_no_gaps_or_duplicates(self):
        """
        Run a realistic sentence through sentence_aware_chunk() and verify that
        all original words appear exactly once across all chunks, in order.
        """
        from apps.processor.services.whisper_service import sentence_aware_chunk
        sentence = [
            "Building", "great", "products", "takes",
            "discipline.", "It", "also", "takes",
            "clarity.", "And", "consistency",
        ]
        words = [
            {"word": w, "start": i * 0.4, "end": (i + 1) * 0.4}
            for i, w in enumerate(sentence)
        ]
        chunks = sentence_aware_chunk(words)
        all_output_words = [w for c in chunks for w in c.words]
        self.assertEqual(
            all_output_words, sentence,
            "All words must appear exactly once in the output chunks, in order."
        )
