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
        args, _ = mock_assemble.call_args
        video_clips = args[1]
        self.assertEqual(len(video_clips), 2)
        self.assertEqual(video_clips[0][1], 3.5)
        # 7.5 (total audio duration) - 3.5 (first clip duration) = 4.0
        self.assertEqual(video_clips[1][1], 4.0)
        
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


