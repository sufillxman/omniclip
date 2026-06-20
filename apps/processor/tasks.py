import os
import logging
import ffmpeg
from celery import shared_task
from django.db import transaction
from django.conf import settings
from apps.engine.models import Project, MediaAsset
from apps.processor.services.audio_service import generate_tts_audio
from apps.processor.services.media_service import fetch_background_video_with_status
from apps.processor.services.whisper_service import WhisperAlignmentService, sentence_aware_chunk
from apps.processor.services.ffmpeg_service import assemble_final_video, resolve_local_path

logger = logging.getLogger(__name__)

def mock_refund_credits(project):
    """
    Auto-refund utility to reimburse user credits when a rendering pipeline fails.
    """
    from django.db import transaction
    from apps.accounts.models import Credits
    from django.conf import settings

    render_cost = getattr(settings, 'RENDER_COST', 10)
    logger.info(f"Initiating credit refund for project {project.id} (user: {project.user.email}).")
    try:
        with transaction.atomic():
            credits_obj, created = Credits.objects.select_for_update().get_or_create(user=project.user)
            credits_obj.balance += render_cost
            current_history = credits_obj.transaction_history if credits_obj.transaction_history else []
            current_history.append({
                "type": "refund",
                "amount": render_cost,
                "project_id": str(project.id)
            })
            credits_obj.transaction_history = current_history
            credits_obj.save()
            logger.info(f"Successfully refunded {render_cost} credits to user {project.user.email} (New Balance: {credits_obj.balance}).")
    except Exception as exc:
        logger.error(f"Failed to refund credits for project {project.id}: {exc}")

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_video_render_task(self, project_id):
    """
    Shared Celery task to render a project video using Audio-Driven Sync architecture.

    Pipeline:
      1. Fetch project
      2. Set status → PROCESSING
      3. Generate TTS voiceover (HybridTTSService: ElevenLabs → EdgeTTS fallback)
      4. Run Whisper alignment → extract exact per-segment timestamps from real audio
      5. Fetch background video clips (one per Whisper-aligned segment)
      6. Pass precise durations to FFmpeg → stitch final video
      7. Set status → COMPLETED
    """
    logger.info(f"Starting background video render task for project: {project_id}")

    # 1. Fetch the project
    try:
        project = Project.objects.get(id=project_id)
    except Project.DoesNotExist as exc:
        logger.error(f"Project {project_id} not found in database.")
        raise self.retry(exc=exc, max_retries=1)

    # 2. Update render status to Processing
    try:
        with transaction.atomic():
            project.render_status = Project.RenderStatus.PROCESSING
            project.save()
            logger.info(f"Project {project_id} status updated to: PROCESSING")
    except Exception as exc:
        logger.error(f"Database update error for project {project_id} status (Processing): {exc}")
        try:
            project.render_status = Project.RenderStatus.FAILED
            project.save()
            mock_refund_credits(project)
        except Exception:
            pass
        raise self.retry(exc=exc)

    # 3. Generate TTS audio, run Whisper alignment, fetch video clips
    audio_path = None
    video_clips = []
    layout = project.script_data.get('layout', 'landscape')

    try:
        # Clear old assets bound to this project if running retries
        with transaction.atomic():
            project.media_assets.all().delete()

        # ── Step A: Generate TTS voiceover ───────────────────────────────
        voiceover_script = project.script_data.get('voiceover_script', '')
        if voiceover_script:
            audio_path = generate_tts_audio(voiceover_script, voice_id="standard_male", project_id=project.human_name)
            with transaction.atomic():
                MediaAsset.objects.create(
                    project=project,
                    media_type=MediaAsset.MediaType.AUDIO,
                    file_url=audio_path
                )
            logger.info(f"Successfully generated and registered TTS Audio: {audio_path}")

        # ── Step B: Whisper alignment — derive exact timestamps from real audio ──
        raw_timeline = project.json_timeline  # [{chunk_index, text, visual_keyword}, ...]
        aligned_timeline = WhisperAlignmentService.align(audio_path, raw_timeline)
        logger.info(
            f"[WhisperAlignment] Aligned {len(aligned_timeline)} segments with real audio timestamps."
        )

        # ── Step C: Fetch background video clips with Whisper-precise durations ──
        for idx, chunk in enumerate(aligned_timeline):
            keyword    = chunk.get('visual_keyword', 'default_topic')
            start_time = chunk.get('start_time', 0.0)
            end_time   = chunk.get('end_time', 0.0)
            duration   = max(0.1, end_time - start_time)  # floor at 100ms to avoid zero-length clips

            logger.info(
                f"Processing segment {idx}: keyword='{keyword}', "
                f"whisper_start={start_time}s, whisper_end={end_time}s, duration={duration:.3f}s"
            )

            video_path, was_cloned = fetch_background_video_with_status(
                keyword, project_id=project.human_name, chunk_index=idx, layout=layout
            )
            # Store clip as a dict with clone provenance for FFmpeg
            video_clips.append({
                "path":       video_path,
                "duration":   duration,
                "is_cloned":  was_cloned,
                "source_idx": idx - 1 if was_cloned else -1,
            })

            with transaction.atomic():
                MediaAsset.objects.create(
                    project=project,
                    media_type=MediaAsset.MediaType.VIDEO_CLIP,
                    file_url=video_path
                )
            logger.info(
                f"Successfully fetched and registered video clip: {video_path}"
                + (f" [CLONED from chunk_{idx-1}]" if was_cloned else "")
            )

        # ── Step D: Adjust last chunk duration to match real audio duration ──
        if audio_path and video_clips:
            try:
                resolved_audio     = resolve_local_path(audio_path)
                probe_data         = ffmpeg.probe(resolved_audio)
                real_audio_duration = float(probe_data['format']['duration'])

                sum_prior_durations  = sum(clip["duration"] for clip in video_clips[:-1])
                new_last_duration    = max(0.1, real_audio_duration - sum_prior_durations)

                old_last_duration    = video_clips[-1]["duration"]
                video_clips[-1]["duration"] = new_last_duration
                logger.info(
                    f"Adjusted last video clip duration from {old_last_duration:.3f}s to "
                    f"{new_last_duration:.3f}s to match real audio duration {real_audio_duration:.3f}s."
                )
            except Exception as probe_err:
                logger.error(f"Error probing audio duration or adjusting chunk: {probe_err}")

        # ── Step E: Build sentence-aware subtitle chunks for ASS burn-in ──────
        # sentence_aware_chunk() requires word-level timestamps, which the
        # WhisperAlignmentService already extracted internally during align().
        # We re-extract them here from the enriched aligned_timeline by treating
        # each segment's text as a flat word stream with proportional timestamps.
        subtitle_chunks = []
        try:
            flat_words = []
            for seg in aligned_timeline:
                seg_start  = seg.get('start_time', 0.0)
                seg_end    = seg.get('end_time',   0.0)
                seg_text   = seg.get('text', '')
                seg_words  = seg_text.split()
                if not seg_words:
                    continue
                # Distribute time uniformly across words within this segment
                word_dur   = (seg_end - seg_start) / len(seg_words)
                for wi, w in enumerate(seg_words):
                    flat_words.append({
                        "word":  w,
                        "start": round(seg_start + wi * word_dur, 3),
                        "end":   round(seg_start + (wi + 1) * word_dur, 3),
                    })

            subtitle_chunks = sentence_aware_chunk(flat_words)
            logger.info(
                f"[Subtitles] Built {len(subtitle_chunks)} sentence-aware subtitle chunks "
                f"from {len(flat_words)} word timestamps."
            )
        except Exception as sub_err:
            logger.error(
                f"[Subtitles] sentence_aware_chunk() failed: {sub_err}. "
                f"Rendering without subtitles."
            )

    except Exception as exc:
        logger.error(f"Failed to fetch assets or build MediaAssets for project {project_id}: {exc}")
        try:
            with transaction.atomic():
                project.render_status = Project.RenderStatus.FAILED
                project.save()
                mock_refund_credits(project)
        except Exception:
            pass
        raise exc

    # 4. Assemble the final video using FFmpeg with Whisper-exact durations and ASS subtitles
    final_output_path = os.path.join(settings.MEDIA_ROOT, 'projects', project.human_name, 'final', 'final_output.mp4')
    try:
        logger.info("Stitching clips together via FFmpeg Service...")
        assemble_final_video(
            audio_path,
            video_clips,
            final_output_path,
            subtitle_chunks=subtitle_chunks if subtitle_chunks else None,
            layout=layout,
        )

        with transaction.atomic():
            MediaAsset.objects.create(
                project=project,
                media_type=MediaAsset.MediaType.VIDEO_FINAL,
                file_url=final_output_path
            )
        logger.info(f"Final video exported and registered at: {final_output_path}")

    except Exception as exc:
        logger.error(f"FFmpeg assembly pipeline failed for project {project_id}: {exc}")
        try:
            with transaction.atomic():
                project.render_status = Project.RenderStatus.FAILED
                project.save()
                mock_refund_credits(project)
        except Exception:
            pass
        raise exc

    # 5. Update status to Completed
    try:
        with transaction.atomic():
            project.render_status = Project.RenderStatus.COMPLETED
            project.save()
            logger.info(f"Project {project_id} render completed successfully.")
    except Exception as exc:
        logger.error(f"Database update error for project {project_id} status (Completed): {exc}")
        try:
            with transaction.atomic():
                project.render_status = Project.RenderStatus.FAILED
                project.save()
                mock_refund_credits(project)
        except Exception:
            pass
        raise self.retry(exc=exc)

    return f"Project {project_id} render completed successfully."
