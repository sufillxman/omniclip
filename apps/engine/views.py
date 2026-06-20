import uuid
from django.shortcuts import get_object_or_404
from django.db import transaction
from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from apps.engine.models import Project, MediaAsset
from apps.engine.serializers import ProjectSerializer
from apps.engine.services.gemini_service import GeminiService
from apps.processor.tasks import process_video_render_task
from apps.accounts.models import Credits

class GenerateScriptThrottle(UserRateThrottle):
    scope = 'generate_script_limit'

class FullRenderThrottle(UserRateThrottle):
    scope = 'full_render_limit'

class GenerateScriptAPIView(APIView):
    """
    POST /api/v1/engine/generate-script/
    Accepts prompt, tone, and format_type (optional, defaults to Real-Video).
    Creates a new Project instance populated with structure-compliant script and timeline generated via Gemini API.
    """
    permission_classes = [IsAuthenticated]
    throttle_classes = [GenerateScriptThrottle]

    def post(self, request, *args, **kwargs):
        prompt = request.data.get('prompt')
        tone = request.data.get('tone')
        format_type = request.data.get('format_type', Project.FormatType.REAL_VIDEO)

        if not prompt or not tone:
            return Response({"error": "prompt and tone are required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            gemini_data = GeminiService.generate_script(
                base_prompt=f"Topic: {prompt}. Tone: {tone}.",
                format_type=format_type
            )
        except Exception as exc:
            return Response(
                {"error": f"Failed to generate script via Gemini API: {str(exc)}"},
                status=status.HTTP_502_BAD_GATEWAY
            )

        # Parse generated data
        script_data = {
            "title": gemini_data.get("title", ""),
            "seo_tags": gemini_data.get("seo_tags", []),
            "voiceover_script": gemini_data.get("voiceover_script", "")
        }
        json_timeline = gemini_data.get("json_timeline", [])

        # Create Project in DB
        project = Project.objects.create(
            user=request.user,
            base_prompt=prompt,
            format_type=format_type,
            script_data=script_data,
            json_timeline=json_timeline,
            render_status=Project.RenderStatus.PENDING
        )

        serializer = ProjectSerializer(project)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class EditScriptAPIView(APIView):
    """
    PUT /api/v1/engine/edit-script/{id}/
    Allows user to update the script_data field of their own project before rendering.
    """
    permission_classes = [IsAuthenticated]

    def put(self, request, pk, *args, **kwargs):
        project = get_object_or_404(Project, id=pk, user=request.user)
        script_data = request.data.get('script_data')

        if script_data is None:
            return Response({"error": "script_data is required"}, status=status.HTTP_400_BAD_REQUEST)

        project.script_data = script_data
        project.save()

        serializer = ProjectSerializer(project)
        return Response(serializer.data, status=status.HTTP_200_OK)

class FullRenderAPIView(APIView):
    """
    POST /api/v1/engine/render/full/
    Verifies that the target project belongs to the user, triggers the render process,
    assigns a dummy Celery task ID, updates render_status to Processing, and returns details.
    """
    permission_classes = [IsAuthenticated]
    throttle_classes = [FullRenderThrottle]

    def post(self, request, *args, **kwargs):
        project_id = request.data.get('project_id')
        voice_id = request.data.get('voice_id')
        bgm_preset = request.data.get('bgm_preset')

        if not project_id:
            return Response({"error": "project_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with transaction.atomic():
                # select_for_update to serialize requests and apply idempotency lock
                project = Project.objects.select_for_update().get(id=project_id, user=request.user)

                # Idempotency guard:
                # "if project.render_status == PROCESSING, it rejects the request with HTTP 409 Conflict."
                if project.render_status == Project.RenderStatus.PROCESSING:
                    return Response(
                        {"error": "Project rendering is already in progress."},
                        status=status.HTTP_409_CONFLICT
                    )

                # Atomic balance check and deduction:
                # "implement select_for_update() to atomically check the user's Credits balance (ensure it's >= RENDER_COST) and atomically deduct the credits before dispatching the Celery task."
                credits_obj, created = Credits.objects.select_for_update().get_or_create(user=request.user)
                render_cost = getattr(settings, 'RENDER_COST', 10)

                if credits_obj.balance < render_cost:
                    return Response(
                        {"error": f"Insufficient credits. Required: {render_cost}, Available: {credits_obj.balance}."},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Deduct credits atomically
                credits_obj.balance -= render_cost
                current_history = credits_obj.transaction_history if credits_obj.transaction_history else []
                current_history.append({
                    "type": "deduction",
                    "amount": render_cost,
                    "project_id": str(project.id)
                })
                credits_obj.transaction_history = current_history
                credits_obj.save()

                # Mark project as PROCESSING to hold lock state
                project.render_status = Project.RenderStatus.PROCESSING
                project.save()

            # Dispatch Celery task AFTER the transaction commits to avoid race condition where worker
            # starts before database transaction is committed.
            task = process_video_render_task.delay(str(project.id))

            # Store the task.id in a separate atomic write
            with transaction.atomic():
                project = Project.objects.get(id=project.id)
                project.celery_task_id = task.id
                project.save()

            return Response({
                "message": "Rendering started",
                "celery_task_id": task.id
            }, status=status.HTTP_200_OK)

        except Project.DoesNotExist:
            return Response({"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ProjectStatusAPIView(APIView):
    """
    GET /api/v1/engine/project/{id}/status/
    The polling endpoint to check progress. Returns current status and video URL if Completed.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, id, *args, **kwargs):
        project = get_object_or_404(Project, id=id, user=request.user)

        video_url = None
        if project.render_status == Project.RenderStatus.COMPLETED:
            # Query for the final video asset bound to the project
            asset = project.media_assets.filter(media_type=MediaAsset.MediaType.VIDEO_FINAL).first()
            if asset:
                video_url = asset.file_url

        return Response({
            "status": project.render_status,
            "video_url": video_url
        }, status=status.HTTP_200_OK)
