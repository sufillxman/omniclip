from django.test import TestCase, override_settings
from django.conf import settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from unittest.mock import patch
from rest_framework.test import APITestCase

from rest_framework import status
from apps.engine.models import Project, MediaAsset
from apps.accounts.models import Credits
import uuid

User = get_user_model()

class EngineModelsTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='engineuser',
            email='engine@example.com',
            password='password123'
        )

    def test_project_model_creation(self):
        """Verify Project model fields and UUID generation."""
        project = Project.objects.create(
            user=self.user,
            base_prompt="Create an educational video about gravity",
            format_type=Project.FormatType.REAL_VIDEO,
            script_data={"title": "Gravity Intro"},
            json_timeline=[{"start": 0, "end": 2.5, "text": "Gravity holds us down."}],
            celery_task_id="celery-task-1234"
        )
        self.assertIsInstance(project.id, uuid.UUID)
        self.assertEqual(project.user, self.user)
        self.assertEqual(project.base_prompt, "Create an educational video about gravity")
        self.assertEqual(project.format_type, Project.FormatType.REAL_VIDEO)
        self.assertEqual(project.script_data, {"title": "Gravity Intro"})
        self.assertEqual(project.json_timeline[0]["text"], "Gravity holds us down.")
        self.assertEqual(project.celery_task_id, "celery-task-1234")
        self.assertEqual(project.render_status, Project.RenderStatus.PENDING)
        self.assertEqual(str(project), f"Project {project.id} (Real-Video) - Pending")

    def test_media_asset_model_creation(self):
        """Verify MediaAsset fields and relationships."""
        project = Project.objects.create(
            user=self.user,
            base_prompt="Create a promo video",
            format_type=Project.FormatType.ANIME_VIDEO
        )
        asset = MediaAsset.objects.create(
            project=project,
            media_type=MediaAsset.MediaType.VIDEO_FINAL,
            file_url="/media/final_outputs/rendered_video.mp4"
        )
        self.assertIsInstance(asset.id, uuid.UUID)
        self.assertEqual(asset.project, project)
        self.assertEqual(asset.media_type, MediaAsset.MediaType.VIDEO_FINAL)
        self.assertEqual(asset.file_url, "/media/final_outputs/rendered_video.mp4")
        self.assertEqual(str(asset), f"Asset {asset.id} (Video_Final) for Project {project.id}")


class EngineViewsTestCase(APITestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(
            username='user1',
            email='user1@example.com',
            password='password123'
        )
        self.user2 = User.objects.create_user(
            username='user2',
            email='user2@example.com',
            password='password123'
        )
        self.project1 = Project.objects.create(
            user=self.user1,
            base_prompt="Prompt 1",
            format_type=Project.FormatType.REAL_VIDEO,
            script_data={"text": "Old text"}
        )
        self.project2 = Project.objects.create(
            user=self.user2,
            base_prompt="Prompt 2",
            format_type=Project.FormatType.CAROUSEL
        )
        # Pre-create credit balances
        Credits.objects.create(user=self.user1, balance=100)
        Credits.objects.create(user=self.user2, balance=100)

    def test_anonymous_requests_are_denied(self):
        """Verify engine endpoints reject anonymous requests with 403."""
        urls = [
            reverse('generate-script'),
            reverse('edit-script', args=[self.project1.id]),
            reverse('render-full'),
            reverse('project-status', args=[self.project1.id]),
        ]
        for url in urls:
            response = self.client.post(url) if 'status' not in url and 'edit' not in url else self.client.get(url)
            self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @patch('apps.engine.views.GeminiService.generate_script')
    def test_generate_script_api(self, mock_gen_script):
        """Verify post generation maps prompt parameter, calls gemini service, and saves record."""
        # Mock structured JSON output conforming to new Audio-Driven Sync schema
        mock_gen_script.return_value = {
            "title": "Short catchy title",
            "seo_tags": ["tag1", "tag2"],
            "voiceover_script": "Full continuous text for TTS",
            "json_timeline": [
                {"chunk_index": 0, "text": "Full continuous text for TTS", "visual_keyword": "keyword1"}
            ]
        }

        self.client.force_authenticate(user=self.user1)
        post_data = {
            'prompt': '5 facts about space',
            'tone': 'Excited',
            'format_type': Project.FormatType.ANIME_VIDEO
        }
        response = self.client.post(reverse('generate-script'), post_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['base_prompt'], '5 facts about space')
        self.assertEqual(response.data['format_type'], Project.FormatType.ANIME_VIDEO)
        self.assertEqual(response.data['script_data']['title'], 'Short catchy title')
        self.assertEqual(response.data['script_data']['voiceover_script'], 'Full continuous text for TTS')
        self.assertEqual(len(response.data['json_timeline']), 1)
        self.assertEqual(response.data['json_timeline'][0]['text'], 'Full continuous text for TTS')

        # Check DB
        new_project = Project.objects.get(id=response.data['id'])
        self.assertEqual(new_project.user, self.user1)


    def test_edit_script_api_isolation(self):
        """Verify user can update script data on own projects and is blocked on others."""
        self.client.force_authenticate(user=self.user1)
        
        # 1. Edit own project
        put_data = {'script_data': {'text': 'New updated text'}}
        edit_url = reverse('edit-script', args=[self.project1.id])
        response = self.client.put(edit_url, put_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['script_data']['text'], 'New updated text')

        # Check DB
        self.project1.refresh_from_db()
        self.assertEqual(self.project1.script_data['text'], 'New updated text')

        # 2. Try to edit user2's project
        edit_url2 = reverse('edit-script', args=[self.project2.id])
        response = self.client.put(edit_url2, put_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_full_render_api_isolation(self):
        """Verify full render endpoint schedules a mock task and updates status."""
        self.client.force_authenticate(user=self.user1)

        # 1. Render own project
        post_data = {
            'project_id': str(self.project1.id),
            'voice_id': 'voice-1',
            'bgm_preset': 'rock'
        }
        response = self.client.post(reverse('render-full'), post_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('celery_task_id', response.data)
        self.assertEqual(response.data['message'], 'Rendering started')

        # Check DB
        self.project1.refresh_from_db()
        self.assertEqual(self.project1.render_status, Project.RenderStatus.PROCESSING)
        self.assertEqual(self.project1.celery_task_id, response.data['celery_task_id'])

        # 2. Try to render user2's project
        post_data['project_id'] = str(self.project2.id)
        response = self.client.post(reverse('render-full'), post_data)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_project_status_polling_isolation(self):
        """Verify polling returns status and url, and restricts queries by user."""
        self.client.force_authenticate(user=self.user1)

        # 1. Fetch own project status (Pending)
        status_url = reverse('project-status', args=[self.project1.id])
        response = self.client.get(status_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], Project.RenderStatus.PENDING)
        self.assertIsNone(response.data['video_url'])

        # 2. Fetch completed project with asset
        self.project1.render_status = Project.RenderStatus.COMPLETED
        self.project1.save()
        asset = MediaAsset.objects.create(
            project=self.project1,
            media_type=MediaAsset.MediaType.VIDEO_FINAL,
            file_url='/media/final_outputs/result.mp4'
        )

        response = self.client.get(status_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], Project.RenderStatus.COMPLETED)
        self.assertEqual(response.data['video_url'], '/media/final_outputs/result.mp4')

        # 3. Access user2 status
        status_url2 = reverse('project-status', args=[self.project2.id])
        response = self.client.get(status_url2)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_full_render_deducts_credits(self):
        """Verify render-full API call deducts credits from user balance."""
        self.client.force_authenticate(user=self.user1)
        credits_obj = self.user1.credits
        self.assertEqual(credits_obj.balance, 100)

        post_data = {
            'project_id': str(self.project1.id),
            'voice_id': 'voice-1',
            'bgm_preset': 'rock'
        }
        response = self.client.post(reverse('render-full'), post_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        credits_obj.refresh_from_db()
        self.assertEqual(credits_obj.balance, 90)

    def test_full_render_idempotency_guard(self):
        """Verify calling render-full on a project already being processed returns HTTP 409 Conflict."""
        self.project1.render_status = Project.RenderStatus.PROCESSING
        self.project1.save()

        self.client.force_authenticate(user=self.user1)
        post_data = {
            'project_id': str(self.project1.id),
            'voice_id': 'voice-1',
            'bgm_preset': 'rock'
        }
        response = self.client.post(reverse('render-full'), post_data)
        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(response.data['error'], "Project rendering is already in progress.")

    @patch('apps.engine.views.GeminiService.generate_script')
    def test_throttling_limits(self, mock_gen_script):
        """Verify rate limiting kicks in after exceeding limit."""
        from apps.engine.views import GenerateScriptThrottle
        from django.core.cache import cache
        cache.clear()

        GenerateScriptThrottle.rate = '2/hour'
        try:
            mock_gen_script.return_value = {
                "title": "catchy title",
                "seo_tags": ["tag"],
                "voiceover_script": "text",
                "json_timeline": []
            }
            self.client.force_authenticate(user=self.user1)
            post_data = {'prompt': 'prompt', 'tone': 'tone'}

            # 1st request - ok
            resp1 = self.client.post(reverse('generate-script'), post_data)
            self.assertEqual(resp1.status_code, status.HTTP_201_CREATED)

            # 2nd request - ok
            resp2 = self.client.post(reverse('generate-script'), post_data)
            self.assertEqual(resp2.status_code, status.HTTP_201_CREATED)

            # 3rd request - throttled!
            resp3 = self.client.post(reverse('generate-script'), post_data)
            self.assertEqual(resp3.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
        finally:
            GenerateScriptThrottle.rate = None

    def test_project_post_delete_signal_removes_media_folder(self):
        """Verify that deleting a Project deletes its corresponding media folder from disk."""
        project = Project.objects.create(
            user=self.user1,
            base_prompt="Temp project for cleanup test",
            format_type=Project.FormatType.REAL_VIDEO
        )
        import os
        # Create a temp file in the project folder
        project_dir = os.path.join(settings.MEDIA_ROOT, 'projects', project.human_name)
        os.makedirs(project_dir, exist_ok=True)
        temp_file_path = os.path.join(project_dir, 'dummy.txt')
        with open(temp_file_path, 'w') as f:
            f.write('dummy content')
            
        self.assertTrue(os.path.exists(temp_file_path))
        
        # Delete project
        project.delete()
        
        # Assert directory is removed
        self.assertFalse(os.path.exists(project_dir))


from unittest.mock import MagicMock
from apps.engine.services.gemini_service import GeminiService

class GeminiServiceTestCase(TestCase):
    @patch('google.genai.Client')
    def test_generate_script_calls_client(self, mock_client_cls):
        """Verify GeminiService generate_script correctly initializes and calls GenAI client."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = '{"title": "Space Voyage", "seo_tags": ["space"], "voiceover_script": "Space.", "json_timeline": []}'
        mock_client.models.generate_content.return_value = mock_response

        # Call service method
        result = GeminiService.generate_script("base prompt", "Real-Video")

        # Verify Client initialized with API key
        mock_client_cls.assert_called_once_with(api_key=settings.GEMINI_API_KEY)
        
        # Verify generate_content called
        mock_client.models.generate_content.assert_called_once()
        args, kwargs = mock_client.models.generate_content.call_args
        self.assertEqual(kwargs['model'], 'gemini-2.5-flash')
        
        # Verify config attributes on GenerateContentConfig
        from apps.engine.services.gemini_service import VideoScript
        self.assertEqual(kwargs['config'].response_mime_type, 'application/json')
        self.assertEqual(kwargs['config'].response_schema, VideoScript)
        
        # Verify correct return structure
        self.assertEqual(result['title'], 'Space Voyage')
