import os
import uuid
import shutil
from django.db import models
from django.conf import settings
from django.utils.text import slugify
from django.db.models.signals import post_delete
from django.dispatch import receiver

class Project(models.Model):
    class FormatType(models.TextChoices):
        REAL_VIDEO = 'Real-Video', 'Real-Video'
        ANIME_VIDEO = 'Anime-Video', 'Anime-Video'
        CAROUSEL = 'Carousel', 'Carousel'

    class RenderStatus(models.TextChoices):
        PENDING = 'Pending', 'Pending'
        PROCESSING = 'Processing', 'Processing'
        COMPLETED = 'Completed', 'Completed'
        FAILED = 'Failed', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='projects'
    )
    base_prompt = models.TextField()
    format_type = models.CharField(
        max_length=20,
        choices=FormatType.choices
    )
    script_data = models.JSONField(default=dict, blank=True)
    json_timeline = models.JSONField(default=list, blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    render_status = models.CharField(
        max_length=20,
        choices=RenderStatus.choices,
        default=RenderStatus.PENDING
    )

    def __str__(self):
        return f"Project {self.id} ({self.format_type}) - {self.render_status}"

    @property
    def human_name(self) -> str:
        prompt_segment = self.base_prompt[:30]
        slug = slugify(prompt_segment).replace('-', '_')
        uuid_segment = str(self.id)[:8]
        if slug:
            return f"{slug}_{uuid_segment}"
        return f"project_{uuid_segment}"

class MediaAsset(models.Model):
    class MediaType(models.TextChoices):
        AUDIO = 'Audio', 'Audio'
        IMAGE = 'Image', 'Image'
        VIDEO_CLIP = 'Video_Clip', 'Video_Clip'
        VIDEO_FINAL = 'Video_Final', 'Video_Final'


    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='media_assets'
    )
    media_type = models.CharField(
        max_length=20,
        choices=MediaType.choices
    )
    file_url = models.CharField(max_length=1000)

    def __str__(self):
        return f"Asset {self.id} ({self.media_type}) for Project {self.project.id}"

@receiver(post_delete, sender=Project)
def delete_project_media(sender, instance, **kwargs):
    """
    Erase the project's media directory from the disk when the project is deleted.
    """
    project_root = os.path.join(settings.MEDIA_ROOT, 'projects', instance.human_name)
    shutil.rmtree(project_root, ignore_errors=True)
