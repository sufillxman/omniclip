from django.contrib import admin
from apps.engine.models import Project, MediaAsset

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'format_type', 'render_status')
    list_filter = ('format_type', 'render_status')
    search_fields = ('user__email', 'base_prompt', 'celery_task_id')

@admin.register(MediaAsset)
class MediaAssetAdmin(admin.ModelAdmin):
    list_display = ('id', 'project', 'media_type', 'file_url')
    list_filter = ('media_type',)
    search_fields = ('project__id', 'file_url')
