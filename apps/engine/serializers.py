from rest_framework import serializers
from apps.engine.models import Project, MediaAsset

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = [
            'id', 'user', 'base_prompt', 'format_type', 
            'script_data', 'json_timeline', 'celery_task_id', 'render_status'
        ]
        # Read-only attributes to prevent frontend users from overriding internal states or IDs
        read_only_fields = ['id', 'user', 'script_data', 'json_timeline', 'celery_task_id', 'render_status']

class MediaAssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = MediaAsset
        fields = ['id', 'project', 'media_type', 'file_url']
        read_only_fields = ['id']
