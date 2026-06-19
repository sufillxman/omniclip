from django.urls import path
from apps.engine.views import (
    GenerateScriptAPIView, EditScriptAPIView, FullRenderAPIView, ProjectStatusAPIView
)

urlpatterns = [
    path('generate-script/', GenerateScriptAPIView.as_view(), name='generate-script'),
    path('edit-script/<uuid:pk>/', EditScriptAPIView.as_view(), name='edit-script'),
    path('render/full/', FullRenderAPIView.as_view(), name='render-full'),
    path('project/<uuid:id>/status/', ProjectStatusAPIView.as_view(), name='project-status'),
]
