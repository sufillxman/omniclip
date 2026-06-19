from django.urls import path, include
from rest_framework.routers import DefaultRouter
from apps.accounts.views import (
    UserViewSet, BrandKitViewSet, CreditsViewSet, VerifyAndSaveAPIKeyView
)

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')
router.register(r'brandkits', BrandKitViewSet, basename='brandkit')
router.register(r'credits', CreditsViewSet, basename='credits')

urlpatterns = [
    path('', include(router.urls)),
    path('keys/verify/', VerifyAndSaveAPIKeyView.as_view(), name='verify-key'),
]
