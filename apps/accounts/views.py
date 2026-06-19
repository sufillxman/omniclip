from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import get_user_model
from apps.accounts.models import BrandKit, Credits, UserAPIKey
from apps.accounts.serializers import (
    UserSerializer, BrandKitSerializer, CreditsSerializer, UserAPIKeySerializer
)

User = get_user_model()

class UserViewSet(viewsets.ModelViewSet):
    """
    ViewSet to view and edit the current logged-in user's profile.
    """
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return User.objects.filter(id=self.request.user.id)

class BrandKitViewSet(viewsets.ModelViewSet):
    """
    ViewSet to manage user Brand Kits.
    Users can only access and edit their own Brand Kits.
    """
    serializer_class = BrandKitSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return BrandKit.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class CreditsViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ReadOnlyViewSet to retrieve user credits.
    Users can only view their own credits. Balance changes are internal.
    """
    serializer_class = CreditsSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Credits.objects.filter(user=self.request.user)

class VerifyAndSaveAPIKeyView(APIView):
    """
    Endpoint (POST /api/v1/accounts/keys/verify/) to verify and save a user API Key.
    Simulates validation by asserting the key length is greater than 5.
    If valid, it updates an existing entry for that service or creates a new one.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = UserAPIKeySerializer(data=request.data)
        if serializer.is_valid():
            service = serializer.validated_data['service']
            api_key = serializer.validated_data['api_key']

            # Simulated key verification (length check > 5)
            if len(api_key) <= 5:
                return Response(
                    {"error": "API key verification failed. Key length must be greater than 5 characters."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Retrieve existing key if present to update, avoiding uniqueness conflicts
            instance = UserAPIKey.objects.filter(user=request.user, service=service).first()
            if instance:
                serializer = UserAPIKeySerializer(instance, data=request.data, partial=True)
                serializer.is_valid(raise_exception=True)

            serializer.save(user=request.user, is_valid=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
