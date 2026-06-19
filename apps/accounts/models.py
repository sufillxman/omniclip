import uuid
import base64
import hashlib
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
# pyrefly: ignore [missing-import]
from cryptography.fernet import Fernet

def get_fernet_cipher():
    """
    Initializes a Fernet cipher instance using FIELD_ENCRYPTION_KEY.
    """
    encryption_key = getattr(settings, 'FIELD_ENCRYPTION_KEY', None)
    if not encryption_key:
        raise ValueError("FIELD_ENCRYPTION_KEY is not configured in settings.")
    return Fernet(encryption_key.encode('utf-8'))

class User(AbstractUser):
    class Tier(models.TextChoices):
        FREE = 'Free', 'Free'
        PRO = 'Pro', 'Pro'
        AGENCY = 'Agency', 'Agency'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    stripe_customer_id = models.CharField(max_length=255, blank=True, null=True)
    account_tier = models.CharField(
        max_length=10,
        choices=Tier.choices,
        default=Tier.FREE
    )

    # Use email as username field for authenticating
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email

class BrandKit(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='brand_kits'
    )
    primary_hex = models.CharField(max_length=7, default='#FFFFFF')
    secondary_hex = models.CharField(max_length=7, default='#000000')
    custom_font_url = models.URLField(max_length=500, blank=True, null=True)
    watermark_logo_url = models.URLField(max_length=500, blank=True, null=True)

    def __str__(self):
        return f"Brand Kit for {self.user.email}"

class Credits(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='credits'
    )
    balance = models.IntegerField(default=0)
    transaction_history = models.JSONField(default=list, blank=True)

    class Meta:
        verbose_name = "Credits"
        verbose_name_plural = "Credits"

    def __str__(self):
        return f"{self.user.email} - Balance: {self.balance}"

class UserAPIKey(models.Model):
    class Service(models.TextChoices):
        GEMINI = 'Gemini', 'Gemini'
        ELEVENLABS = 'ElevenLabs', 'ElevenLabs'
        PEXELS = 'Pexels', 'Pexels'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='api_keys'
    )
    service = models.CharField(
        max_length=20,
        choices=Service.choices
    )
    encrypted_api_key = models.CharField(max_length=512)
    is_valid = models.BooleanField(default=True)

    class Meta:
        verbose_name = "User API Key"
        verbose_name_plural = "User API Keys"
        unique_together = ('user', 'service')

    def __str__(self):
        return f"{self.service} key for {self.user.email}"

    def get_api_key(self) -> str:
        """Decrypts and returns the raw API key."""
        if not self.encrypted_api_key:
            return ""
        cipher = get_fernet_cipher()
        decrypted = cipher.decrypt(self.encrypted_api_key.encode('utf-8'))
        return decrypted.decode('utf-8')

    def set_api_key(self, raw_key: str):
        """Encrypts and stores the raw API key."""
        if not raw_key:
            self.encrypted_api_key = ""
            return
        cipher = get_fernet_cipher()
        encrypted = cipher.encrypt(raw_key.encode('utf-8'))
        self.encrypted_api_key = encrypted.decode('utf-8')

    @property
    def api_key(self) -> str:
        return self.get_api_key()

    @api_key.setter
    def api_key(self, value: str):
        self.set_api_key(value)
