import re
from rest_framework import serializers
from django.contrib.auth import get_user_model
from apps.accounts.models import BrandKit, Credits, UserAPIKey

User = get_user_model()

HEX_COLOR_RE = re.compile(r'^#(?:[0-9a-fA-F]{3}){1,2}$')

def validate_hex_color(value):
    if not HEX_COLOR_RE.match(value):
        raise serializers.ValidationError("Must be a valid hex color code (e.g., #FFFFFF or #FFF).")
    return value

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'stripe_customer_id', 'account_tier']
        read_only_fields = ['id', 'stripe_customer_id', 'account_tier']

class BrandKitSerializer(serializers.ModelSerializer):
    primary_hex = serializers.CharField(validators=[validate_hex_color], default='#FFFFFF')
    secondary_hex = serializers.CharField(validators=[validate_hex_color], default='#000000')

    class Meta:
        model = BrandKit
        fields = ['id', 'user', 'primary_hex', 'secondary_hex', 'custom_font_url', 'watermark_logo_url']
        read_only_fields = ['id', 'user']


class CreditsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Credits
        fields = ['id', 'user', 'balance', 'transaction_history']
        read_only_fields = ['id', 'user', 'balance', 'transaction_history']

class UserAPIKeySerializer(serializers.ModelSerializer):
    # Critical security rule: api_key must be write_only=True so the frontend never receives it.
    api_key = serializers.CharField(write_only=True, required=True, min_length=5)

    class Meta:
        model = UserAPIKey
        fields = ['id', 'service', 'api_key', 'is_valid']
        read_only_fields = ['id', 'is_valid']

    def create(self, validated_data):
        api_key = validated_data.pop('api_key')
        instance = UserAPIKey(**validated_data)
        instance.api_key = api_key
        instance.save()
        return instance

    def update(self, instance, validated_data):
        api_key = validated_data.pop('api_key', None)
        if api_key is not None:
            instance.api_key = api_key
        return super().update(instance, validated_data)
