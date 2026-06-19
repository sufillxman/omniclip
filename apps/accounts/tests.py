from django.test import TestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from apps.accounts.models import BrandKit, Credits, UserAPIKey
from apps.accounts.serializers import UserSerializer, BrandKitSerializer, CreditsSerializer, UserAPIKeySerializer
import uuid

User = get_user_model()

class AccountsModelTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='testuser@example.com',
            password='testpassword123'
        )

    def test_user_creation_and_uuid(self):
        """Verify user is created successfully with a UUID primary key."""
        self.assertEqual(self.user.email, 'testuser@example.com')
        self.assertIsInstance(self.user.id, uuid.UUID)
        self.assertEqual(self.user.account_tier, User.Tier.FREE)

    def test_brand_kit_creation(self):
        """Verify brand kit model creation and associations."""
        brand_kit = BrandKit.objects.create(
            user=self.user,
            primary_hex='#FF0000',
            secondary_hex='#00FF00',
            custom_font_url='https://example.com/font.ttf',
            watermark_logo_url='https://example.com/logo.png'
        )
        self.assertEqual(brand_kit.user, self.user)
        self.assertEqual(brand_kit.primary_hex, '#FF0000')
        self.assertEqual(brand_kit.secondary_hex, '#00FF00')
        self.assertEqual(str(brand_kit), f"Brand Kit for {self.user.email}")

    def test_credits_creation(self):
        """Verify credits creation and default transaction history."""
        credits = Credits.objects.create(
            user=self.user,
            balance=100,
            transaction_history=[
                {"type": "deposit", "amount": 100, "timestamp": "2026-06-16T12:00:00Z"}
            ]
        )
        self.assertEqual(credits.user, self.user)
        self.assertEqual(credits.balance, 100)
        self.assertEqual(len(credits.transaction_history), 1)
        self.assertEqual(credits.transaction_history[0]["type"], "deposit")
        self.assertEqual(str(credits), f"{self.user.email} - Balance: 100")

    def test_user_api_key_encryption_decryption(self):
        """Verify API keys are stored encrypted (AES-256) but decryptable via properties."""
        plain_key = "sk-gemini-1234567890abcdef"
        api_key_entry = UserAPIKey.objects.create(
            user=self.user,
            service=UserAPIKey.Service.GEMINI
        )
        # Set plain key using setter
        api_key_entry.api_key = plain_key
        api_key_entry.save()

        # Refresh from DB
        db_entry = UserAPIKey.objects.get(id=api_key_entry.id)
        
        # Verify stored encrypted string is not the plain text
        self.assertNotEqual(db_entry.encrypted_api_key, plain_key)
        # Verify getter correctly decrypts the key
        self.assertEqual(db_entry.api_key, plain_key)
        # Verify str representation
        self.assertEqual(str(db_entry), f"Gemini key for {self.user.email}")


class AccountsSerializerTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='serializeruser',
            email='serializeruser@example.com',
            password='testpassword123'
        )

    def test_user_serializer(self):
        """Verify UserSerializer fields and read-only attributes."""
        serializer = UserSerializer(instance=self.user)
        data = serializer.data
        self.assertEqual(data['username'], 'serializeruser')
        self.assertEqual(data['email'], 'serializeruser@example.com')
        self.assertEqual(data['account_tier'], 'Free')
        
        # Try to modify read-only fields
        update_data = {
            'username': 'updatedusername',
            'email': 'updated@example.com',
            'stripe_customer_id': 'cus_test123',
            'account_tier': 'Pro'
        }
        serializer = UserSerializer(instance=self.user, data=update_data, partial=True)
        self.assertTrue(serializer.is_valid())
        updated_user = serializer.save()
        
        self.assertEqual(updated_user.username, 'updatedusername')
        self.assertEqual(updated_user.email, 'updated@example.com')
        # Read-only fields should remain unchanged
        self.assertIsNone(updated_user.stripe_customer_id)
        self.assertEqual(updated_user.account_tier, User.Tier.FREE)

    def test_brand_kit_serializer_validation(self):
        """Verify hex color validation on BrandKitSerializer."""
        valid_data = {
            'user': self.user.id,
            'primary_hex': '#FF0055',
            'secondary_hex': '#000'
        }
        serializer = BrandKitSerializer(data=valid_data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        
        invalid_data = {
            'user': self.user.id,
            'primary_hex': 'FF0055', # missing '#'
            'secondary_hex': '#1234' # invalid length
        }
        serializer = BrandKitSerializer(data=invalid_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('primary_hex', serializer.errors)
        self.assertIn('secondary_hex', serializer.errors)

    def test_credits_serializer_read_only(self):
        """Verify Credits fields (balance and history) are strictly read-only."""
        credits = Credits.objects.create(user=self.user, balance=500)
        serializer = CreditsSerializer(instance=credits)
        self.assertEqual(serializer.data['balance'], 500)

        # Attempting to write changes
        update_data = {
            'balance': 1000,
            'transaction_history': [{'amount': 1000}]
        }
        serializer = CreditsSerializer(instance=credits, data=update_data, partial=True)
        self.assertTrue(serializer.is_valid())
        updated_credits = serializer.save()
        # Ensure values did not change
        self.assertEqual(updated_credits.balance, 500)
        self.assertEqual(updated_credits.transaction_history, [])

    def test_user_api_key_serializer_security(self):
        """Verify API Key write-only behavior and successful model encryption integration."""
        plain_key = "elevenlabs-test-secret-key-99"
        input_data = {
            'service': UserAPIKey.Service.ELEVENLABS,
            'api_key': plain_key
        }
        
        # 1. Test creation
        serializer = UserAPIKeySerializer(data=input_data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        
        # We need to supply user via save() or associate it (in real view, this is user=request.user)
        key_entry = serializer.save(user=self.user)
        
        # Verify encryption in DB was successful
        self.assertEqual(key_entry.api_key, plain_key)
        self.assertNotEqual(key_entry.encrypted_api_key, plain_key)
        
        # 2. Test serialization back to client (does NOT contain the key)
        serializer_out = UserAPIKeySerializer(instance=key_entry)
        out_data = serializer_out.data
        
        self.assertIn('is_valid', out_data)
        self.assertIn('service', out_data)
        self.assertNotIn('api_key', out_data)
        self.assertNotIn('encrypted_api_key', out_data)
        
        # 3. Test updates via serializer
        new_key = "pexels-new-key-value-12345"
        update_data = {
            'api_key': new_key
        }
        serializer_update = UserAPIKeySerializer(instance=key_entry, data=update_data, partial=True)
        self.assertTrue(serializer_update.is_valid(), serializer_update.errors)
        updated_entry = serializer_update.save()
        
        # Verify new key is successfully encrypted
        self.assertEqual(updated_entry.api_key, new_key)
        self.assertNotEqual(updated_entry.encrypted_api_key, new_key)


class AccountsViewsTestCase(APITestCase):
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
        self.brand_kit1 = BrandKit.objects.create(
            user=self.user1,
            primary_hex='#111111',
            secondary_hex='#222222'
        )
        self.brand_kit2 = BrandKit.objects.create(
            user=self.user2,
            primary_hex='#AAAAAA',
            secondary_hex='#BBBBBB'
        )
        self.credits1 = Credits.objects.create(user=self.user1, balance=200)
        self.credits2 = Credits.objects.create(user=self.user2, balance=800)

    def test_anonymous_requests_are_denied(self):
        """Verify endpoints reject unauthenticated requests with 403 Forbidden."""
        urls = [
            reverse('user-list'),
            reverse('brandkit-list'),
            reverse('credits-list'),
            reverse('verify-key'),
        ]
        for url in urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_user_viewset_isolation(self):
        """Verify UserViewSet returns only the authenticated user's profile."""
        self.client.force_authenticate(user=self.user1)
        response = self.client.get(reverse('user-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['email'], 'user1@example.com')

    def test_brand_kit_viewset_isolation_and_creation(self):
        """Verify brand kit queries are restricted to self, and perform_create associates requester."""
        # 1. Accessing queryset
        self.client.force_authenticate(user=self.user1)
        response = self.client.get(reverse('brandkit-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['primary_hex'], '#111111')

        # Try to view user2's brand kit directly
        detail_url = reverse('brandkit-detail', args=[self.brand_kit2.id])
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # 2. Creating brand kit auto-association
        # Delete first to avoid multiple kits if testing creation
        self.brand_kit1.delete()
        
        post_data = {
            'primary_hex': '#FFFFFF',
            'secondary_hex': '#000000'
        }
        response = self.client.post(reverse('brandkit-list'), post_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['primary_hex'], '#FFFFFF')
        
        # Verify created kit belongs to user1 in the DB
        created_kit = BrandKit.objects.get(id=response.data['id'])
        self.assertEqual(created_kit.user, self.user1)

    def test_credits_viewset_read_only_isolation(self):
        """Verify CreditsViewSet returns user's balance and blocks edits."""
        self.client.force_authenticate(user=self.user1)
        
        # Retrieve list
        response = self.client.get(reverse('credits-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['balance'], 200)

        # Retrieve detail of other user credits (should be 404)
        detail_url = reverse('credits-detail', args=[self.credits2.id])
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # Try to create credits (should fail with 405 Method Not Allowed)
        response = self.client.post(reverse('credits-list'), {'balance': 9999})
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

    def test_verify_and_save_api_key(self):
        """Verify custom API key verification and idempotency."""
        self.client.force_authenticate(user=self.user1)
        
        # 1. Invalid key (fails custom length check, but passes serializer min_length=5)
        post_data = {
            'service': UserAPIKey.Service.GEMINI,
            'api_key': '12345'
        }
        response = self.client.post(reverse('verify-key'), post_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)


        # 2. Valid key
        post_data['api_key'] = 'valid-key-value-long-enough'
        response = self.client.post(reverse('verify-key'), post_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['is_valid'])
        self.assertEqual(response.data['service'], UserAPIKey.Service.GEMINI)
        # Ensure raw key is write_only and not returned
        self.assertNotIn('api_key', response.data)

        # Verify it exists in DB
        db_key = UserAPIKey.objects.get(user=self.user1, service=UserAPIKey.Service.GEMINI)
        self.assertEqual(db_key.api_key, 'valid-key-value-long-enough')

        # 3. Post again to verify update (idempotency)
        post_data['api_key'] = 'another-updated-valid-key-long'
        response = self.client.post(reverse('verify-key'), post_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        db_key.refresh_from_db()
        self.assertEqual(db_key.api_key, 'another-updated-valid-key-long')
        # Check that we still have only 1 key in the database for this service
        self.assertEqual(UserAPIKey.objects.filter(user=self.user1, service=UserAPIKey.Service.GEMINI).count(), 1)
