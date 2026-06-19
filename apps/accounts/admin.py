from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from apps.accounts.models import User, BrandKit, Credits, UserAPIKey

class CustomUserAdmin(UserAdmin):
    model = User
    list_display = ('email', 'username', 'stripe_customer_id', 'account_tier', 'is_staff', 'is_active')
    fieldsets = UserAdmin.fieldsets + (
        ('Stripe & Economy Settings', {'fields': ('stripe_customer_id', 'account_tier')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Stripe & Economy Settings', {'fields': ('stripe_customer_id', 'account_tier')}),
    )

admin.site.register(User, CustomUserAdmin)
admin.site.register(BrandKit)
admin.site.register(Credits)
admin.site.register(UserAPIKey)
