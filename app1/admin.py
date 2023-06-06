from django.contrib import admin



def user_post_save(user, instance, created, **kwargs):
    if created and user.objects.count() > 6:
        instance.is_active = False
        instance.save()
# Register your models here.
