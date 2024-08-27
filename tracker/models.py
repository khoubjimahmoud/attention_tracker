from django.db import models

class UserAttention(models.Model):
    user_id = models.CharField(max_length=100)
    photo = models.ImageField(upload_to='photos/')
    attention_data = models.JSONField()

    def __str__(self):
        return self.user_id
