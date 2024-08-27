from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('user_attention_data/', views.user_attention_data, name='user_attention_data'),
]
