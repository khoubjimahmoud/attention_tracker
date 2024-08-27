from django.shortcuts import render
from django.http import JsonResponse
import json
import os

def home(request):
    attention_data_path = os.path.join('static', 'data', 'attention_data.json')
    with open(attention_data_path) as f:
        attention_data = json.load(f)
    return render(request, 'tracker/home.html', {'attention_percentages': attention_data})

def user_attention_data(request):
    user = request.GET.get('user')
    user_data_path = os.path.join('static', 'data', user.replace(' ', '_') + '_data.json')
    with open(user_data_path) as f:
        user_data = json.load(f)
    return JsonResponse(user_data)
