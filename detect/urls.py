from django.urls import path
# from .views_old import history, detect
from .views import classify_image, history

urlpatterns = [
    path('history/', history, name='history'),
    path('detect/', classify_image, name='detect'),
]
