from django.urls import path
# from .views_old import history, detect
from .views import classify_image, history
from .viewws import EfficientNetV2InferencerAPIView

urlpatterns = [
    path('history/', history, name='history'),
    # path('detect/', classify_image, name='detect'),
    path('detect/', EfficientNetV2InferencerAPIView.as_view(), name='detect'),
]
