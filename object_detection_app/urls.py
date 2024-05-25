from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('webcam_feed/', views.webcam_feed_view, name='webcam_feed'),
    path('weapon_detection_status/', views.weapon_detection_status, name='weapon_detection_status'),
    path('webcam_feed_view/', views.webcam_feed_view, name='webcam_feed_view'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
