from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('webcam_feed/', views.webcam_feed_view, name='webcam_feed'),
    path('weapon_detection_status/', views.weapon_detection_status, name='weapon_detection_status'),
]
