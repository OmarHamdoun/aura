from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),

    path("stream/camera/", views.stream_camera, name="stream_camera"),
    path("stream/video/", views.stream_video, name="stream_video"),

    path("captions/camera/", views.captions_camera, name="captions_camera"),
    path("captions/video/", views.captions_video, name="captions_video"),
    path("captions/camera/clear/", views.clear_captions_camera, name="clear_captions_camera"),
    path("captions/video/clear/", views.clear_captions_video, name="clear_captions_video"),

    # NEW decision routes
    path("decide/", views.decide, name="decide"),
    path("decide/camera/", views.decide_camera, name="decide_camera"),
    path("decide/video/", views.decide_video, name="decide_video"),

    # OpenTTS proxy
    path("tts/voices/", views.opentts_voices, name="opentts_voices"),
    path("tts/speak/", views.opentts_tts, name="opentts_tts"),
]
