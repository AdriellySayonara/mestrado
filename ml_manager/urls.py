# ml_manager/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('ml-panel/train/', views.train_model_view, name='ml_train_model'),
    path('ml-panel/results/', views.results_dashboard_view, name='ml_results_dashboard'),
]