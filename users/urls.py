# users/urls.py
from django.urls import path, include
from . import views

urlpatterns = [
    path('accounts/', include('django.contrib.auth.urls')), 
    path('accounts/signup/', views.MedicoSignUpView.as_view(), name='signup'),
    # --- NOVA URL ---
    path('paciente/novo/', views.paciente_signup_view, name='paciente_signup'),
]