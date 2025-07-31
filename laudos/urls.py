# laudos/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/medico/', views.medico_dashboard, name='medico_dashboard'),
    path('dashboard/paciente/', views.paciente_dashboard, name='paciente_dashboard'),
    
    # --- URL ATUALIZADA PARA CHAMAR A NOVA VIEW ---
    path('laudo/<int:pk>/', views.laudo_detail_view, name='laudo_detail'),
    # ----------------------------------------------

    path('laudo/<int:pk>/pdf/', views.laudo_pdf_view, name='laudo_pdf'),
]