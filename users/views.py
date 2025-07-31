# users/views.py
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import MedicoSignUpForm, PacienteSignUpForm # Importa o novo formulário

class MedicoSignUpView(CreateView):
    form_class = MedicoSignUpForm
    success_url = reverse_lazy('login') 
    template_name = 'registration/signup.html'

# --- NOVA VIEW ---
@login_required
def paciente_signup_view(request):
    # Garante que apenas médicos possam cadastrar pacientes
    if not hasattr(request.user, 'medico'):
        messages.error(request, "Apenas médicos podem cadastrar novos pacientes.")
        return redirect('home')

    if request.method == 'POST':
        form = PacienteSignUpForm(request.POST)
        if form.is_valid():
            # Passa o médico logado para o método save do formulário
            form.save(medico=request.user.medico)
            messages.success(request, "Paciente cadastrado com sucesso!")
            return redirect('medico_dashboard') # Volta para o dashboard do médico
        else:
            messages.error(request, "Por favor, corrija os erros no formulário.")
    else:
        form = PacienteSignUpForm()
    
    return render(request, 'registration/paciente_signup.html', {'form': form})