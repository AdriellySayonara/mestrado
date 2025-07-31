from django.shortcuts import render, redirect

def home(request):
    if request.user.is_authenticated:
        # Verifica se o usuário tem um perfil de Medico
        if hasattr(request.user, 'medico'):
            return redirect('medico_dashboard')
        # Verifica se o usuário tem um perfil de Paciente
        elif hasattr(request.user, 'paciente'):
            return redirect('paciente_dashboard')
    
    # Se não estiver logado ou não tiver perfil, mostra a home page
    return render(request, 'home.html')