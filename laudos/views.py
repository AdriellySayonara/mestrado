# laudos/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count
from django.http import HttpResponse
from django.template.loader import render_to_string
import json
import base64

try:
    from weasyprint import HTML
except ImportError:
    HTML = None

from .forms import LaudoForm
from .models import Laudo
from .services import classify_image

@login_required
def medico_dashboard(request):
    if not hasattr(request.user, 'medico'):
        messages.error(request, 'Acesso negado. Esta área é restrita para médicos.')
        return redirect('home')

    medico = request.user.medico
    
    if request.method == 'POST':
        form = LaudoForm(request.POST, request.FILES, medico=medico)
        if form.is_valid():
            laudo = form.save(commit=False)
            laudo.medico = medico
            laudo.save()
            classify_image(laudo)
            messages.success(request, 'Nova análise solicitada e processada com sucesso!')
            return redirect('medico_dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"Erro no campo '{field}': {error}")
    else:
        form = LaudoForm(medico=medico)

    laudos_existentes = Laudo.objects.filter(medico=medico).select_related('paciente__user').order_by('-data_laudo')
    
    diagnostico_data = laudos_existentes.filter(resultado_classificacao__isnull=False).exclude(resultado_classificacao__exact='').values('resultado_classificacao').annotate(total=Count('id')).order_by('-total')
    
    chart_labels = [item['resultado_classificacao'] for item in diagnostico_data]
    chart_data = [item['total'] for item in diagnostico_data]

    context = {
        'form': form,
        'laudos_existentes': laudos_existentes,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_data),
    }
    return render(request, 'dashboards/medico_dashboard.html', context)



@login_required
def paciente_dashboard(request):
    if not hasattr(request.user, 'paciente'):
        messages.error(request, 'Acesso negado. Esta área é restrita para pacientes.')
        return redirect('home')

    paciente = request.user.paciente
    # A query busca todos os laudos para o paciente logado
    meus_laudos = Laudo.objects.filter(paciente=paciente).select_related('medico__user').order_by('-data_laudo')
    
    context = {
        'meus_laudos': meus_laudos,
        # Passamos o objeto do médico responsável para fácil acesso no template
        'medico_responsavel': paciente.medico_responsavel 
    }
    return render(request, 'dashboards/paciente_dashboard.html', context)

@login_required
def laudo_detail_view(request, pk):
    laudo = get_object_or_404(Laudo.objects.select_related('paciente__user', 'medico__user', 'modelo_utilizado'), pk=pk)

    user_is_medico = hasattr(request.user, 'medico') and request.user.medico == laudo.medico
    user_is_paciente = hasattr(request.user, 'paciente') and request.user.paciente == laudo.paciente
    if not (user_is_medico or user_is_paciente or request.user.is_superuser):
        return HttpResponse("Acesso não autorizado.", status=403)

    context = {
        'laudo': laudo,
        'paciente': laudo.paciente,
        'medico': laudo.medico,
    }
    return render(request, 'laudos/laudo_detail.html', context)

@login_required
def laudo_pdf_view(request, pk):
    if not HTML:
        return HttpResponse("A biblioteca WeasyPrint não está instalada.", status=501)
    try:
        laudo = Laudo.objects.select_related('paciente__user', 'medico__user', 'modelo_utilizado').get(pk=pk)
        
        user_is_medico = hasattr(request.user, 'medico') and request.user.medico == laudo.medico
        user_is_paciente = hasattr(request.user, 'paciente') and request.user.paciente == laudo.paciente
        
        if not (user_is_medico or user_is_paciente or request.user.is_superuser):
            return HttpResponse("Acesso não autorizado.", status=403)

        image_data_uri = None
        if laudo.imagem_lesao and hasattr(laudo.imagem_lesao, 'path'):
            try:
                with open(laudo.imagem_lesao.path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                image_data_uri = f"data:image/jpeg;base64,{encoded_string}"
            except FileNotFoundError:
                print(f"Arquivo de imagem não encontrado para o laudo {laudo.pk}")

        context = {
            'laudo': laudo,
            'imagem_data_uri': image_data_uri
        }
        
        html_string = render_to_string('laudos/laudo_pdf.html', context)
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'inline; filename="laudo_{laudo.paciente.user.username}_{laudo.pk}.pdf"'
        HTML(string=html_string, base_url=request.build_absolute_uri()).write_pdf(response)
        
        return response
    except Laudo.DoesNotExist:
        return HttpResponse("Laudo não encontrado.", status=404)