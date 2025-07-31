# ml_manager/views.py (Refatorado para o novo Painel de Análise)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import user_passes_test
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

from .forms import TrainingForm, ResultsFilterForm
from .tasks import run_experimental_training_task
from laudos.models import ModeloML

def is_superuser(user):
    return user.is_superuser

@user_passes_test(is_superuser)
def train_model_view(request):
    if request.method == 'POST':
        form = TrainingForm(request.POST, request.FILES)
        if form.is_valid():
            options = form.cleaned_data.copy()
            dataset_zip_file = request.FILES['dataset_zip']
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp_datasets'))
            filename = fs.save(dataset_zip_file.name, dataset_zip_file)
            zip_path = fs.path(filename)
            
            options_for_db = options.copy()
            options_for_db.pop('dataset_zip', None)

            if not options_for_db.get('rf_max_depth'):
                options_for_db['rf_max_depth'] = None

            new_model_instance = ModeloML.objects.create(
                nome_versao=options['version_name'],
                cenario_clinico=options['clinical_scenario'],
                parametros_treinamento=options_for_db
            )
            run_experimental_training_task.delay(new_model_instance.id, zip_path, options_for_db)
            messages.success(request, f"O treinamento para '{options['version_name']}' foi iniciado!")
            return redirect('ml_train_model')
        else:
            messages.error(request, 'O formulário contém erros. Por favor, corrija-os.')
    else:
        form = TrainingForm()
    
    context = {'form': form}
    return render(request, 'ml_manager/train_form.html', context)

def generate_comparative_boxplot(grouped_data, metric_key, metric_label):
    labels = list(grouped_data.keys())
    data_to_plot = [grouped_data[key][metric_key] for key in labels if grouped_data[key].get(metric_key)]
    
    if not data_to_plot:
        return None

    short_labels = [f"Comb. {chr(65 + i)}" for i in range(len(labels))]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, palette="viridis")
    plt.xticks(ticks=range(len(short_labels)), labels=short_labels)
    plt.title(f'Comparativo de {metric_label} por Pipeline')
    plt.ylabel(metric_label)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"


@user_passes_test(is_superuser)
def results_dashboard_view(request):
    filter_form = ResultsFilterForm(request.GET or None)
    
    context = {'filter_form': filter_form}

    if filter_form.is_valid() and filter_form.cleaned_data.get('cenario_clinico') and filter_form.cleaned_data.get('final_classifier'):
        selected_scenario = filter_form.cleaned_data['cenario_clinico']
        selected_classifier_code = filter_form.cleaned_data['final_classifier']
        
        modelos = ModeloML.objects.filter(
            cenario_clinico=selected_scenario,
            parametros_treinamento__final_classifier=selected_classifier_code
        ).order_by('data_upload')
        
        context.update({
            'selected_scenario': selected_scenario,
            'selected_classifier': dict(filter_form.fields['final_classifier'].choices).get(selected_classifier_code),
        })

        grouped_results = defaultdict(lambda: defaultdict(list))
        pipeline_keys_in_order = []
        pipeline_map = defaultdict(list)
        
        for modelo in modelos:
            params = modelo.parametros_treinamento
            if not params or not modelo.metricas_raw: continue

            # --- INÍCIO DA CORREÇÃO: LÓGICA INTELIGENTE PARA CRIAR A CHAVE ---
            pipeline_type = params.get('pipeline_type', 'N/A')
            
            key_parts = [f"Pipeline: {dict(TrainingForm.PIPELINE_CHOICES).get(pipeline_type)}"]

            if pipeline_type != 'classic_only':
                key_parts.append(f"CNN: {params.get('base_architecture', 'N/A')}")
                key_parts.append(f"Ajuste Fino: {'Sim' if params.get('use_fine_tuning') else 'Não'}")
                key_parts.append(f"Aumento de Dados (CNN): {'Sim' if params.get('apply_augmentation') else 'Não'}")

            if pipeline_type != 'end_to_end':
                key_parts.append(f"Balanceamento (Classificador): {params.get('augmentation_method_classic', 'N/A').upper()}")
                key_parts.append(f"Seleção de Features: {params.get('feature_selection_method', 'N/A')}")

            key = "\n".join(key_parts)
            # --- FIM DA CORREÇÃO ---

            if key not in pipeline_map: pipeline_keys_in_order.append(key)
            pipeline_map[key].append(modelo.nome_versao)
            for metric, values in modelo.metricas_raw.items():
                if values: grouped_results[key][metric].extend(values)
        
        context['pipeline_keys'] = pipeline_keys_in_order
        context['pipeline_map'] = dict(pipeline_map)
        context['modelos_count'] = len(pipeline_keys_in_order)

        if len(pipeline_keys_in_order) > 1:
            context['chart_acuracia_uri'] = generate_comparative_boxplot(grouped_results, 'acuracia', 'Acurácia')
            context['chart_sensibilidade_uri'] = generate_comparative_boxplot(grouped_results, 'sensibilidade', 'Sensibilidade')
            context['chart_especificidade_uri'] = generate_comparative_boxplot(grouped_results, 'especificidade', 'Especificidade')
            context['chart_auc_roc_uri'] = generate_comparative_boxplot(grouped_results, 'auc_roc', 'AUC-ROC')
            context['chart_kappa_uri'] = generate_comparative_boxplot(grouped_results, 'kappa', 'Kappa')
    
    return render(request, 'ml_manager/results_dashboard.html', context)