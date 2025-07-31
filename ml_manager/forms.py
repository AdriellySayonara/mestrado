# ml_manager/forms.py (smote_k_neighbors corrigido para IntegerField)
from django import forms
from laudos.models import ModeloML 

class TrainingForm(forms.Form):
    # --- DADOS E IDENTIFICAÇÃO ---
    version_name = forms.CharField(
        label="Nome da Versão do Modelo", 
        max_length=100, 
        help_text="Um nome único e descritivo para este experimento."
    )
    dataset_zip = forms.FileField(
        label="Dataset (.zip)", 
        help_text="Arquivo .zip com pastas de imagens organizadas por classe."
    )
    clinical_scenario = forms.CharField(
        label="Cenário de Diagnóstico", 
        max_length=255, 
        help_text="Nome do cenário para agrupar os modelos. Ex: cenario_hanseniase_melanoma . (Sem acentos ou caracteres especiais)"
    )

    # --- PRÉ-PROCESSAMENTO GLOBAL ---
    apply_augmentation = forms.BooleanField(
        required=False,
        label="Aplicar Aumento de Dados (Augmentation) na CNN?",
        help_text="Gera variações das imagens (rotação, zoom, etc.) para a CNN durante o treinamento."
    )

    # --- TIPO DE PIPELINE EXPERIMENTAL ---
    PIPELINE_CHOICES = [
        ('cnn_hybrid', 'Híbrido (Features CNN + Clássicas + Classificador)'),
        ('cnn_only', 'Apenas CNN (Features CNN + Classificador)'),
        ('classic_only', 'Apenas Clássico (Features Clássicas + Classificador)'),
        ('end_to_end', 'Ponta a Ponta (CNN como Classificador Final)') 
    ]
    pipeline_type = forms.ChoiceField(
        choices=PIPELINE_CHOICES, 
        widget=forms.RadioSelect, 
        label="Tipo de Pipeline Experimental", 
        initial='cnn_only'
    )

    # --- PARÂMETROS DA CNN (Visíveis condicionalmente) ---
    CHOICES_ARCHITECTURE = [('inception_v3', 'Inception V3'), ('efficientnet_b0', 'EfficientNetB0'), ('densenet_201', 'DenseNet201')]
    base_architecture = forms.ChoiceField(label="Arquitetura da CNN Base", choices=CHOICES_ARCHITECTURE, initial='inception_v3')
    use_fine_tuning = forms.BooleanField(required=False, label="Realizar Ajuste Fino (Fine-Tuning) na CNN", help_text="Especializa as camadas da CNN nos seus dados.")
    
    # --- OPÇÕES DO CLASSIFICADOR CLÁSSICO (Visíveis condicionalmente) ---
    fine_tuning_split_ratio = forms.IntegerField(label="Porcentagem de dados para o Ajuste Fino", initial=60, min_value=10, max_value=90, help_text="Define a % do dataset usada para criar o modelo especialista.")
    
    augmentation_method_classic = forms.ChoiceField(
        choices=[('none', 'Nenhum'), ('smote', 'SMOTE')],
        widget=forms.RadioSelect,
        label="Método de Balanceamento para o Classificador",
        initial='smote'
    )
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Revertido para IntegerField, que é o correto para sua necessidade.
    # O IntegerField garante que o valor seja tratado como um número inteiro.
    smote_k_neighbors = forms.IntegerField(
        label="Número de Vizinhos (k) para o SMOTE", 
        initial=3, 
        min_value=1,
        help_text="Digite o número de vizinhos a serem considerados."
    )
    # --------------------------------
    
    feature_selection_method = forms.ChoiceField(
        label="Método de Seleção de Features", 
        choices=[('none', 'Nenhuma'), ('genetic_algorithm', 'Algoritmo Genético')],
        widget=forms.RadioSelect, 
        initial='none'
    )
    
    k_folds = forms.IntegerField(label="Dobras (K-Folds)", initial=10)
    n_repeats = forms.IntegerField(label="Repetições da Validação", initial=10)
    
    CLASSIFIER_CHOICES = [('random_forest', 'Random Forest'), ('svm', 'SVM'), ('mlp', 'MLP')]
    final_classifier = forms.ChoiceField(choices=CLASSIFIER_CHOICES, label="Algoritmo de Classificação Final", initial='random_forest', widget=forms.Select(attrs={'id': 'id_final_classifier'}))
    
    rf_n_estimators = forms.IntegerField(label="Nº de Árvores (RF)", initial=100)
    rf_max_depth = forms.IntegerField(label="Profundidade Máx. (RF)", required=False, help_text="Deixe em branco ou 0 para sem limite.")
    svm_c = forms.FloatField(label="Parâmetro C (SVM)", initial=1.0)
    svm_kernel = forms.ChoiceField(label="Kernel (SVM)", choices=[('rbf', 'RBF'), ('linear', 'Linear'), ('poly', 'Polinomial')], initial='rbf')
    mlp_hidden_layer_sizes = forms.CharField(label="Camadas Ocultas (MLP)", initial="100,", help_text="Ex: 100,50,25")
    mlp_max_iter = forms.IntegerField(label="Nº Máx. de Iterações (MLP)", initial=200)

class ResultsFilterForm(forms.Form):
    cenario_clinico = forms.ChoiceField(
        label="1. Selecione o Cenário Clínico", 
        choices=[], 
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    final_classifier = forms.ChoiceField(
        label="2. Selecione o Classificador para Análise", 
        choices=TrainingForm.CLASSIFIER_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cenarios = ModeloML.objects.values_list('cenario_clinico', flat=True).distinct().order_by('cenario_clinico')
        self.fields['cenario_clinico'].choices = [('', '--- Selecione um Cenário ---')] + [(cenario, cenario) for cenario in cenarios if cenario]