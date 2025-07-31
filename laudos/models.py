# laudos/models.py
from django.db import models
from users.models import Paciente, Medico

class ModeloML(models.Model):
    # --- Identificação e Arquivo ---
    nome_versao = models.CharField(max_length=100, unique=True)
    cenario_clinico = models.CharField(max_length=255, blank=True, null=True, help_text="Descrição do cenário de diagnóstico.")
    arquivo_modelo = models.FileField(upload_to='ml_models/')
    data_upload = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False, help_text="Define se este é o modelo em produção.")
    parametros_treinamento = models.JSONField(null=True, blank=True, help_text="Parâmetros do formulário usados neste treinamento")
    
    # --- NOVO CAMPO PARA MAPEAMENTO DE CLASSES ---
    class_map_json = models.JSONField(null=True, blank=True, help_text="Mapeamento de índice para nome de classe. Ex: {'0': 'caes', '1': 'gatos'}")

    # --- Métricas Estatísticas ---
    acuracia_media = models.FloatField(null=True, blank=True)
    acuracia_std = models.FloatField(null=True, blank=True)
    sensibilidade_media = models.FloatField(null=True, blank=True)
    sensibilidade_std = models.FloatField(null=True, blank=True)
    especificidade_media = models.FloatField(null=True, blank=True)
    especificidade_std = models.FloatField(null=True, blank=True)
    auc_roc_media = models.FloatField(null=True, blank=True)
    auc_roc_std = models.FloatField(null=True, blank=True)
    kappa_media = models.FloatField(null=True, blank=True)
    kappa_std = models.FloatField(null=True, blank=True)
    tempo_treinamento_seg = models.FloatField(null=True, blank=True)
    
    # --- Artefatos ---
    metricas_raw = models.JSONField(null=True, blank=True)
    matriz_confusao_img = models.ImageField(upload_to='reports/confusion_matrix/', null=True, blank=True)
    boxplot_acuracia_img = models.ImageField(upload_to='reports/boxplots/', null=True, blank=True)
    boxplot_sensibilidade_img = models.ImageField(upload_to='reports/boxplots/', null=True, blank=True)
    boxplot_especificidade_img = models.ImageField(upload_to='reports/boxplots/', null=True, blank=True)
    boxplot_auc_roc_img = models.ImageField(upload_to='reports/boxplots/', null=True, blank=True)
    boxplot_kappa_img = models.ImageField(upload_to='reports/boxplots/', null=True, blank=True)
    importancia_features_img = models.ImageField(upload_to='reports/feature_importance/', null=True, blank=True)
    top_features_json = models.JSONField(null=True, blank=True)

    def __str__(self):
        status = "Ativo" if self.is_active else "Inativo"
        return f"{self.nome_versao} ({status})"


class Laudo(models.Model):
    paciente = models.ForeignKey(Paciente, on_delete=models.CASCADE)
    medico = models.ForeignKey(Medico, on_delete=models.PROTECT)
    imagem_lesao = models.ImageField(upload_to='lesoes/%Y/%m/%d/')
    resultado_classificacao = models.CharField(max_length=100, blank=True, null=True)
    confianca = models.FloatField(null=True, blank=True)
    data_laudo = models.DateTimeField(auto_now_add=True)
    modelo_utilizado = models.ForeignKey(ModeloML, on_delete=models.PROTECT, null=True, blank=True)
    perda_sensibilidade_termica = models.BooleanField(default=False)
    apresenta_nodulos = models.BooleanField(default=False)
    observacoes_clinicas = models.TextField(blank=True, null=True)
    grad_cam_img = models.ImageField(upload_to='reports/grad_cam/', null=True, blank=True)

    def __str__(self):
        return f"Laudo para {self.paciente} em {self.data_laudo.strftime('%d/%m/%Y')}"