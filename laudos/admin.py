# laudos/admin.py
from django.contrib import admin, messages
from .models import ModeloML, Laudo

@admin.register(ModeloML)
class ModeloMLAdmin(admin.ModelAdmin):
    list_display = ('nome_versao', 'cenario_clinico', 'is_active', 'acuracia_media', 'auc_roc_media', 'data_upload')
    list_filter = ('is_active', 'cenario_clinico')
    readonly_fields = ('data_upload', 'parametros_treinamento', 'acuracia_media', 'acuracia_std', 'sensibilidade_media', 'sensibilidade_std', 'especificidade_media', 'especificidade_std', 'auc_roc_media', 'auc_roc_std', 'kappa_media', 'kappa_std', 'tempo_treinamento_seg', 'metricas_raw', 'matriz_confusao_img', 'boxplot_acuracia_img', 'boxplot_sensibilidade_img', 'boxplot_especificidade_img', 'boxplot_auc_roc_img', 'boxplot_kappa_img', 'importancia_features_img', 'top_features_json')
    
    # --- AÇÃO CUSTOMIZADA NO LUGAR CORRETO ---
    actions = ['ativar_modelo_selecionado']

    @admin.action(description='Ativar modelo selecionado para produção')
    def ativar_modelo_selecionado(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Por favor, selecione apenas UM modelo para ativar.", messages.ERROR)
            return
        
        modelo_para_ativar = queryset.first()
        ModeloML.objects.exclude(pk=modelo_para_ativar.pk).update(is_active=False)
        modelo_para_ativar.is_active = True
        modelo_para_ativar.save()
        self.message_user(request, f"O modelo '{modelo_para_ativar.nome_versao}' foi ativado com sucesso.", messages.SUCCESS)
    # ---------------------------------------------

@admin.register(Laudo)
class LaudoAdmin(admin.ModelAdmin):
    list_display = ('paciente', 'medico', 'data_laudo', 'resultado_classificacao', 'modelo_utilizado')
    list_filter = ('medico', 'resultado_classificacao', 'modelo_utilizado')
    # A ação customizada foi removida daqui