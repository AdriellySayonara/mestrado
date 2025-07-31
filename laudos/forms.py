# laudos/forms.py
from django import forms
from .models import Laudo
from users.models import Paciente

class LaudoForm(forms.ModelForm):
    # --- CORREÇÃO APLICADA AQUI ---
    # Definimos o campo paciente aqui para ter mais controle
    paciente = forms.ModelChoiceField(
        queryset=Paciente.objects.none(), # Começa com um queryset vazio
        label="Selecione o Paciente",
        empty_label="--- Selecione um paciente ---"
    )
    # -----------------------------

    class Meta:
        model = Laudo
        fields = [
            'paciente', 
            'imagem_lesao', 
            'perda_sensibilidade_termica',
            'apresenta_nodulos',
            'observacoes_clinicas'
        ]
        # Removemos os labels daqui pois já definimos no campo acima
    
    def __init__(self, *args, **kwargs):
        # Pega o médico passado pela view ANTES de chamar o super().__init__
        medico = kwargs.pop('medico', None) 
        super().__init__(*args, **kwargs)

        # --- LÓGICA DE FILTRAGEM CORRIGIDA ---
        # Agora que o formulário foi inicializado, nós atualizamos o queryset
        if medico:
            self.fields['paciente'].queryset = Paciente.objects.filter(medico_responsavel=medico)
        # -------------------------------------
        
        # Estilização dos checkboxes
        self.fields['perda_sensibilidade_termica'].widget.attrs.update({'class': 'form-check-input'})
        self.fields['apresenta_nodulos'].widget.attrs.update({'class': 'form-check-input'})