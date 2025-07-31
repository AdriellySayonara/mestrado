# users/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.db import transaction # Importa transaction para garantir a atomicidade

from .models import Medico, Paciente

class MedicoSignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, help_text='Obrigatório.')
    last_name = forms.CharField(max_length=30, required=True, help_text='Obrigatório.')
    email = forms.EmailField(max_length=254, required=True, help_text='Obrigatório.')
    crm = forms.CharField(max_length=20, required=True, help_text='Obrigatório.')

    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('first_name', 'last_name', 'email')

    @transaction.atomic # Garante que ou tudo é salvo, ou nada é
    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            Medico.objects.create(user=user, crm=self.cleaned_data['crm'])
        return user

# --- NOVO FORMULÁRIO PARA CADASTRO DE PACIENTE ---
class PacienteSignUpForm(forms.ModelForm):
    # Campos do modelo User que queremos no formulário
    first_name = forms.CharField(label='Nome', max_length=30, required=True)
    last_name = forms.CharField(label='Sobrenome', max_length=30, required=True)
    email = forms.EmailField(label='Email', max_length=254, required=True, help_text="Será usado como nome de usuário para login.")
    
    class Meta:
        model = Paciente
        # O formulário não precisa do campo 'medico_responsavel', pois ele será
        # preenchido automaticamente pela view com o médico que está logado.
        fields = [] 

    @transaction.atomic
    def save(self, commit=True, medico=None):
        if not medico:
            raise ValueError("Um médico responsável é necessário para cadastrar um paciente.")

        # Cria um objeto User para o novo paciente
        # Usamos o email como username e uma senha padrão (o paciente pode mudar depois)
        user = User.objects.create_user(
            username=self.cleaned_data['email'],
            email=self.cleaned_data['email'],
            first_name=self.cleaned_data['first_name'],
            last_name=self.cleaned_data['last_name'],
            password='senha_provisoria_123' # IMPORTANTE: Em um sistema real, essa senha deveria ser gerada aleatoriamente e enviada por email.
        )

        # Cria o perfil do Paciente e o associa ao User e ao Médico responsável
        paciente = super().save(commit=False)
        paciente.user = user
        paciente.medico_responsavel = medico
        
        if commit:
            paciente.save()
        
        return paciente