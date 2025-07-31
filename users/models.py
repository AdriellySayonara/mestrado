# users/models.py
from django.db import models
from django.contrib.auth.models import User

# Modelo para o perfil do Médico
class Medico(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    crm = models.CharField(max_length=20, unique=True, verbose_name="CRM")
    # verbose_name é o nome que aparecerá no admin do Django

    # Esta função faz o objeto ser exibido com um nome legível
    def __str__(self):
        return self.user.get_full_name() or self.user.username

# Modelo para o perfil do Paciente
class Paciente(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    medico_responsavel = models.ForeignKey(Medico, on_delete=models.SET_NULL, null=True, blank=True, related_name='pacientes')
    # Outros campos como data de nascimento, telefone, podem ser adicionados aqui
    # Ex: data_nascimento = models.DateField(null=True, blank=True)

    def __str__(self):
        return self.user.get_full_name() or self.user.username