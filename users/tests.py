from django.test import TestCase

# users/tests.py
from django.test import TestCase
from django.contrib.auth.models import User
from .models import Medico, Paciente

class UserModelTest(TestCase):

    def setUp(self):
        """Cria os objetos base para os testes."""
        # Cria um usuário para ser o médico
        self.user_medico = User.objects.create_user(
            username='dr.maria',
            password='test123',
            first_name='Maria',
            last_name='Silva'
        )
        self.medico = Medico.objects.create(
            user=self.user_medico,
            crm='12345-PE'
        )

        # Cria um usuário para ser o paciente
        self.user_paciente = User.objects.create_user(
            username='paciente.joao',
            password='test456',
            first_name='João',
            last_name='Paz'
        )

    def test_criação_perfis(self):
        """
        Testa se a criação de um Paciente e sua associação com um Médico funciona.
        """
        # Cria o paciente associado ao médico criado no setUp
        paciente = Paciente.objects.create(
            user=self.user_paciente,
            medico_responsavel=self.medico
        )

        # Verifica se os objetos foram criados
        self.assertEqual(User.objects.count(), 2)
        self.assertEqual(Medico.objects.count(), 1)
        self.assertEqual(Paciente.objects.count(), 1)

        # Verifica a associação correta
        self.assertEqual(paciente.medico_responsavel, self.medico)
        self.assertEqual(self.medico.pacientes.count(), 1)
        self.assertEqual(str(self.medico), "Maria Silva")
        self.assertEqual(str(paciente), "João Paz")