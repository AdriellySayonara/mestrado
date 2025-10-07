# laudos/tests.py
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from users.models import Medico, Paciente
from .models import Laudo # Assumindo que o modelo Laudo existe em laudos.models

class LaudoViewTest(TestCase):

    def setUp(self):
        """Configuração inicial para os testes de laudos."""
        self.client = Client()
        self.user_medico = User.objects.create_user(
            username='dramaria', password='123456', first_name='Maria'
        )
        self.medico = Medico.objects.create(user=self.user_medico, crm='12345-PE')

        self.user_paciente = User.objects.create_user(
            username='paciente', password='123456', first_name='Carlos'
        )
        self.paciente = Paciente.objects.create(
            user=self.user_paciente,
            medico_responsavel=self.medico
        )
        
        # URL do Dashboard do Médico (que também processa a criação do laudo)
        self.dashboard_url = reverse('medico_dashboard')

    def test_medico_dashboard_acesso_negado_sem_login(self):
        """
        Garante que um usuário não logado é redirecionado da página do dashboard.
        """
        response = self.client.get(self.dashboard_url)
        # O status 302 indica um redirecionamento
        self.assertEqual(response.status_code, 302)
        # Verifica se redirecionou para a página de login
        self.assertRedirects(response, f'/accounts/login/?next={self.dashboard_url}')

    def test_medico_pode_criar_laudo(self):
        """
        Verifica se um médico logado pode submeter o formulário e criar um laudo.
        """
        # Força o login do médico
        self.client.login(username='dramaria', password='123456')

        # Cria um arquivo de imagem em memória para o teste
        dummy_image = SimpleUploadedFile(
            "lesao.jpg",
            b"file_content", # Conteúdo do arquivo em bytes
            content_type="image/jpeg"
        )
        
        form_data = {
            'paciente': self.paciente.id,
            'imagem_lesao': dummy_image,
        }

        # Submete os dados via POST
        response = self.client.post(self.dashboard_url, data=form_data)

        # Verifica se o laudo foi criado no banco
        self.assertEqual(Laudo.objects.count(), 1)
        laudo_criado = Laudo.objects.first()
        self.assertEqual(laudo_criado.paciente, self.paciente)

        # Verifica se o usuário foi redirecionado após o sucesso
        self.assertEqual(response.status_code, 302)