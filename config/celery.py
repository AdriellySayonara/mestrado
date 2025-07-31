# config/celery.py
import os
from celery import Celery

# Define o módulo de configurações do Django para o 'celery'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Cria a instância do Celery
app = Celery('config')

# Usa as configurações do Django (prefixo 'CELERY_')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Carrega automaticamente os arquivos tasks.py de todos os apps registrados
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')