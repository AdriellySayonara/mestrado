# config/__init__.py

# Isso garantirá que o app Celery seja sempre importado quando
# o Django iniciar para que as tarefas compartilhadas (@shared_task) o usem.
from .celery import app as celery_app

__all__ = ('celery_app',)