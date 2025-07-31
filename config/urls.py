# config/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('', include('users.urls')),
    path('', include('laudos.urls')),
    path('', include('ml_manager.urls')),
]

# --- BLOCO CORRIGIDO ---

# A URL para os arquivos de MÍDIA (uploads) é adicionada apenas em modo de desenvolvimento.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# A URL para os arquivos ESTÁTICOS (para a pasta 'static' raiz) é adicionada fora do if.
# Isso garante que o Django Development Server possa encontrá-los.
# Em produção, um servidor web como Nginx geralmente lida com isso.
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
# ---------------------