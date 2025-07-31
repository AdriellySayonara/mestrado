# core/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def int_to_char(value):
    """Converte um inteiro (0-25) para um caractere (A-Z)."""
    try:
        return chr(65 + int(value))
    except (ValueError, TypeError):
        return ''

@register.filter
def get_item(dictionary, key):
    """Permite acessar um item de dicionário com uma chave variável no template."""
    return dictionary.get(key)