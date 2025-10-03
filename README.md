# Projeto de Mestrado
# Sistema Inteligente de Apoio ao Diagnóstico de Lesões de Pele

Este projeto é uma plataforma web desenvolvida em Django, um Produto Mínimo Viável ou Prova de Conceito, projetada como uma ferramenta de identificação de hanseníase e outras lesões de pele por meio de imagens dessas lesões. Dentro do projeto, também foram criados módulo para o treinamento, avaliação e análise comparativa de pipelines de Machine Learning para a classificação de imagens de lesões de pele. O sistema foi desenvolvido como artefato durante o programa de mestrado e possui um módulo de Análise e identificação de lesões, um módulo de experimentação e criação de modelos e um Painel de Visualização de resultados de performance dos modelos, com gráficos box-plots. O sistema também implementa algumas técnicas de pré-processamento de imagens, assim como balanceamento de dados e otimização com algoritmos evolucionários.

## Funcionalidades Principais

- **Interfaces de Usuário:** Painéis dedicados para Médicos e Pacientes.
- **Fábrica de Modelos:** Uma interface de pesquisa para configurar e treinar múltiplos modelos de IA com parâmetros customizáveis.
- **Pipelines Avançados:** Suporte para extração de features clássicas e de Deep Learning, SMOTE, cGANs, e Algoritmos Genéticos.
- **Análise de Resultados:** Um painel consolidado para comparar a performance de diferentes experimentos através de métricas estatísticas e gráficos de box plot.
- **IA Explicável (XAI):** Geração de gráficos de importância de features e mapas de calor Grad-CAM.
- **Laudos em PDF:** Geração automática de laudos para pacientes.

## Pré-requisitos

Antes de começar, garanta que você tem os seguintes softwares instalados em seu sistema:

- **Python:** Versão 3.11.x
- **Git:** Para clonar o repositório.
- *No Windows, você pode instalar a partir do [release oficial da Microsoft](https://github.com/microsoftarchive/redis/releases)
- - **GTK+ for Windows (Apenas para Windows):** Esta é uma dependência de sistema **externa** ao Python, necessária para a biblioteca `WeasyPrint` (geração de PDF) funcionar.
  - 1. Faça o download do instalador em: [GTK+ for Windows Runtime Environment](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).
  - **2.** Execute o instalador e siga os passos.

  ## Guia de Instalação

Siga os passos abaixo para configurar o ambiente de desenvolvimento e rodar o projeto localmente.

### Clonar o Repositório

Abra seu terminal e clone o repositório do GitHub:
```bash
git clone [URL_DO_SEU_REPOSITÓRIO_AQUI]
cd [NOME_DA_PASTA_DO_PROJETO]

# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente virtual
# No Windows:
venv\Scripts\activate

### Instalar as dependências

pip install -r requirements.txt

### Configurar o Banco de Dados
python manage.py migrate

### Criar um Superusuário (Administrador)
python manage.py createsuperuser

## **Passo 4: Como Executar o Projeto**

```markdown
## Executando o Projeto

Para rodar o sistema completo, você precisará de **dois terminais separados**, ambos com o ambiente virtual ativado.

### Terminal 1: Iniciar o Servidor Django

Este terminal irá rodar a aplicação web.

```bash
python manage.py runserver

### Terminal 2: Iniciar o Worker do Celery

### No Windows, use a flag '-P solo' para estabilidade
celery -A config worker --loglevel=info -P solo
Importante: Garanta que o seu serviço Redis esteja rodando em segundo plano antes de iniciar o worker do Celery.


## **Passo 5: Como Usar o Sistema**


```markdown
## Como Usar

1.  **Acesse o Admin:** Vá para `http://127.0.0.1:8000/admin/` e faça login com seu superusuário.
2.  **Cadastre um Médico e um Paciente:**
    -   Primeiro, crie os `Users` para um médico e um paciente.
    -   Depois, vá para as seções "Medicos" e "Pacientes" para criar os perfis correspondentes, associando o paciente ao médico.
3.  **Cadastre-se como Médico:** Alternativamente, acesse a página inicial e use a opção "Cadastrar como Médico".
4.  **Execute um Experimento:**
    -   Faça login como superusuário.
    -   Acesse o menu **"Fábrica de Modelos"**.
    -   Configure um experimento, faça o upload de um dataset (`.zip` com pastas por classe) e inicie o treinamento. Acompanhe o progresso no terminal do Celery.
5.  **Use o Sistema como Médico:**
    -   Faça login com a conta do médico.
    -   No seu painel, cadastre novos pacientes ou solicite a análise de uma imagem para um paciente existente.
    -   Acompanhe o histórico e clique em um laudo para ver os detalhes, incluindo a análise XAI (Grad-CAM) e a opção de baixar o PDF.
6. **Analise os Resultados:**
    -   Acesse o menu **"Painel de Resultados"**.
    -   Use os filtros para selecionar um cenário e um classificador e veja os gráficos comparativos.

## Estrutura do Projeto

-   `config/`: Configurações principais do Django (`settings.py`, `urls.py`).
-   `core/`: App para páginas e lógicas centrais (página inicial).
-   `users/`: App para gerenciamento de usuários (Médicos, Pacientes).
-   `laudos/`: App para a lógica de laudos, classificação em tempo real e visualizações.
-   `ml_manager/`: App para a "Fábrica de Modelos" e o "Painel de Resultados".
