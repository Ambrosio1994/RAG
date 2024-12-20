# 📽️ Pergunte Algo Sobre um Vídeo do YouTube

Este projeto permite que os usuários façam perguntas sobre vídeos do YouTube, analisando automaticamente o conteúdo do vídeo e gerando respostas contextuais usando técnicas avançadas de **Processamento de Linguagem Natural (NLP)**. A aplicação é construída com **Streamlit** e combina modelos de linguagem poderosos com um sistema de recuperação baseado em vetores.

---

## 📜 Descrição

Com este aplicativo, você pode:
1. **Carregar vídeos do YouTube**: Extraia automaticamente o conteúdo de transcrições do vídeo.
2. **Gerar Respostas Inteligentes**: Faça perguntas específicas sobre o vídeo, e o sistema usará o modelo **Anthropic Claude** para fornecer respostas precisas e baseadas no conteúdo.
3. **Histórico de Conversas**: Mantenha um histórico de perguntas e respostas durante a interação.

A ferramenta é ideal para resumir, explorar ou obter informações de vídeos longos, economizando tempo e esforço.

---

## ✨ Funcionalidades

- **Análise de Vídeos do YouTube**: Insira a URL de um vídeo e o conteúdo será extraído automaticamente.
- **Respostas Contextuais**: Pergunte sobre tópicos específicos do vídeo e receba respostas baseadas na transcrição.
- **Configurações Personalizáveis**: Ajuste os parâmetros do modelo, como temperatura e número máximo de tokens.
- **Histórico de Conversas**: Interaja com o sistema de forma contínua e contextualizada.

---

## 📋 Requisitos

Certifique-se de ter os seguintes itens configurados:

- **Python 3.9** ou superior.
- **Bibliotecas Python**:
  - `streamlit`, `langchain-core`, `langchain-community`, `langchain-anthropic`
  - `langchain-huggingface`, `faiss-cpu`
  - `python-dotenv`

As dependências completas estão listadas no arquivo `requirements.txt`.

---