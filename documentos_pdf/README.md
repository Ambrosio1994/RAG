# 🦜🔗 Converse com seus Documentos 📚

Uma aplicação interativa construída com **Streamlit** que permite aos usuários carregar documentos PDF e fazer perguntas sobre o conteúdo. O sistema utiliza técnicas avançadas de **Retrieval-Augmented Generation (RAG)** para encontrar informações nos documentos e responder às perguntas usando modelos de linguagem como **Anthropic Claude**.

---

## 📜 Descrição

Este projeto integra várias bibliotecas do ecossistema **LangChain** para oferecer uma experiência poderosa de consulta a documentos:

1. **Carregamento de Documentos**: Suporte para PDFs com divisão em chunks para processamento eficiente.
2. **Geração de Respostas**: Respostas baseadas no conteúdo dos documentos e enriquecidas por um modelo de linguagem.
3. **Histórico de Conversas**: Um chat interativo que mantém o contexto da conversa.
4. **Fonte de Respostas**: Exibição das fontes utilizadas para gerar as respostas.

---

## ✨ Funcionalidades

- **Carregamento de PDFs**: Envie múltiplos arquivos PDF para análise.
- **Busca Contextual**: Localiza informações relevantes nos documentos.
- **Geração de Respostas**: Responde às perguntas em português usando o modelo **Anthropic Claude**.
- **Personalização de Modelo**: Ajuste da temperatura do modelo para controlar a criatividade das respostas.
- **Histórico de Conversas**: Mantenha o contexto durante toda a interação.
- **Exibição de Fontes**: Links para as partes do documento que sustentam as respostas geradas.

---

## 📋 Requisitos

Certifique-se de ter os seguintes itens configurados:

- **Python 3.9** ou superior.
- **Bibliotecas Python**:
  - `langchain-core`, `langchain-community`, `langchain-anthropic`
  - `streamlit`, `python-dotenv`, `faiss-cpu`
  - `huggingface_hub`, `PyPDF2`

Dependências adicionais estão listadas no arquivo `requirements.txt`.

---
