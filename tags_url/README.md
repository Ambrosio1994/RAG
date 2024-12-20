# 🌐 RAG com URLs - Pesquisa e Respostas Contextuais

Uma aplicação interativa construída com **Streamlit** que permite aos usuários consultar informações de páginas web usando técnicas de **RAG (Retrieval-Augmented Generation)**. A ferramenta combina recuperação de dados contextuais e modelos de linguagem para fornecer respostas baseadas no conteúdo extraído de URLs.

---

## 📜 Descrição

Este projeto facilita a interação com informações encontradas em páginas web. Ele utiliza:
- **LangChain Community**: Para carregar e dividir o conteúdo das URLs em partes processáveis.
- **FAISS**: Para criar uma base de dados vetorial que armazena as representações dos textos.
- **Anthropic Claude**: Para geração de respostas baseadas nos dados recuperados.

A aplicação é ideal para usuários que desejam pesquisar e consultar conteúdo diretamente de páginas web, com respostas concisas e baseadas no contexto.

---

## ✨ Funcionalidades

- **Consulta a Páginas Web**: Insira uma URL e faça perguntas sobre o conteúdo dela.
- **Divisão e Indexação de Texto**: Divida o conteúdo da página em chunks processáveis para recuperação eficiente.
- **Modelos Personalizáveis**: Escolha entre diferentes modelos da família **Claude**.
- **Configurações de Modelo**: Ajuste parâmetros como **temperatura** e **máximo de tokens** para personalizar as respostas.
- **Histórico de Conversa**: Mantenha o contexto com o histórico das interações no chat.

---

## 📋 Requisitos

Certifique-se de ter os seguintes itens configurados:

- **Python 3.9** ou superior.
- **Bibliotecas Python**:
  - `streamlit`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-anthropic`
  - `faiss-cpu`, `huggingface_hub`
  - `python-dotenv`

---