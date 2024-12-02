import os

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS 

from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

import streamlit as st

st.title("Respostas contextualizadas sobre qualquer vídeo do YouTube 📽️")

ANTHROPIC_API_KEY = st.sidebar.text_input("Digite sua chave ANTHROPIC:", type="password")
HUGGINGFACEHUB_API_TOKEN = st.sidebar.text_input("Digite sua chave HUGGING FACE HUB:", type="password")

if not ANTHROPIC_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    st.info("Por favor, insira suas chaves ANTHROPIC e HUGGINGFACEHUB para continuar.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

model = "claude-3-5-sonnet-20241022"
temperature = 0.4
max_tokens = 4000

def load_video(url_video, language=["pt", "pt-BR", "en"], translation="pt"):
  """
  Carrega o video da URL especificada e retorna o conteudo do transcript
  em portugues, com opcoes de idioma e traducao.

  Args:
    url_video (str): URL do video do YouTube.
    language (list[str], optional): Idiomas a serem pesquisados. Defaults to ["pt", "pt-BR", "en"].
    translation (str, optional): Idioma de traducao. Defaults to "pt".

  Returns:
    str: Conteudo do transcript do video em portugues.
  """
  video_loader = YoutubeLoader.from_youtube_url(
      url_video,
      language=language,
      translation=translation,
      )
  
  docs = video_loader.load()

#   transcript = docs[0].page_content

  embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

  vector_store = FAISS.from_documents(docs, embeddings)

  return vector_store

def get_response(query, vector_store, llm):
    """
    Recupera a resposta para uma consulta com base no conteúdo do transcript de um video.

    Args:
        query (str): Consulta do usuário.
        vector_store (FAISS): Armazenamento de vetores de documentos relacionados ao video.
        llm (LLM): Modelo de linguagem a ser usado para gerar a resposta.

    Returns:
        str: Resposta para a consulta do usuário com base no conteúdo do transcript do video.
    """
    
    # Recuperar documentos relevantes
    relevant_docs = vector_store.similarity_search(query, k=5)

    # Concatenar os conteúdos dos documentos relevantes
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Criar o template de prompt
    template = f"""
        Você é um assistente virtual encarregado de executar uma tarefa 
        solicitada pelo usuário com base nas informações a seguir. 
        Aqui está o contexto relevante:
        {context}
        
        Aqui está a tarefa que o usuário deseja que você execute:
        {query}
    """

    prompt_template = PromptTemplate(input_variables=["context", "query"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")
    chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

    response = chain.invoke({"context": context, "query": query})
    return response["text"]

llm = ChatAnthropic(model=model,
                      temperature=temperature,
                      max_tokens=max_tokens)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Digite sua dúvida aqui...")]

url_video = st.sidebar.text_input("Cole aqui a URL do video")

if not url_video:
    st.warning("Por favor, insira a URL do video.")
    st.stop()

vector_store = load_video(url_video)

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="📽️"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="🤔"):
            st.write(message.content)

user_query = st.chat_input("Digite sua dúvida aqui...", max_chars=500)
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human", avatar="🤔"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="📽️"):
        with st.spinner("Pensando..."):    
          response = get_response(user_query, vector_store, llm)
        st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))