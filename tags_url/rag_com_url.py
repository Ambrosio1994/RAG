from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic

import os
import streamlit as st

st.set_page_config(page_title="RAG com URLs",
                    page_icon="🦜")

st.title("Pesquise conteudos dentro de uma pagina web 🌐")

# Configurando as chaves de API
st.sidebar.title("Permissoes")
ANTHROPIC_API_KEY = st.sidebar.text_input("Digite sua chave ANTHROPIC:", type="password")
HUGGINGFACEHUB_API_TOKEN = st.sidebar.text_input("Digite sua chave HUGGING FACE HUB:", type="password")

# Verificando se as chaves foram inseridas
if not ANTHROPIC_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    st.warning("Por favor, insira suas chaves de API.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

def get_retriever(URL: str):
    # Carregar documentos da URL
    loader = UnstructuredURLLoader([URL],
                                   show_progress_bar=True)
    documents = loader.load()
    
    # Dividir documentos em partes menores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Criar embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # Criar um banco de dados FAISS a partir dos documentos
    db = FAISS.from_documents(docs, embeddings)
    
    # Criar um retriever a partir do banco de dados
    retriever = db.as_retriever()

    return retriever

def response(query: str, retriever, llm):

    context = retriever.invoke(query)
    # Definindo o prompt do sistema
    system_prompt = f"""
    Voce e um assistente de IA projetado para responder perguntas com 
    base no contexto fornecido. Sua tarefa  ler o contexto fornecido 
    cuidadosamente e usa-lo para responder  consulta do usu rio.

    Aqui esta o contexto que voce deve usar para formular sua resposta:
    {context}

    A pergunta do usuario:
    {query}

    Siga estas diretrizes ao formular sua resposta:
    1. Leia e entenda cuidadosamente o contexto e a consulta.
    2. Use apenas as informa es fornecidas no contexto para responder  consulta.
    3. Se a resposta  consulta n o for encontrada no contexto, ou se voce nao tiver certeza, simplesmente diga que nao sabe.
    4. Mantenha sua resposta concisa e direta.
    5. Nao inclua nenhuma informa o que nao seja diretamente relevante para responder  consulta.
    6. Nao invente ou infira informacoes que nao estejam explicitamente declaradas no contexto.

    Lembre-se de responder em portugues.
    """

    # Invocando o modelo com o prompt
    response = llm.invoke([SystemMessage(content=system_prompt),
                            HumanMessage(content=query)])
    
    # Retornando o conte do da resposta
    return response.content if hasattr(response, 'content') else response

def clear_chat_history():
    st.session_state.chat_history = [{"role": "assistant", "content": msg_init}]

msg_init = "Sou seu assistente virtual. Como posso ajudar?"

URL = st.sidebar.text_input("Cole a URL que ser  usada para consulta")
if not URL:
    st.warning("Por favor, insira uma URL.")
    st.stop()

st.sidebar.title("Configura es do modelo")
model_names = [
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]

model = st.sidebar.selectbox("Modelo", model_names)

temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.1,
                                help="Controla a criatividade da resposta do modelo")

max_tokens = st.sidebar.slider('max_tokens', min_value=500, max_value=8000, value=1000, step=10,
                               help="Quantidade de tokens a serem gerados pela resposta")

llm = ChatAnthropic(model=model,
                    temperature=temperature,
                    max_tokens=max_tokens)

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=msg_init)]

if "retriever" not in st.session_state:
    st.session_state.retriever = get_retriever(URL)

new_URL = st.sidebar.text_input("Cole a nova URL que ser  usada para consulta")
if new_URL:
    st.session_state.retriever = get_retriever(new_URL)

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="🤔"):
            st.write(message.content)

user_query = st.chat_input(msg_init)

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar="🤔"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="🤖"):
        resp = response(user_query, retriever=st.session_state.retriever, llm=llm)

        st.write(resp)

    st.session_state.chat_history.append(AIMessage(content=resp))
