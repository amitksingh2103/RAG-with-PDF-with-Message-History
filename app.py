import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

groq_api=os.getenv('GROQ_API_KEY)

st.title("🧠 Conversation RAG with PDF + Message History (GROQ)")

model = ChatGroq(model_name='llama-3.1-8b-instant', groq_api_key=groq_api)

    # Session ID
session_id = st.text_input("Session ID:", value='default-session')

    # Initialize session store
if 'store' not in st.session_state:
    st.session_state.store = {}

    # --- PDF Upload ---
uploaded_file = st.file_uploader("📄 Upload your PDF file", type='pdf')
if uploaded_file:
    temp_pdf = "./temp.pdf"
    with open(temp_pdf, 'wb') as f:
        f.write(uploaded_file.getvalue())

        # Load PDF and split into chunks
    loader = PyPDFLoader(temp_pdf)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)

        # Create or load vector store
    persist_dir = './chroma_data'
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
    else:
        vector = Chroma.from_documents(
            splits,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_dir
        )

    retriever = vector.as_retriever()

    contextualize_system_prompt = """Given a chat history and the latest user question,
    reformulate the question to make it standalone. Do not answer the question, only rephrase it."""
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ('system', contextualize_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_prompt)


    system_prompt = """You are a helpful AI assistant.
    Use only the provided context to answer the question.
    If the context does not contain the answer, say "I don't know."
    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    document_chain = create_stuff_documents_chain(model, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # --- Session History Handler ---
    def get_session_history(session_id):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
        )

        # --- User Input ---
    user_input = st.text_input("💬 Ask a question about your PDF:")
    if user_input:
        session_history = get_session_history(session_id)
        with st.spinner("Generating answer..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

        st.subheader("🤖 Assistant Answer:")
        st.write(response['answer'])

        with st.expander("🧾 Chat History"):
            st.write(session_history.messages)


