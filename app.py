import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

# Load groq api key
# Add a text input for the user to input the Groq API key
groq_api_key = st.text_input("Enter your Groq API Key:", type="password")

st.title("GROQBoT")

# Add a text input for the user to input the URL
website_url = st.text_input("Enter the URL of the website:")

if website_url:
    if "vectors" not in st.session_state:
        st.session_state.vectors_initialized = False

    if "vectors_initialized" in st.session_state and not st.session_state.vectors_initialized:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = WebBaseLoader(website_url)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    llm = ChatGroq(groq_api_key= groq_api_key, model_name="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate.from_template(
        """
    Answer questions based only on the context provided. Answer the question in as much detail as possible. 
    <context> {context} </context>
    "Question": {input}
    """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    prompt = st.text_input("Ask me anything!")

    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt})
        print(f"Time taken to generate response: {time.process_time() - start}")
        st.write(response['answer'])

        # Add Streamlit Expander
        with st.expander("Document Similarity Search"):
            # FInd relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-------------------------")
else:
    if not groq_api_key:
        st.write("Please enter your Groq API Key.")
    if not website_url:
        st.write("Please enter a valid URL.")
