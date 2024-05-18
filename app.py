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
# GROQ_API_KEY = "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"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.title("GROQBoT")

# Add a text input for the user to input the URL
website_url = st.text_input("Enter the URL of the website:")

if website_url:
    if "vectors" not in st.session_state:
        st.session_state.vectors_initialized = False

    if "vectors_initialized" in st.session_state and not st.session_state.vectors_initialized:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = WebBaseLoader(website_url)
        # st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    llm = ChatGroq(groq_api_key= GROQ_API_KEY, model_name="llama3-8b-8192")

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
    st.write("Please enter a valid URL.")