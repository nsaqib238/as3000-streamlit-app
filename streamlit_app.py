
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key securely from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Load and process PDF
loader = PyPDFLoader("AS3000.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("AS3000 Code Assistant")
query = st.text_input("Ask your question about AS3000")

if query:
    answer = qa.run(query)
    st.write("### 🔍 Answer")
    st.write(answer)
