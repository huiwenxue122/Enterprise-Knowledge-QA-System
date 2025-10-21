import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Initialize OpenAI
llm = OpenAI(temperature=0.2)
embeddings = OpenAIEmbeddings()

# Streamlit UI
st.set_page_config(page_title="Enterprise Knowledge QA", page_icon="🏢")
st.title("🏢 Enterprise Knowledge QA System")
st.write("Upload one or more PDF documents and ask any question based on their content.")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            # Temporarily save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load_and_split()
            all_docs.extend(docs)

        # Create FAISS vector store (persistent)
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local("faiss_index")

    st.success("✅ Documents processed and indexed!")

    # Initialize retriever
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Ask question
    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
            st.markdown(f"### 💬 Answer:\n{answer}")

            # Perform similarity search for citation
            docs = vectorstore.similarity_search(query, k=2)
            st.markdown("### 📚 Sources:")
            for i, d in enumerate(docs):
                source_info = d.metadata.get("page", "N/A")
                st.write(f"**Source {i+1}:** Page {source_info}")
                st.caption(d.page_content[:300] + "...")
else:
    st.info("👆 Please upload one or more PDF documents to begin.")
