# 🏢 Enterprise Knowledge QA System

A Retrieval-Augmented Generation (RAG) demo powered by LangChain, OpenAI, and Streamlit.

This project demonstrates how to build an enterprise document question-answering system that allows users to query internal knowledge (e.g., company manuals, product guides, policies) directly using natural language.
It’s the foundation for a full Enterprise Knowledge QA Assistant.

## 🚀 Features

🧠 Retrieval-Augmented Generation (RAG): Retrieve context from documents before generating answers.

📄 PDF document ingestion and splitting via PyPDFLoader.

💾 Vector storage with Chroma (can later be swapped for FAISS).

💬 Natural language interface built with Streamlit.

🔑 Secure API key handling via .env.

## 🧰 Tech Stack

Python 3.11

LangChain

OpenAI (GPT-4o-mini or GPT-3.5-turbo)

Chroma as vector database

Streamlit for web UI

python-dotenv for environment variables

PyPDF for document parsing

## 📂 Project Structure
Enterprise-Knowledge-QA-System/
├── app.py
├── annualreport.pdf          # sample document
├── .env                      # your OpenAI key (DO NOT upload)
└── README.md

## ⚙️ Setup Instructions
1️⃣ Create and activate a virtual environment
# Go to your project folder
cd ~/LangchainDocuments

# Create a Python 3.11 virtual environment
/opt/homebrew/bin/python3.11 -m venv .lcenv

# Activate it
source .lcenv/bin/activate

# Check Python version (should be 3.11.x)
python -V

2️⃣ Install dependencies
pip install --upgrade pip setuptools wheel

pip install \
  "numpy==1.26.4" \
  langchain langchain-community langchain-openai \
  chromadb pypdf streamlit python-dotenv openai

3️⃣ Set your OpenAI API key

Create a .env file in the project root:

OPENAI_API_KEY=sk-your-real-key-here


⚠️ Never commit .env to GitHub. Add it to .gitignore.

4️⃣ Run the app

Use the virtual environment’s Python to ensure dependencies are correct:

python -m streamlit run app.py


Your browser will open automatically at:

http://localhost:8501

## 🧪 How to Use

Keep annualreport.pdf in the same folder (or replace it with your own PDF).

Enter a natural language question in the text box, e.g.

“Summarize the key findings of this report.”

“What does the company identify as its biggest risk?”

The model retrieves relevant text chunks and generates an answer.

Expand Document Similarity Search to view the most relevant passage retrieved.

To use your own file:
edit in app.py:

loader = PyPDFLoader('your_file.pdf')

## 🧱 Core Pipeline Overview
1. Load PDF  →  Split into pages/chunks
2. Embed text using OpenAIEmbeddings
3. Store embeddings in Chroma vector DB
4. User prompt triggers similarity search
5. Relevant chunks + prompt → GPT model
6. LLM generates contextualized answer

## 🧩 Common Commands for Review
# Activate environment
source .lcenv/bin/activate

# Run the Streamlit app
python -m streamlit run app.py

# Install a missing package
pip install <package-name>

# Check which Python is used
which python

⚠️ Common Issues
Error	Fix
ModuleNotFoundError: No module named 'langchain'	Activate env → source .lcenv/bin/activate
pypdf.errors.DependencyError	pip install --upgrade cryptography
ImportError: Could not import chromadb	pip install "chromadb>=0.5.0"
Wrong Python (Anaconda) used	Always run python -m streamlit run app.py
## 📈 Next Steps (for Enterprise QA version)

✅ Replace Chroma with FAISS and persist embeddings locally.

📂 Support multiple PDFs (use st.file_uploader(..., accept_multiple_files=True)).

🧭 Add conversation memory (ConversationBufferMemory).

🧩 Add citation and source display for transparency.

🚀 Deploy via FastAPI + Docker for real enterprise scenarios.

## 🧑‍💻 Author

Built by Huiwen — graduate student exploring AI applications for real-world enterprise knowledge systems.


