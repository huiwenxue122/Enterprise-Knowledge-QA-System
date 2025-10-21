# 🏢 Enterprise Knowledge QA System

An **AI-powered document question answering app** built with **Streamlit** and **LangChain**.  
Users can upload one or more PDF documents and ask natural-language questions — the system retrieves relevant content and generates precise answers using OpenAI models.

---

## 🚀 Features
✅ Upload multiple PDF documents  
✅ Extract and embed document text with LangChain  
✅ Vector search using FAISS  
✅ Question answering powered by OpenAI API  
✅ Simple web UI built with Streamlit  

---

## 🧰 Tech Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Backend**: Python 3.10+  
- **LLM Framework**: [LangChain](https://github.com/hwchase17/langchain)  
- **Vector Store**: FAISS  
- **Embedding Model**: OpenAI Embeddings API  
- **LLM Model**: GPT-based models from OpenAI  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/huiwenxue122/Enterprise-Knowledge-QA-System-Clean.git
cd Enterprise-Knowledge-QA-System-Clean
```
### 2️⃣ Create a virtual environment
```bash
python3 -m venv .ekqs_env
source .ekqs_env/bin/activate  # On macOS / Linux
# or
.ekqs_env\Scripts\activate     # On Windows
```
### 3️⃣ Install dependencies
```bash
pip install -U pip wheel setuptools
pip install -r requirements.txt
```
🔑 Environment Setup

Create a .env file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```
▶️ Run the App
```bash
streamlit run rag_app.py
```

Then open your browser and go to:
```bash
http://localhost:8501
```
## 🧠 How It Works

Upload PDFs → The system loads and parses the documents.

Embed Text → Text chunks are transformed into vectors using OpenAI embeddings.

Store Vectors → FAISS stores these embeddings for fast semantic retrieval.

Ask Questions → User queries are embedded and matched against the stored document vectors.

Generate Answers → The LLM formulates a context-aware answer based on retrieved content.

## 📂 Project Structure
├── rag_app.py             # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env                   # Your API key (excluded from Git)
├── .gitignore
└── result/                # Local folder for outputs (ignored by Git)

## 📸 Example UI

## 🧑‍💻 Author

## Caire Xue






