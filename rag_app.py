import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# === LangChain 新结构 ===
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pypdf import PdfReader
import re

# ===== 初始化 =====
load_dotenv()

def fix_page_number_in_content(content, correct_page):
    """修复文档内容中的页码信息"""
    # 查找并替换 "Apple Inc. | 2024 Form 10-K | X" 格式的页码
    pattern = r'Apple Inc\. \| 2024 Form 10-K \| \d+'
    replacement = f'Apple Inc. | 2024 Form 10-K | {correct_page}'
    return re.sub(pattern, replacement, content)

INDEX_DIR = "faiss_index"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="🏢 Enterprise Knowledge QA System", page_icon="🏢")
st.title("🏢 Enterprise Knowledge QA System")
st.markdown("Upload enterprise documents (e.g., annual reports, policies) and ask questions interactively.")

# ===== 初始化模型 =====
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ===== 上传文件 =====
uploaded_files = st.file_uploader("📄 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    page_mapping = {}  # 存储文档索引到页码的映射
    with st.spinner("Processing PDFs..."):
        for file_idx, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # 获取PDF总页数
            reader = PdfReader(tmp_path)
            total_pages = len(reader.pages)
            
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # 计算页码偏移量
            page_offset = 0
            offset_found = False
            
            # 查找文档内容中的页码模式来确定偏移量
            for doc in docs:
                content = doc.page_content
                # 查找 "Apple Inc. | 2024 Form 10-K | X" 模式
                match = re.search(r'Apple Inc\. \| 2024 Form 10-K \| (\d+)', content)
                if match:
                    content_page = int(match.group(1))
                    loader_page = doc.metadata.get('page', 0)
                    # 计算偏移量：内容页码 - loader页码
                    page_offset = (content_page - 1) - loader_page
                    offset_found = True
                    break
            
            if not offset_found:
                page_offset = 0
            
            # 为每个原始文档添加文件名和页码信息
            for doc in docs:
                doc.metadata['filename'] = uploaded_file.name
                doc.metadata['total_pages'] = total_pages
                # 确保页码信息正确传递
                if 'page' not in doc.metadata:
                    doc.metadata['page'] = 0
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            split_docs = splitter.split_documents(docs)
            
            # 为每个文档片段添加文件名和正确的页码映射
            for doc in split_docs:
                # 获取原始页码（从0开始）
                original_page = doc.metadata.get('page', 0)
                # 应用偏移量并转换为打印页码（从1开始）
                corrected_page = original_page + page_offset
                print_page = corrected_page + 1
                
                # 确保文件名和页码信息正确传递
                doc.metadata['filename'] = uploaded_file.name
                # Source 显示小页码（原始页码）
                original_print_page = original_page + 1
                doc.metadata['print_page'] = original_print_page
                doc.metadata['total_pages'] = total_pages
                
                # 修复文档内容中的页码信息 - 使用大页码（经过偏移量修正的页码）
                doc.page_content = fix_page_number_in_content(doc.page_content, print_page)
                
                # 存储映射关系
                doc_id = f"{uploaded_file.name}_{original_page}"
                page_mapping[doc_id] = {
                    'filename': uploaded_file.name,
                    'print_page': print_page,
                    'total_pages': total_pages
                }
                
            
            all_docs.extend(split_docs)

        # 建立向量数据库
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(INDEX_DIR)

    st.success("✅ Documents processed and indexed!")

    # ===== 构建 RAG QA 链 =====
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # ===== 用户提问 =====
    query = st.text_input("💬 Ask a question about your documents:")
    if query:
        with st.spinner("Generating answer..."):
            # Retrieve relevant documents
            docs = retriever.invoke(query)
            
            # Format context from retrieved documents
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Create prompt and get answer
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an enterprise knowledge assistant. Use the provided context to answer the question. If the context contains relevant information, provide a comprehensive answer. If some information is missing, mention what you can provide based on the available context."),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": query})

            st.markdown(f"### 🧠 Answer:\n{answer}")

            st.markdown("### 📚 Sources:")
            for i, d in enumerate(docs[:2]):
                # 获取文件名和打印页码
                filename = d.metadata.get('filename', 'Unknown')
                print_page = d.metadata.get('print_page', 'N/A')
                total_pages = d.metadata.get('total_pages', 'N/A')
                
                st.write(f"**Source {i+1}:** {filename} - Page {print_page} of {total_pages}")
                # 使用 st.text_area 来显示长文本，并设置合适的行数
                st.text_area(
                    f"Content {i+1}:", 
                    value=d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content,
                    height=100,
                    key=f"source_{i}",
                    disabled=True
                )
else:
    st.info("👆 Please upload at least one PDF to begin.")


