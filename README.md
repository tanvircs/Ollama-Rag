# 📚 Document Assistant

## Overview
The **Document Assistant** leverages **Streamlit**, **LangChain**, **Chroma**, and **Ollama** to process PDFs and provide intelligent responses to user queries. It encodes documents using embedding models, stores them in a **Chroma vector database**, and retrieves relevant context to enhance natural language understanding.

---

## ✨ Features
- **📄 PDF Processing**: Reads and processes PDF documents for retrieval.
- **🔍 Advanced Retrieval**: Utilizes multi-query retrievers for improved search accuracy.
- **🧠 Generative AI Integration**: Powered by **DeepSeek R1:1.5B** for context-aware responses.
- **⚡ Streamlit UI**: Simple and interactive web interface.
- **📊 Logging & Response Tracking**: Logs interactions and saves responses for review.

---

## 🚀 Workflows

1. **Load & Process PDFs**
   - Extract text and split it into meaningful chunks.
2. **Vectorize & Store Data**
   - Encode document embeddings using **Ollama** and store them in **ChromaDB**.
3. **Query & Generate Response**
   - Use a **Multi-Query Retriever** to fetch relevant context.
4. **Display Response**
   - Generate and display context-aware answers using the **LLM**.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- **Ollama** installed and running
- Required Python packages from `requirements.txt`

### 1️⃣ Clone the Repository
```bash
git clone <repo-url>
cd <repo-name>
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Start Ollama Service
```bash
ollama serve
```

### 5️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```plaintext
.
├── app.py                # Main Streamlit application
├── src/
│   ├── document_loader.py # PDF processing functions
│   ├── retriever.py       # Multi-query retriever setup
│   ├── model.py          # LLM interaction module
├── requirements.txt      # Python package dependencies
├── logs/                 # Log files
├── responses/            # CSV responses
├── chroma_db/            # Vector database storage
```

---

## ❤️ Credits
**Developed by**: Tanvir Ahmed  
**GitHub**: [tanvircs](https://github.com/tanvircs)

---
