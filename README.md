```bash
# 📄 PDF-Based RAG with Streamlit, LangChain, and Ollama

## Overview
The **PDF-Based RAG** leverages **Streamlit**, **LangChain**, **ChromaDB**, and **Ollama models** to provide users with intelligent responses based on PDF documents. This system processes PDFs, encodes them using text embeddings, and utilizes **ChromaDB** for efficient vector storage and retrieval, delivering precise responses through a user-friendly web interface.

## ✨ Features
- **Document Preprocessing**: Splits and processes PDF documents into manageable chunks.
- **Vector Storage**: Efficient indexing and storage using **ChromaDB**.
- **Intelligent Retrieval**: Uses **Multi-Query Retrieval** for improved accuracy.
- **Generative AI Integration**: Powered by **Ollama** models (`llama3.2:1b` & `deepseek-r1:1.5b`).
- **Simple Web Interface**: Built with **Streamlit** for seamless user experience.
- **Logging & Response Storage**: Saves user queries and responses for analysis.

## 🚀 Workflows
1. **Load and Process PDF**: Extracts text and splits it into chunks.
2. **Embed & Store in ChromaDB**: Generates embeddings using **nomic-embed-text**.
3. **Retrieve Context**: MultiQueryRetriever fetches relevant document chunks.
4. **Generate Answer**: LLM models produce responses based on retrieved data.
5. **Log Responses**: Stores interactions in a CSV file.

## 🛠️ Installation & Usage

# Clone the repository
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# Run the application
streamlit run app.py

## 📂 Project Structure
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── logs/                  # Logging output
├── chroma_db/             # Vector storage database
├── data/                  # PDF document storage
└── utils/                 # Document processing & retrieval scripts

## 🔧 Customization
# Switch Model: Update `MODEL_NAME` in `app.py`
MODEL_NAME = "llama3.2:1b"  # or "deepseek-r1:1.5b"

# Change Document: Modify `DOC_PATH`
DOC_PATH = "your-document.pdf"

## 📌 Example Query
# User: "How does this paper help improve vaccine adoption?"
# Response: "Based on the extracted context, the paper discusses..."

## ❤️ Credits
# Developed by: [Tanvir Ahmed](https://github.com/tanvircs)

# This one-page README ensures a structured and easy-to-follow guide for deployment. 🚀
```

