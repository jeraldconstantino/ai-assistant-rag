# RAG-based AI Assistant

A Retrieval-Augmented Generation (RAG) AI Assistant built with FastAPI backend and Streamlit frontend. This application allows users to upload documents (PDF, TXT, DOCX) and interact with an AI assistant that can answer questions based on the uploaded content using OpenAI's language models.

## 🚀 Features

### Core Functionality
- **Document Ingestion**: Upload and process PDF, TXT, and DOCX files
- **Vector Storage**: Automatic document chunking and embedding using ChromaDB
- **Intelligent Query Routing**: Distinguishes between document-based and conversational queries
- **RAG Pipeline**: Retrieval-augmented generation for accurate, context-aware responses
- **Conversation Memory**: Maintains chat history for contextual conversations
- **Customizable LLM Parameters**: Adjustable temperature, max tokens, top-p, frequency/presence penalties

### User Interface
- **Modern Streamlit UI**: Clean, responsive web interface
- **File Upload**: Drag-and-drop file upload with progress indicators
- **Real-time Chat**: Interactive chat interface with conversation history
- **LLM Settings Panel**: Sidebar controls for fine-tuning model parameters
- **Status Indicators**: Visual feedback for file processing and ingestion

### Backend API
- **FastAPI REST API**: High-performance asynchronous backend
- **Multiple Inference Modes**: Direct and context-aware inference endpoints
- **File Management**: Automatic file processing and ingestion tracking
- **Error Handling**: Comprehensive logging and error management

## 🏗️ Architecture

```
RAG AI Assistant/
├── app/
│   ├── be/                             # Backend (FastAPI)
│   │   ├── main.py                     # FastAPI application entry point
│   │   ├── api/
│   │   │   └── routes.py               # API endpoints
│   │   ├── core/
│   │   │   ├── config.py               # Backend configuration
│   │   │   └── prompt_template.txt     # LLM prompt template
│   │   ├── data/
│   │   │   ├── raw/                    # Uploaded files storage
│   │   │   └── vector_store/           # ChromaDB vector database
│   │   ├── schemas/
│   │   │   ├── inference_models.py     # Pydantic models for inference
│   │   │   └── ingestion_models.py     # Pydantic models for ingestion
│   │   └── utils/
│   │       ├── inference.py            # RAG inference logic
│   │       ├── ingestion.py            # Document processing logic
│   │       └── model.py                # OpenAI model integration
│   └── fe/                             # Frontend (Streamlit)
│       ├── main.py                     # Streamlit application entry point
│       ├── ui.py                       # Main UI components
│       ├── api_requests/
│       │   ├── inference.py            # Backend API client for inference
│       │   └── ingestion.py            # Backend API client for ingestion
│       └── core/
│           └── config.py               # Frontend configuration
├── requirements.txt                    # Python dependencies
└── README.md                          
```

## 📋 Prerequisites

- Python 3.12+
- OpenAI API Key
- Git

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jeraldconstantino/ai-assistant-rag.git
   cd ai-assistant-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🚀 Usage

### Starting the Application

1. **Start the Backend Server**
   ```bash
   uvicorn app.be.main:app
   ```
   The FastAPI server will start on `http://localhost:8000`

2. **Start the Frontend (in a new terminal)**
   ```bash
   cd app/fe
   streamlit run main.py
   ```
   The Streamlit app will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload Documents**
   - Click the "📎 Attach Files" button
   - Upload PDF, TXT, or DOCX files
   - Wait for the ingestion process to complete

2. **Configure LLM Settings** (Optional)
   - Use the sidebar to adjust:
     - Temperature (0.0-1.0): Controls response creativity
     - Max Tokens (100-4000): Limits response length
     - Top P (0.0-1.0): Nuclear sampling parameter
     - Frequency/Presence Penalties: Reduces repetition

3. **Start Chatting**
   - Type questions in the chat input
   - The AI will automatically determine if your question needs document context
   - Receive intelligent responses based on your uploaded documents

## 🔧 Configuration

### Backend Configuration (`app/be/core/config.py`)
- **OpenAI Settings**: API key, model selection
- **Vector Store**: ChromaDB configuration
- **File Paths**: Data storage locations
- **Model Parameters**: Default LLM settings

### Frontend Configuration (`app/fe/core/config.py`)
- **UI Settings**: Title, layout, styling
- **API Endpoints**: Backend service URLs

### Prompt Template (`app/be/core/prompt_template.txt`)
- Customizable prompt template for RAG responses
- Includes few-shot examples for consistent behavior
- Structured for document-based Q&A tasks

## 📚 API Endpoints

### Backend API (`http://localhost:8000`)

- **POST `/api/inference`**: Context-aware inference with document retrieval
- **POST `/api/direct-inference`**: Direct LLM inference without document context
- **POST `/api/ingestion`**: File upload and document processing
- **GET `/`**: Health check endpoint

### Request/Response Examples

**Inference Request:**
```json
{
  "query": "What are the company's main products?",
  "history": "Previous conversation context...",
  "ai_model_parameters": {
    "temperature": 0.0,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

**Inference Response:**
```json
{
  "response": "Based on the uploaded documents, the company's main products include..."
}
```

## 📝 Dependencies

### Core Technologies
- **FastAPI**: Modern Python web framework for APIs
- **Streamlit**: Web app framework for ML/AI applications
- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for embeddings
- **OpenAI**: Language models and embeddings

### Key Libraries
- **Document Processing**: PyPDF, docx2txt, langchain-text-splitters
- **Vector Operations**: langchain-chroma, langchain-openai
- **Web Framework**: fastapi, streamlit, uvicorn
- **Utilities**: pydantic, loguru, python-multipart

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.

---

**Note**: This application requires an active OpenAI API key and internet connection for full functionality. Ensure your API key has sufficient credits for embeddings and language model usage.
