# Poker Tutor: RAG-based AI Chatbot

Poker Tutor is a poker education chatbot built with **Streamlit**, **LangChain**, using **OpenAI GPT models**, designed to answer poker-related questions using **Retrieval-Augmented Generation (RAG)**.  
It processes poker books in PDF format, extracts and embeds their contents into a vector database, and uses them to provide **source-referenced** answers.

This software was developed as part of my Bachelor’s thesis.

---

## Project Overview

### Objective
The aim of the project was to develop a **poker tutoring and learning support tool** that helps players improve their poker knowledge and decision-making.  
The software is an **LLM based chatbot** capable of answering poker-related questions and providing strategic advice through natural conversation.

To achieve this, the project uses:
- **LLM (Large Language Model):** a large-scale neural model responsible for generating human-like responses.  
- **RAG (Retrieval-Augmented Generation):** a technique that enhances accuracy and reliability by grounding the model’s answers in the content of user-provided documents.

---

### Stack Used
| Tool | Purpose |
|------|----------|
| **OpenAI** | Provides the large language models (LLMs) used for text generation. |
| **LangChain** | Connects the model to external data and manages retrieval, logic, and prompt construction. |
| **Streamlit** | Builds the web interface for displaying and interacting with the chatbot. |

---

### PDF Processing Pipeline
The core of the system is the **PDF processing workflow**, which transforms uploaded poker books into retrievable knowledge for the chatbot.

| Step | Description |
|------|--------------|
| **Load** | Reads the raw text from PDF files. |
| **Clean** | Removes unwanted characters, line breaks, and hyphenations. |
| **Chunk** | Splits long texts into smaller parts, as models can only process a limited number of tokens at once. |
| **Annotate** | (Optional) Could be used to label or categorize text segments for advanced processing. |
| **Vectorize** | Converts text chunks into numerical embeddings and stores them in a vector database for similarity search. |

---

### How vector embedding works

---

### How RAG works


---

### Additional Settings
The chatbot interface allows several adjustable options:
- **Name:** personalize the interaction (the model greets the user by name).  
- **Select Mode:** choose between two behavior modes:  
  - *Only RAG* – answers strictly from uploaded sources.  
  - *General Knowledge* – combines document-based answers with broader poker knowledge.  
- **Style:** define the response style (Normal, Explain, Summarize, or Step-by-Step).  
- **Focus:** emphasize a specific topic (Poker Basics, Expected Value, or Bluffing).

---

### Software Design
The software follows an **object-oriented structure** with clearly separated components:
- **PDF class:** handles PDF loading, cleaning, chunking, and vectorization.  
- **PDF Manager class:** coordinates the document workflow and session state in Streamlit.  
- **PokerTutor Chat class:** manages conversation flow, memory, and context retrieval.

The design includes:
- **Class Diagram:** visualizing the main class relationships.  
- **Software Architecture Diagram:** showing the integration of Streamlit, LangChain, and the vector database.


## Project Structure


- `app.py` - Streamlit frontend and control flow
- `pdf_class.py` - PDF class for loading, cleaning, chunking, and vectorizing
- `pdf_manager.py` - Manages multiple PDFs and session state
- `chat_class.py` - PokerTutor chatbot logic (retrieval + generation)
- `.env` - Environment variables
- `requirements.txt` - Dependencies

## Local Installation and Setup

### 1. Clone the Repository
```git clone https://github.com/yourusername/poker-tutor-rag.git 
cd poker-tutor-rag>```

### 2. Create a Virtual Environment
Option A: **Python Virtual Environment**

Create a virtual environment:

`python -m venv poker_tutor`

Activate the environment:

Windows: `poker_tutor\Scripts\activate`

Linux/MacOS: `source poker_tutor/bin/activate`

Install the dependencies:

`pip install -r requirements.txt`

Option B: **Conda Virtual Environment**

Create and activate the environment:

```conda create --name poker_tutor python=3.10
conda activate poker_tutor
pip install -r requirements.txt```

### Environment Variables
Create a .env file in the project root:

```OPENAI_API_KEY="your_api_key"
OPENAI_MODEL="gpt-4o-mini"
PDF_FOLDER="pdf_files"
TABLE_FOLDER="tables"
STATUS_FILE="status_file.json"
EMBEDDING_MODEL="text-embedding-ada-002"
DB_DIR = "./chroma_db"
LOG_FILE = "log_file.log"```

### Running the application
`streamlit run app.py`

### Recommended settings
- Use the left sidebar (open with the arrow) to select PDFs.
- Choose **PyPDFLoader** for most PDFs.  
- Use **pdfplumber** only for PDFs filled with tables.

### Requirements

```streamlit==1.42.0
langchain==0.3.17
langchain-community==0.3.16
langchain-core==0.3.34
langchain-openai==0.3.5
langchain-text-splitters==0.3.6
openai==1.61.1
chromadb==0.6.3
tiktoken==0.8.0
python-dotenv==1.0.1
pdfplumber==0.11.5
pandas==2.2.3
langchain-chroma==0.2.2
pypdf==5.3.0
pydantic==2.10.6
tabulate==0.9.0```
