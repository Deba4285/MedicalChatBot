# AI Medical Chatbot 🩺

An AI-powered medical chatbot built with Python and Large Language Models (LLMs).
The project uses a vector database to retrieve relevant information from medical documents and generate responses to user queries.

## 🚀 Features

- AI-powered question answering
- Medical document retrieval
- Vector database integration
- Conversational memory
- LLM integration using Groq API
- Environment variable management

## 🛠️ Technologies Used

- Python
- LangChain
- Groq API
- Vector Database
- Embeddings
- Retrieval-Augmented Generation (RAG)

## 📁 Project Structure

```text
chat-bot/
│
├── data/                      # Medical documents and data
├── vectorstore/               # Vector database
├── .myenv/                    # Virtual environment (not uploaded)
├── .env                       # API keys (not uploaded)
├── connect_memory_with_llm.py # Chatbot with conversational memory
├── llm.py                     # LLM integration
├── medibot.py                 # Medical chatbot application
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignored files
└── README.md                  # Project documentation
