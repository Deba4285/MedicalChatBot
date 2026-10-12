import os
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


# Load environment variables
load_dotenv()

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the information provided in the context to answer the user's question.

Rules:
1. Answer using only the provided context.
2. If the answer is not available in the context, say:
   "I don't know based on the provided medical documents."
3. Do not make up information.
4. Start the answer directly without small talk.
5. Provide a clear and complete answer.

Context:
{context}

Question:
{question}

Answer:
"""


# Load FAISS vector store
@st.cache_resource
def get_vecstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"FAISS vector store not found at: {DB_FAISS_PATH}"
        )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# Create custom prompt
def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


# Load Groq LLM
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        max_tokens=512,
        api_key=api_key
    )


# Create the RetrievalQA chain
@st.cache_resource
def get_qa_chain():
    vectorstore = get_vecstore()
    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt()
        }
    )

    return qa_chain


# Main Streamlit application
def main():
    st.set_page_config(
        page_title="MediBot",
        page_icon="🩺",
        layout="centered"
    )

    st.title("🩺 MediBot")
    st.caption("Ask questions about your medical documents.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        # Display and save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching medical documents..."):
                try:
                    qa_chain = get_qa_chain()

                    response = qa_chain.invoke({
                        "query": prompt
                    })

                    result = response["result"]

                    # Display answer
                    st.markdown(result)

                    # Display source documents
                    source_documents = response.get(
                        "source_documents", []
                    )

                    if source_documents:
                        with st.expander("View source documents"):
                            for i, document in enumerate(
                                source_documents, start=1
                            ):
                                st.markdown(f"**Source {i}**")
                                st.write(document.page_content)

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()