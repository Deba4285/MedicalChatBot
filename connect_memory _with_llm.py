import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
huggingface_repo_id="prism-ml/Ternary-Bonsai-2-27B-gguf"

def load_llm():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY")
    )

    return llm
# Step 2: Connect LLM with FAISS and Create Chain

CUS_PRMTP_PTM = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
db_faiss_path="vectorstore/db_faiss"
def set_custm_prmpt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=['context',"question"])
    return prompt

embd=HuggingFaceEmbeddings( model="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(db_faiss_path,embd,allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(
        search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custm_prmpt(CUS_PRMTP_PTM)
    }
)

user_query=input('ask  :')
response=qa_chain.invoke({
    "query":user_query
})
print("result : ",response["result"])