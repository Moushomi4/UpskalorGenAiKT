import os
import shutil
import logging
from operator import itemgetter

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

# --- CONFIGURATION ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyA8ybpS5Rea5bO9e-mKKxTumKN8mLhXELs"
#AIzaSyC9LeUR6oIVGQ8xo_o2gPXYX737Ih_Im0k
# AIzaSyA8ybpS5Rea5bO9e-mKKxTumKN8mLhXELs
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"
K_RETRIEVAL = 3
SECRET_KEY = "super-secret-key"
ALGORITHM = "HS256"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Global variables to hold our RAG components
rag_chain = None
retriever = None

def build_rag_chain(current_retriever):
    """Rebuilds the chain with the latest retriever."""
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3
)
    template = """
    You are InnerAlign Wellness Assistant. 
    USER CONTEXT: {drift_context}
    PROACTIVE GUIDANCE: {bucket_guidance}
    KNOWLEDGE BASE: {context}
    
    INSTRUCTIONS:
    1. If the user message is a simple greeting, use PROACTIVE GUIDANCE.
    2. Always be empathetic. 
    3. Provide wellness advice from the knowledge base.
    
    User Message: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        {
            "context": itemgetter("question") 
                       | current_retriever 
                       | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": itemgetter("question"),
            "drift_context": itemgetter("drift_context"),
            "bucket_guidance": itemgetter("bucket_guidance"),
        }
        | prompt
        | llm
        | RunnableLambda(lambda x: x.content)
    )

@app.on_event("startup")
async def startup_event():
    global rag_chain, retriever
    try:
        embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        index_name = "document_index"
        index_file = os.path.join(INDEX_PATH, f"{index_name}.faiss")

        if os.path.exists(INDEX_PATH) and os.path.exists(index_file):
            vectorstore = FAISS.load_local(
                INDEX_PATH,
                embed_model,
                index_name=index_name,  
                allow_dangerous_deserialization=True,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})
            rag_chain = build_rag_chain(retriever)
            logger.info("AI System loaded with existing index.")
        else:
            logger.warning("No valid index found. System waiting for upload.")
    except Exception as e:
        logger.error(f"AI Init Failed: {e}")

async def get_current_admin(token: str = Depends(oauth2_scheme)):
    # NOTE: For testing, you can bypass this or provide a valid JWT
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return payload
    except Exception:
        raise HTTPException(status_code=403, detail="Access denied")

@app.post("/admin/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...), 
    # admin: dict = Depends(get_current_admin) # Commented out for easier initial testing
):
    global rag_chain, retriever
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(file_path) if file.filename.endswith(".pdf") else TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        index_name = "document_index"

        if os.path.exists(os.path.join(INDEX_PATH, f"{index_name}.faiss")):
            vectorstore = FAISS.load_local(INDEX_PATH, embed_model, index_name=index_name, allow_dangerous_deserialization=True)
            vectorstore.add_documents(docs)
        else:
            vectorstore = FAISS.from_documents(docs, embed_model)

        vectorstore.save_local(INDEX_PATH, index_name=index_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})
        rag_chain = build_rag_chain(retriever)

        return {"message": f"Success! {file.filename} integrated."}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat")
async def chat(question: str, drift: str = "Neutral", guidance: str = "General wellness"):
    if not rag_chain:
        return {"error": "Knowledge base is empty. Please upload a document first."}
    
    response = rag_chain.invoke({
        "question": question,
        "drift_context": drift,
        "bucket_guidance": guidance
    })
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
