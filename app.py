from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="mining_education_docs")

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Step 1: Semantic search in Chroma DB
        results = collection.query(
            query_texts=[request.query],
            n_results=request.top_k
        )
        
        # Extract relevant documents
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Step 2: Generate response using Groq
        context = "\n\n".join(documents)
        prompt = f"""
        You are a helpful assistant that answers questions based on the provided context.
        Context: {context}
        
        Question: {request.query}
        Answer:
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate answers based on the given context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # or "llama2-70b-4096"
            temperature=0.3,
            max_tokens=1024
        )
        
        # Step 3: Format response
        answer = chat_completion.choices[0].message.content
        sources = [metadata.get('source', '') for metadata in metadatas]
        
        return QueryResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))