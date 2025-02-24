from typing import List
from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF for PDF extraction
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from transformers import pipeline
import os
from collections import deque

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Initialize FastAPI
app = FastAPI()

# Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Vector Store
embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
index = faiss.IndexFlatL2(embedding_dim)
documents = []  # Store document text and metadata

# Load Transformer Pipelines
text_generator = pipeline("text-generation", model="google/flan-t5-base")
summarizer = pipeline("summarization", model="google/flan-t5-base")

# Load Chatbot Model (e.g., Mistral-7B or LLaMA-2-7B)
chatbot_model = pipeline("text-generation", model="mistralai/Mistral-Small-24B-Instruct-2501", max_new_tokens=200)

# Conversation Memory (Stores last 5 exchanges per user)
conversation_history = {}

class QueryRequest(BaseModel):
    query: str
    user_id: str  # Unique user ID for tracking conversation history

def extract_text_from_pdf(pdf_file: UploadFile):
    """Extract text from a PDF file."""
    pdf_data = pdf_file.file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    return [page.get_text("text") for page in doc]

def generate_embeddings(text: str):
    """Generate embeddings using Sentence Transformers."""
    return embedding_model.encode(text, convert_to_numpy=True)

@app.post("/upload_reports/")
async def upload_reports(files: List[UploadFile] = File(...)):
    """Process and store PDF embeddings."""
    global documents, index

    for file in files:
        sections = extract_text_from_pdf(file)

        for section in sections:
            documents.append({"text": section, "source": file.filename})

            # Generate embeddings
            embeddings = generate_embeddings(section)

            # Store in FAISS
            embeddings = embeddings.reshape(1, -1)
            index.add(embeddings)

    return {"message": "Reports uploaded and indexed successfully."}

@app.post("/query/")
async def query_reports(request: QueryRequest):
    """Retrieve relevant text from reports based on query."""
    query_embedding = generate_embeddings(request.query).reshape(1, -1)

    if index.ntotal == 0:
        return {"query": request.query, "response": "No data available for search."}

    # Search FAISS for the top 5 most similar sections
    D, I = index.search(query_embedding, k=5)

    # Retrieve matching sections
    relevant_sections = [
        documents[i] for i in I[0] if 0 <= i < len(documents)
    ]

    return {
        "query": request.query,
        "retrieved_text": [doc["text"] for doc in relevant_sections],
        "sources": [doc["source"] for doc in relevant_sections]
    }

@app.post("/rag_chat/")
async def rag_chat(request: QueryRequest):
    """Perform Retrieval-Augmented Generation (RAG) chat and summarization."""

    # Retrieve relevant sections
    query_embedding = generate_embeddings(request.query).reshape(1, -1)
    D, I = index.search(query_embedding, k=5)
    relevant_sections = [documents[i]["text"] for i in I[0] if 0 <= i < len(documents)]
    context_text = "\n\n".join(relevant_sections)

    if not context_text:
        return {"query": request.query, "response": "No relevant documents found."}

    # Generate AI response using text generation
    prompt = f"Given the following document context, answer: {request.query}\n\nContext:\n{context_text}"
    ai_response = text_generator(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']

    # Summarize the generated response
    summary = summarizer(ai_response, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    return {
        "query": request.query,
        "response": summary,  # Returning the summarized response
        "sources": [documents[i]["source"] for i in I[0] if 0 <= i < len(documents)]
    }

@app.post("/chatbot/")
async def chatbot(request: QueryRequest):
    """Chat with the AI model, maintaining conversation history."""

    # Retrieve conversation history for the user
    user_id = request.user_id
    if user_id not in conversation_history:
        conversation_history[user_id] = deque(maxlen=5)  # Store last 5 exchanges

    # Prepare chat history
    chat_history = "\n".join(conversation_history[user_id])
    chat_prompt = f"{chat_history}\nUser: {request.query}\nAI:"

    # Generate AI chatbot response
    ai_response = chatbot_model(chat_prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

    # Store conversation history
    conversation_history[user_id].append(f"User: {request.query}")
    conversation_history[user_id].append(f"AI: {ai_response}")

    return {
        "query": request.query,
        "response": ai_response,
        "chat_history": list(conversation_history[user_id])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
