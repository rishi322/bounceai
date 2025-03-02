from collections import deque
from typing import List, Dict
import os
import faiss
import fitz  # PyMuPDF for PDF extraction
import numpy as np
import textblob
import re
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from huggingface_hub import login

# Load Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables.")
login(token=hf_token)

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rag_model = pipeline("text-generation", model="t5-base", device=0 if torch.cuda.is_available() else -1)
summarizer = pipeline("summarization", model="t5-base")
sentiment_analyzer = pipeline("sentiment-analysis", model="t5-base")
ner_model = pipeline("ner", model="dslim/bert-base-NER")

# FAISS Vector Store
embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
index = faiss.IndexFlatL2(embedding_dim)
documents = []  # Store document text and metadata

# Global variables to store extracted text
text2 = ""
textfetch = []

# Define Pydantic models
class ReportRequest(BaseModel):
    report1: str
    report2: str

class QueryRequest(BaseModel):
    query: str

class TextRequest(BaseModel):
    text: str

# Helper functions
def extract_text_from_pdf(pdf_file: UploadFile):
    """Extract text from PDF file."""
    global text2
    pdf_data = pdf_file.file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text_sections = [page.get_text("text") for page in doc]
    text2 = "\n".join(text_sections)
    return text_sections

def generate_embeddings(text: str):
    """Generate embeddings using Sentence Transformers."""
    return embedding_model.encode(text, convert_to_numpy=True)

def summarize_text(text: str):
    """Summarize long text using a Hugging Face model."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

def analyze_sentiment(text: str):
    """Analyze sentiment of the text."""
    analysis = textblob.TextBlob(text)
    sentiment_score = analysis.sentiment.polarity  # -1 (negative) to +1 (positive)
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def generate_word_cloud(text: str):
    """Generate a word cloud from the given text and return an image stream."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Generated Word Cloud", fontsize=14)
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png")
    plt.close()
    img_stream.seek(0)
    return img_stream

# API endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the RAG Market Report API"}

@app.get("/hi/")
def hi():
    return {"message": "Hi there!"}

@app.post("/upload_reports/")
async def upload_reports(files: List[UploadFile] = File(...)):
    """Process and store PDF embeddings."""
    global documents, index, text2, textfetch
    text2 = ""  # Reset previous data
    textfetch = []
    for file in files:
        sections = extract_text_from_pdf(file)
        textfetch.append({"text": " ".join(sections), "source": file.filename})
        for section in sections:
            documents.append({"text": section, "source": file.filename})
            embeddings = generate_embeddings(section).reshape(1, -1)
            index.add(embeddings)
    return {"message": "Reports uploaded and processed successfully."}

@app.get("/analyze_reports/")
def analyze_reports():
    """Summarize and analyze uploaded reports."""
    global textfetch
    if len(textfetch) < 2:
        raise HTTPException(status_code=400, detail="At least two reports are required.")
    summaries = [summarize_text(report["text"]) for report in textfetch]
    return {"summaries": summaries}

@app.post("/generate_wordcloud/")
async def generate_wordcloud(request: ReportRequest):
    """Generate a word cloud from two reports."""
    text1 = "\n".join([doc["text"] for doc in documents if doc["source"] == request.report1])
    text2 = "\n".join([doc["text"] for doc in documents if doc["source"] == request.report2])
    if not text1 or not text2:
        raise HTTPException(status_code=404, detail="One or both reports not found.")
    img_stream = generate_word_cloud(text1 + " " + text2)
    return StreamingResponse(img_stream, media_type="image/png")

@app.post("/check_plagiarism/")
async def check_plagiarism(files: List[UploadFile] = File(...)):
    """Check plagiarism similarity between two uploaded PDF reports."""
    text1 = extract_text_from_pdf(files[0])
    text2 = extract_text_from_pdf(files[1])
    similarity_score = util.pytorch_cos_sim(
        generate_embeddings(text1), generate_embeddings(text2)
    ).item()
    return {"similarity_score": round(similarity_score, 4)}

@app.post("/rag_chat/")
async def rag_chat(request: QueryRequest):
    """Perform Retrieval-Augmented Generation (RAG) chat."""
    query_embedding = generate_embeddings(request.query).reshape(1, -1)
    D, I = index.search(query_embedding, k=5)
    relevant_sections = [documents[i]["text"] for i in I[0] if 0 <= i < len(documents)]
    context_text = "\n\n".join(relevant_sections)
    if not context_text:
        return {"query": request.query, "response": "No relevant documents found."}
    ai_response = rag_model(
        f"Given the following document context, answer: {request.query}\n\n{context_text}",
        max_length=500,
        num_return_sequences=1
    )[0]['generated_text']
    return {"query": request.query, "response": ai_response}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)