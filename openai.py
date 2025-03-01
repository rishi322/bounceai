from collections import deque
from typing import List
import os
import faiss
import fitz  # PyMuPDF for PDF extraction
import numpy as np
import textblob
import re
import torch
from fastapi import FastAPI, UploadFile, File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import os

from transformers import pipeline, AutoTokenizer
from typing import List, Dict
# Load Language Model for RAG (Example: OpenAI GPT or Mistral-7B)
rag_model = pipeline("text-generation", model="t5-base")  # Replace with better RAG model

# Global variable to store extracted text
text2 = []
# Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
OPENAI_API_KEY = "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"

model_name = "t5-base"
generator = pipeline("summarization", model=model_name)
sentiments = pipeline("sentiment-analysis",model=model_name)

# Load tokenizer for proper truncation
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline("ner",model="dslim/bert-base-NER")
class ReportRequest(BaseModel):
    report1: str
    report2: str
# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_pipeline = pipeline("text-generation", model="t5-base", device=0)  # Use GPU if available

# FAISS Vector Store
embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
index = faiss.IndexFlatL2(embedding_dim)
documents = []  # Store document text and metadata

class QueryRequest(BaseModel):
    query: str
textfetch = []


def summarize_text2(text):
    """Summarizes long reports before passing them to the AI model."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]


def generate_ai_answer(prompt, text1, text2):
    """Generates AI-based responses using Hugging Face's Mistral-7B."""
    combined_text = f"Report 1: {text1}\n\nReport 2: {text2}\n\n{prompt}"

    response = llm_pipeline(combined_text, max_length=500, do_sample=True)
    return response[0]["generated_text"]


@app.get("/ai_prompt/")
def ai_prompt_analysis():
    """Processes reports and generates an AI response based on user prompt."""

    if len(textfetch) < 2:
        raise HTTPException(status_code=400, detail="At least two reports are required.")

    summaries = []
    sources = []

    for report in textfetch:
        if not isinstance(report, dict) or "text" not in report or "source" not in report:
            raise HTTPException(status_code=400, detail="Each report must have 'text' and 'source' keys.")

        summary = summarize_text(report["text"])
        summaries.append(summary)
        sources.append(report["source"])

    # Generate AI response
    ai_response = generate_ai_answer("what are the key differencies", summaries[0], summaries[1])

    # Create final response
    final_response = {
        "sources": sources,
        "summaries": summaries,
        "ai_analysis": ai_response
    }

    return final_response
# PDF Processing Function

def extract_text_from_pdf(pdf_file: UploadFile):
    """Extract text from PDF file."""
    global text2
    pdf_data = pdf_file.file.read()

    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text_sections = [page.get_text("text") for page in doc]


    text2 = "\n".join(text_sections)

    return text_sections


summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)


class ReportRequest(BaseModel):
    textfetch: list
    prompt: str = "Summarize the reports"


def summarize_text(text):
    """Summarizes long market reports."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]


@app.get("/analyze_reports/")
def analyze_reports():
    """Summarizes and performs prompt-based analysis on reports."""
    global textfetch
    if len(textfetch) < 2:
        raise HTTPException(status_code=400, detail="At least two reports are required.")

    summaries = []
    sources = []

    for report in textfetch:
        if not isinstance(report, dict) or "text" not in report or "source" not in report:
            raise HTTPException(status_code=400, detail="Each report must have 'text' and 'source' keys.")

        summary = summarize_text(report["text"])
        summaries.append(summary)
        sources.append(report["source"])

    # Create final response
    final_response = {
        "sources": sources,
        "summaries": summaries,
        "custom_analysis": f"Performing 'Compare the textfetch texts of different sources'...",
        "custom_insights": f"Insights generated based on 'The response is' [COMING SOON]"
    }

    return final_response
# Generate Embeddings
def generate_embeddings(text: str):
    """Generate embeddings using Sentence Transformers."""
    return embedding_model.encode(text, convert_to_numpy=True)
class MarketReport(BaseModel):
    text1: str
    text2: str

def extract_entities(text):
    """Extracts named entities from text using spaCy."""

    entities = {"ORG": [], "PERSON": [], "PRODUCT": [], "GPE": []}  # Organization, People, Product, Locations
    ner_result = nlp(text)
    for entity in ner_result:
        label = entity["entity"].split("_")[-1]  # Extract NER label
        word = entity["word"].replace("##", "")  # Fix subword tokenization
        if label in entities:
            entities[label].append(word)

    return {key: list(set(values)) for key, values in entities.items()}  # Remove duplicates


def extract_financial_numbers(text):
    """Extracts financial figures and key performance indicators (KPIs)."""
    financial_data = {
        "Revenue": re.findall(r"(\$\d+[\d,\.]*)\s*(million|billion|trillion)?\s*revenue", text, re.IGNORECASE),
        "Profit/Loss": re.findall(r"(\$\d+[\d,\.]*)\s*(million|billion|trillion)?\s*(profit|loss)", text, re.IGNORECASE),
        "Market Share": re.findall(r"(\d+\.?\d*)%\s*market share", text, re.IGNORECASE),
        "Stock Price": re.findall(r"\$\d+[\d,\.]*\s*(?:per share|stock price)", text, re.IGNORECASE)
    }
    return financial_data

@app.get("/reportgen/")
def compare_market_reports():
    """Compares two market reports for Named Entity Recognition (NER) and Financial Analysis."""
    global textfetch
    # Extract Named Entities
    entities1 = extract_entities(textfetch[0]["text"])
    entities2 = extract_entities(textfetch[1]["text"])

    # Extract Financial Figures
    finance1 = extract_financial_numbers(textfetch[0]["text"])
    finance2 = extract_financial_numbers(textfetch[1]["text"])

    # Compute text similarity
    embedding1 = embedding_model.encode(textfetch[0]["text"], convert_to_tensor=True)
    embedding2 = embedding_model.encode(textfetch[1
                                        ]["text"], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Format Output
    comparison_result = {
        "Named Entity Comparison": {
            "Report 1 Entities": entities1,
            "Report 2 Entities": entities2
        },
        "Financial KPI Comparison": {
            "Report 1 Financials": finance1,
            "Report 2 Financials": finance2
        },
        "Similarity Score": round(similarity_score, 4),
        "Key Differences": {
            "Companies Focused": list(set(entities1["ORG"]) ^ set(entities2["ORG"])),
            "Financial Discrepancies": {
                "Revenue": f"Report 1 mentions {finance1['Revenue']} while Report 2 mentions {finance2['Revenue']}",
                "Profit/Loss": f"Report 1 mentions {finance1['Profit/Loss']} while Report 2 mentions {finance2['Profit/Loss']}",
                "Market Share": f"Report 1 mentions {finance1['Market Share']} while Report 2 mentions {finance2['Market Share']}",
                "Stock Price": f"Report 1 mentions {finance1['Stock Price']} while Report 2 mentions {finance2['Stock Price']}",
            }
        }
    }

    return comparison_result

@app.get("/")
def home():
    return {"message": "Welcome to the RAG Market Report API"}
text = []
@app.post("/upload_reports/")
async def upload_reports(files: List[UploadFile] = File(...)):
    """Process and store PDF embeddings."""
    global documents, index
    global text2,textfetch
    text2 = ''  # Reset previous data
    textfetch = []
    for file in files:
        sections = extract_text_from_pdf(file)
        text.append(sections)
        pdf_data = file.file.read()
        print(file)
        pdf = file.file.read()
        textfetch.append({"text": " ".join(sections), "source": file.filename})
        print(textfetch)


    for section in sections:
            documents.append({"text": section, "source": file.filename})

            # Generate embeddings
            embeddings = generate_embeddings(section)

            # Reshape for FAISS
            embeddings = embeddings.reshape(1, -1)
            index.add(embeddings)



    return {"message": "Reports uploaded and processed successfully."}
@app.get('/hi/')
def hi():
    return text[1]
def compare_reports_hf(reports: List[Dict[str, str]]):
    """Compares multiple reports using a Hugging Face summarization model."""
    try:
        # Merge text from all reports
        combined_text = "\n\n---\n\n".join([f"Source: {r['source']}\nText: {r['text']}" for r in reports])

        # Tokenize and truncate properly
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        # Generate AI-based summary using `max_new_tokens`
        response = generator(truncated_text, max_new_tokens=150, min_length=50, do_sample=False)

        if not response or "summary_text" not in response[0]:
            raise HTTPException(status_code=500, detail="Summarization model failed to generate output.")

        return response[0]["summary_text"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face API error: {str(e)}")



# def compare_reports_hf(reports: List[Dict[str, str]]):
#     """Compares multiple reports using a Hugging Face model."""
#     try:
#         # Merge text from all reports
#         combined_text = "\n\n---\n\n".join([f"Source: {r['source']}\nText: {r['text']}" for r in reports])
#         combined_text = combined_text[:1000]
#         # Define the comparison query
#         prompt = f"Summarise the following document:\n\n{combined_text}"
#
#
#         # Generate AI response
#         response = generator(prompt,max_length=1024, truncation=False)[0]["generated_text"]
#
#         return response
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Hugging Face API error: {str(e)}")

#
# @app.get("/compare_pdf/")
# async def compare_reports():
#     """API endpoint to compare reports using Hugging Face summarization."""
#     if not textfetch or len(textfetch) < 1:
#         raise HTTPException(status_code=400, detail="At least two reports are required for comparison.")
#
#     # Get AI-based comparison
#     comparison_result = compare_reports_hf(textfetch)
#
#     return {"comparison": comparison_result}

class TextRequest(BaseModel):
    text: str


def summarize_text(text, max_tokens=1024):
    """Summarizes long market reports efficiently."""
    inputs = tokenizer.encode(text, truncation=True, max_length=1024)  # Fixed bug here
    truncated_text = tokenizer.decode(inputs, skip_special_tokens=True)  # Fixed bug here
    max_summary_length = min(len(inputs) // 2, 150)  # Ensure summary length isn't too long

    summary = generator(truncated_text, max_new_tokens=150, min_length=0, do_sample=False)[0]["summary_text"]
    return summary

def analyze_sentiment(text):
    """Extracts sentiment polarity from a market report."""
    if not text:
        return "Neutral"

    analysis = textblob.TextBlob(text)
    sentiment_score = analysis.sentiment.polarity  # -1 (negative) to +1 (positive)
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"


@app.get('/marketreports')
def compare_market_reports():
    """Compares two market reports and generates AI-driven insights."""
    global textfetch

    try:
        if not textfetch or len(textfetch) < 2:
            return {"error": "At least two reports are required for comparison."}

        summaries = []
        embeddings = []
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Load a sentence transformer model

        # Step 1: Summarize each report & analyze sentiment
        for report in textfetch:
            if not isinstance(report, dict) or "text" not in report or "source" not in report:
                return {"error": "Invalid report format. Each item must be a dictionary with 'text' and 'source' keys."}

            summary = summarize_text(report["text"])  # Fixed summarization
            sentiment = analyze_sentiment(summary)  # Fixed sentiment analysis

            summaries.append({"source": report["source"], "summary": summary, "sentiment": sentiment})

            # Convert summaries to embeddings for AI-based similarity
            embeddings.append(model.encode(summary, convert_to_tensor=True))

        # Step 2: Compare reports using AI similarity
        if len(embeddings) < 2:
            return {"error": "Insufficient embeddings for comparison."}

        similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        insights = [{
            "report_1": summaries[0]["source"],
            "report_2": summaries[1]["source"],
            "similarity_score": round(similarity_score, 4),
            "key_difference": f"Report '{summaries[0]['source']}' highlights '{summaries[0]['summary'][:100]}...' while '{summaries[1]['source']}' focuses on '{summaries[1]['summary'][:100]}...'.",
        }]

        # Step 3: Generate AI-Powered Insights
        report_summary = {
            "summarized_reports": summaries,
            "comparative_insights": insights
        }

        return report_summary

    except Exception as e:
        return {"error": f"Error in market report comparison: {str(e)}"}

def generate_word_cloud(text: str):
    """Generate a word cloud from the given text and return an image stream."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")

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


def generate_word_cloud(text: str):
    """Generate a word cloud from the given text and return an image stream."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")

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


@app.post("/generate_wordcloud/")
async def generate_wordcloud(request: ReportRequest):
    """Generate a word cloud from two reports and return the image."""

    # Retrieve stored text
    text1 = "\n".join([doc["text"] for doc in documents if doc["source"] == request.report1])
    text2 = "\n".join([doc["text"] for doc in documents if doc["source"] == request.report2])

    if not text1 or not text2:
        raise HTTPException(status_code=404, detail="One or both reports not found.")

    # Combine text for word cloud
    combined_text = text1 + " " + text2

    # Generate word cloud image
    img_stream = generate_word_cloud(combined_text)

    return StreamingResponse(img_stream, media_type="image/png")

def compute_similarity(text1, text2):
    """Compute TF-IDF Cosine Similarity using PyTorch Tensors."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2]).toarray()  # Convert to dense NumPy array

    # ✅ Convert to PyTorch tensors
    tensor1 = torch.tensor(tfidf_matrix[0], dtype=torch.float32)
    tensor2 = torch.tensor(tfidf_matrix[1], dtype=torch.float32)

    # ✅ Compute cosine similarity in PyTorch
    similarity = torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))

    return round(float(similarity.item()), 4)  # Convert tensor to float for JSON response


def extract_text_from_pdf2(file: UploadFile):
    """Extract text from a PDF file."""
    try:
        pdf_data = file.file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = " ".join([page.get_text("text") for page in doc])
        if not any(text):
            print("warning there is no text")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_similarity(text1, text2):
    """Compute similarity between two texts using NLP embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings =  embedding_model.encode(text1, convert_to_numpy=True).reshape(1, -1)
    embeddings2 = embedding_model.encode(text2, convert_to_numpy=True).reshape(1, -1)
    similarity = cosine_similarity(embeddings, embeddings2)[0][0]

    return float(similarity)
@app.post("/similarity/")
async def similarity(files: List[UploadFile] = File(...)):
    # Extract text from PDFs
    text1 = extract_text_from_pdf2(files[0])
    text3 = extract_text_from_pdf2(files[1])
    similarity = compute_semantic_similarity(text1, text3)
    return {"similarity": round(float(similarity), 4)}

@app.post("/check_plagiarism/")
async def check_plagiarism(files: List[UploadFile] = File(...)):
    """Check plagiarism similarity between two uploaded PDF reports."""
    try:
        # Extract text from PDFs
        text1 = extract_text_from_pdf2(files[0])
        text3 = extract_text_from_pdf2(files[1])

        print(type(text3) )
        if not text1.strip() or not text3.strip():
            return {"error": "One or both files are empty."}

        # Compute similarity score
        similarity_score = compute_similarity(text1, text3)
        similarity = compute_semantic_similarity(text1, text3)
        return {
            "message": "Plagiarism check completed",
            "similarity_score": similarity_score,
            "interpretation": f"{similarity_score * 100}% similarity detected.",
            "similarity": similarity
        }
    except Exception as e:
        print(e)
        return {"error": str(e)}

def generate_word_cloud_from_text(text1):
    """Generate a word cloud from the given text and return an image stream."""
    if not text1.strip():
        return None

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text1)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Generated Word Cloud", fontsize=14)

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png")
    plt.close()
    img_stream.seek(0)

    return img_stream

@app.get("/generate_dynamic_wordcloud/")
async def generate_dynamic_wordcloud():
    """Generate a word cloud from the dynamically updated text2 variable."""

    global text2
    print(text2)
    if not text2:
        raise HTTPException(status_code=404, detail="No text data available to generate word cloud.")

    combined_text = (text2)
    img_stream = generate_word_cloud_from_text(combined_text)

    return StreamingResponse(img_stream, media_type="image/png")

@app.post("/query/")
async def query_reports(request: QueryRequest):
    """Retrieve relevant text and generate insights using RAG."""
    query_embedding = generate_embeddings(request.query).reshape(1, -1)

    # Ensure FAISS has data
    if index.ntotal == 0:
        return {"query": request.query, "response": "No data available for search."}

    # Search FAISS for similar sections
    D, I = index.search(query_embedding, k=3)

    # Retrieve matching sections
    relevant_sections = [
        documents[i] for i in I[0] if 0 <= i < len(documents)
    ]

    # Prepare context for RAG model
    context_text = "\n\n".join([f"Source: {doc['source']}\n{doc['text']}" for doc in relevant_sections])

    # Generate AI-powered response
    ai_response = rag_model(
        f"Given the following document context, answer: {request.query}\n\n{context_text}",
        max_length=500,
        num_return_sequences=1
    )[0]['generated_text']

    return {
        "query": request.query,
        "response": ai_response,
        "sources": [doc["source"] for doc in relevant_sections]
    }

@app.post("/compare_reports/")
async def compare_reports(request: QueryRequest):
    """Compare insights from multiple reports based on a query."""
    query_embedding = generate_embeddings(request.query).reshape(1, -1)

    if index.ntotal == 0:
        return {"query": request.query, "response": "No data available for comparison."}

    # Retrieve top 5 relevant sections
    D, I = index.search(query_embedding, k=5)

    report_sections = {}
    for i in I[0]:
        if 0 <= i < len(documents):
            source = documents[i]["source"]
            text = documents[i]["text"]
            if source not in report_sections:
                report_sections[source] = []
            report_sections[source].append(text)

    # Format comparative analysis
    comparison_text = "\n\n".join(
        [f"**{source}**:\n{'\n'.join(sections)}" for source, sections in report_sections.items()]
    )

    return {
        "query": request.query,
        "comparison": comparison_text,
        "sources": list(report_sections.keys())
    }



# Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Vector Store
embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
index = faiss.IndexFlatL2(embedding_dim)
documents = []  # Store document text and metadata

# Load Transformer Pipelines
text_generator = pipeline("text-generation", model="AI-Sweden-Models/gpt-sw3-356m")
summarizer = pipeline("summarization", model="google/flan-t5-base")

# Load Chatbot Model (e.g., Mistral-7B or LLaMA-2-7B)
chatbot_model = AutoModelForCausalLM.from_pretrained("google/gemma-2bkk")
tok = AutoTokenizer.from_pretrained("google/gemma-2b")
# Conversation Memory (Stores last 5 exchanges per user)
conversation_history = {}

class QueryRequest(BaseModel):
    query: str
    user_id: str  # Unique user ID for tracking conversation history

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
    inputs = tok(request.query, return_tensors="pt")
    outputs = chatbot_model.generate(**inputs)
    print(outputs)
    print(tok.batch_decode(outputs, skip_special_tokens=True))
    ['Pour a cup of bolognese into a large bowl and add the pasta']
    # # Retrieve conversation history for the user
    # user_id = request.user_id
    # if user_id not in conversation_history:
    #     conversation_history[user_id] = deque(maxlen=5)  # Store last 5 exchanges
    #
    # # Prepare chat history
    # chat_history = "\n".join(conversation_history[user_id])
    # chat_prompt = f"{chat_history}\nUser: {request.query}\nAI:"
    #
    # # Generate AI chatbot response
    # ai_response = chatbot_model(chat_prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    #
    # # Store conversation history
    # conversation_history[user_id].append(f"User: {request.query}")
    # conversation_history[user_id].append(f"AI: {ai_response}")

    return {
        "query": request.query,
        "response": tok.batch_decode(outputs, skip_special_tokens=True)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
