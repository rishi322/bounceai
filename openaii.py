from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store extracted text
text2 = []
async def update_sections(new_sections: List[str]):
    """Update the global text2 variable dynamically."""
    global text2
    if not new_sections:
        raise HTTPException(status_code=400, detail="No new text provided.")

    text2 = new_sections
    return {"message": "Sections updated successfully", "current_text": text2}

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

@app.post("/upload_reports/")
async def upload_reports(files: List[UploadFile] = File(...)):
    """Extract text from uploaded PDFs and store it globally."""
    global text2
    text2 = []  # Reset previous data

    for file in files:
        pdf_data = file.file.read()
        text2.append(pdf_data.decode("utf-8", errors="ignore"))
        print(text2)
    return {"message": "Reports uploaded and processed successfully."}

@app.get("/generate_dynamic_wordcloud/")
async def generate_dynamic_wordcloud():
    """Generate a word cloud from the dynamically updated text2 variable."""
    global text2
    if not text2:
        raise HTTPException(status_code=404, detail="No text data available to generate word cloud.")

    combined_text = " ".join(text2)
    img_stream = generate_word_cloud_from_text(combined_text)

    return StreamingResponse(img_stream, media_type="image/png")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)