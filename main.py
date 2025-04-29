from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import time
import json
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Text Tidy API",
    description="API for refining and translating text using Google's Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Input model
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to be processed")
    language: Optional[str] = Field("en", description="Target language for translation")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis")

# Process text with Gemini API
async def process_text_with_gemini(input_text: str, language: str = "en", analysis_type: str = "comprehensive") -> Dict[str, Any]:
    prompt = f"""AI Instructions:
- Refine the provided text and translate it into English seamlessly.
- Deliver a moderately detailed response structured with concise, clear bullet points.
- Emphasize key details from the input while excluding irrelevant or redundant information.
- Ensure the entire response is in English, with no introductory fluff.

Input text {language}: {input_text}"""
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    try:
        response = requests.post(
            f"{api_endpoint}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            result = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"text": result}
        else:
            raise HTTPException(status_code=500, detail="No valid response from Gemini API")
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Request to AI service timed out")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

# API endpoint
@app.post("/api/tidy-text/")
async def tidy_text(input: TextInput):
    try:
        start_time = time.time()
        result = await process_text_with_gemini(input.text, input.language, input.analysis_type)
        processing_time = time.time() - start_time
        return {
            "result": result["text"],
            "processing_time": processing_time,
            "model_used": "gemini-1.5-flash"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)