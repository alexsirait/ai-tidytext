from fastapi import FastAPI, HTTPException, Request,Query
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

# Process text with Gemini API with customizable prompt
async def process_text_with_gemini(input_text: str, language: str, analysis_type: str, prompt_template: str) -> Dict[str, Any]:
    prompt = prompt_template.format(input_text=input_text, language=language, analysis_type=analysis_type)
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

# Endpoint: /api/tidy-text
@app.post("/api/tidy-text")
async def tidy_text(input: TextInput):
    prompt_template = (
        "AI Instructions:\n"
        "- Refine the provided text and translate it into English seamlessly.\n"
        "- Deliver a moderately detailed response structured with concise, clear bullet points.\n"
        "- Emphasize key details from the input while excluding irrelevant or redundant information.\n"
        "- Ensure the entire response is in English, with no introductory fluff.\n\n"
        "Input text {language}: {input_text}"
    )
    try:
        start_time = time.time()
        result = await process_text_with_gemini(input.text, input.language, input.analysis_type, prompt_template)
        return {
            "result": result["text"],
            "processing_time": time.time() - start_time,
            "model_used": "gemini-1.5-flash"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: /api/project-overview

@app.post("/api/project-overview")
async def project_overview(input: TextInput):
    project_api_url = "http://192.168.88.62:40000/api/project_overview/get_project_by_share/zxshnxnmkh"
    params = {
        "page": 1,
        "page_size": 9999,
        "sort_by": "highest_progress",
        "department_code": "GA22",
        "year": "2025",
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(project_api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        project_data = response.json()

        # Ambil list rows di dalam data
        projects = project_data["data"][0]["related"]["rows"]

        # Ambil field yang diperlukan dari setiap proyek
        filtered_projects = [
            {
                "title": p.get("title", "-"),
                "project_status": p.get("project_status", "-"),
                "project_progress_status": p.get("project_progress_status", "-"),
                "project_progress": p.get("project_progress", 0),
                "start_date" : p.get("start_date", "-"),
                "end_date": p.get("end_date", "-"),
            }
            for p in projects
        ]

        project_info = ""
        for idx, p in enumerate(filtered_projects, start=1):
            project_info += (
                f"{idx}. *{p['title']}*\n"
                f"   • Status: {p['project_status']}\n"
                f"   • Start Date: {p['start_date']}\n"
                f"   • End Date: {p['end_date']}\n"                
                f"   • Progress Status: {p['project_progress_status']}\n"
                f"   • Progress: {p['project_progress']}%\n\n"

            )

        prompt_template = (
            "Project Overview Instructions:\n"
            "- Below is a list of project details.\n"
            "- Extract key project information based on the user's question.\n"
            "- Identify the project name from the question (e.g., 'meeting room') and match it to the project data.\n"
            "- Identify the project name or category from the user's question (e.g., 'meeting room', 'ai cob') and match it to the project data as closely as possible.\n"
            "- Perform a substring match: If the user's question contains part of a project name (e.g., 'ai cob', 'COB Battery'), find the most relevant project even if the name isn't exact.\n"
            "- Keep the response natural and clear: avoid technical phrases like 'Based on the user's question...' or 'Found projects...'. Simply answer the question directly and clearly.\n"
            "- If the project is found, return the progress details clearly.\n"
            "- Show all with same character (e.g., 'meeting room') and show all in the project data.\n"
            "- Keep your response concise and informative, highlighting the most important aspects.\n"
            "- You can answer all language\n"
            "- Thinking deep what user want to talk\n"
            "- If the project is not found, return a response saying the project was not found.\n\n"
            "User Question:\n{user_question}\n\n"
            "Project Data:\n{project_info}"
        )


        start_time = time.time()
        prompt = prompt_template.format(user_question=input.text, project_info=project_info )
        result = await process_text_with_gemini(
            input_text=prompt,
            language=input.language,
            analysis_type=input.analysis_type,
            prompt_template="{input_text}"
        )

        return {
            "result": result["text"],
            "processing_time": time.time() - start_time,
            "model_used": "gemini-1.5-flash"
        }

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Request to project data API timed out")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch project data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@app.get("/api/project")
async def project_overview(
    department_code: str = Query("GA22"),
    year: str = Query("2025")
):
    project_api_url = "http://192.168.88.62:40000/api/project_overview/get_project_by_share/zxshnxnmkh"
    params = {
        "page": 1,
        "page_size": 9999,
        "sort_by": "highest_progress",
        "department_code": department_code,
        "year": year,
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(project_api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        project_data = response.json()

        # Ambil list rows di dalam data
        projects = project_data["data"][0]["related"]["rows"]

        # Ambil field yang diperlukan dari setiap proyek
        filtered_projects = [
            {
                "title": p.get("title", "-"),
                "project_status": p.get("project_status", "-"),
                "project_progress_status": p.get("project_progress_status", "-"),
                "project_progress": p.get("project_progress", 0),
                "start_date" : p.get("start_date", "-"),
                "end_date": p.get("end_date", "-"),
            }
            for p in projects
        ]

        project_info = ""
        for idx, p in enumerate(filtered_projects, start=1):
            project_info += (
                f"{idx}. *{p['title']}*\n"
                f"   • Status: {p['project_status']}\n"
                f"   • Start Date: {p['start_date']}\n"
                f"   • End Date: {p['end_date']}\n"                
                f"   • Progress Status: {p['project_progress_status']}\n"
                f"   • Progress: {p['project_progress']}%\n\n"
            )

        return {"result": project_info}

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Request to project data API timed out")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch project data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
