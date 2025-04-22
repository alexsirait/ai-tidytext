from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
import os
import time
import json
from typing import Optional, Dict, Any, List
from functools import lru_cache
import asyncio
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
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Request rate limiting
RATE_LIMIT = 10  # requests per minute
rate_limit_dict: Dict[str, list] = {}

# Input and output models
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to be processed")
    language: Optional[str] = Field("en", description="Target language for translation")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis: 'comprehensive', 'summary', 'sentiment', 'entities', or 'all'")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace")
        return v.strip()
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ["comprehensive", "summary", "sentiment", "entities", "all"]
        if v not in valid_types:
            raise ValueError(f"Analysis type must be one of: {', '.join(valid_types)}")
        return v

class TextResponse(BaseModel):
    result: str
    processing_time: float
    model_used: str
    analysis: Optional[Dict[str, Any]] = None

# Cache for processed texts
@lru_cache(maxsize=100)
def get_cached_result(text: str, language: str, analysis_type: str) -> Optional[Dict[str, Any]]:
    return None

# Rate limiting middleware
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean up old requests
    if client_ip in rate_limit_dict:
        rate_limit_dict[client_ip] = [t for t in rate_limit_dict[client_ip] if current_time - t < 60]
    
    # Check rate limit
    if client_ip in rate_limit_dict and len(rate_limit_dict[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Add current request
    if client_ip not in rate_limit_dict:
        rate_limit_dict[client_ip] = []
    rate_limit_dict[client_ip].append(current_time)
    
    response = await call_next(request)
    return response

app.middleware("http")(rate_limit_middleware)

# Process text with Gemini API
async def process_text_with_gemini(input_text: str, language: str = "en", analysis_type: str = "comprehensive") -> Dict[str, Any]:
    start_time = time.time()
    
    # Check cache first
    cache_key = f"{input_text}:{language}:{analysis_type}"
    cached_result = get_cached_result(input_text, language, analysis_type)
    if cached_result:
        return cached_result
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Enhanced prompt based on analysis type
        if analysis_type == "comprehensive":
            prompt = f"""
            AI Instructions:
            - Refine the provided text and translate it into {language} seamlessly.
            - Deliver a comprehensive analysis with the following sections:
              1. Main Content: A refined and translated version of the text
              2. Key Points: Extract 3-5 most important points from the text
              3. Context: Provide relevant context or background information
              4. Implications: Explain the significance or implications of the content
              5. Related Topics: Suggest 2-3 related topics for further exploration
            
            - Format the output as a structured response with clear section headers
            - Ensure the entire response is in {language}
            - Maintain the original meaning and context
            - Be concise but informative

            Input text: '{input_text}'
            """
        elif analysis_type == "summary":
            prompt = f"""
            AI Instructions:
            - Create a concise summary of the provided text in {language}
            - Extract the most important information and key takeaways
            - Organize the summary in a clear, logical structure
            - Include any relevant statistics, facts, or figures
            - Keep the summary to approximately 25% of the original length
            - Ensure the summary is in {language}

            Input text: '{input_text}'
            """
        elif analysis_type == "sentiment":
            prompt = f"""
            AI Instructions:
            - Analyze the sentiment and emotional tone of the provided text
            - Provide a detailed sentiment analysis with the following components:
              1. Overall Sentiment: Positive, negative, or neutral
              2. Emotional Intensity: High, medium, or low
              3. Key Emotions: Identify the primary emotions expressed
              4. Tone Analysis: Describe the tone (formal, informal, technical, etc.)
              5. Supporting Evidence: Quote specific parts of the text that support your analysis
            
            - Format the output as a structured response with clear section headers
            - Ensure the entire response is in {language}
            - Be objective and analytical

            Input text: '{input_text}'
            """
        elif analysis_type == "entities":
            prompt = f"""
            AI Instructions:
            - Extract and analyze key entities from the provided text
            - Identify and categorize the following:
              1. People: Names of individuals mentioned
              2. Organizations: Companies, institutions, or groups
              3. Locations: Places, countries, or geographical references
              4. Dates/Times: Temporal references
              5. Concepts: Key ideas or themes
              6. Technical Terms: Specialized vocabulary or jargon
            
            - For each entity, provide:
              - The entity name
              - Its category
              - Its significance in the context of the text
              - Any relationships to other entities
            
            - Format the output as a structured response with clear section headers
            - Ensure the entire response is in {language}
            - Be thorough but concise

            Input text: '{input_text}'
            """
        else:  # "all" analysis type
            prompt = f"""
            AI Instructions:
            - Perform a comprehensive analysis of the provided text in {language}
            - Provide the following sections in your response:
              1. Refined Text: A polished and translated version of the original text
              2. Summary: A concise summary of the main points (25% of original length)
              3. Key Entities: People, organizations, locations, dates, and concepts
              4. Sentiment Analysis: Overall sentiment, emotional intensity, and tone
              5. Context: Relevant background information
              6. Implications: Significance and potential impact
              7. Related Topics: Suggestions for further exploration
            
            - Format each section with clear headers
            - Ensure the entire response is in {language}
            - Be comprehensive but well-organized

            Input text: '{input_text}'
            """
        
        # Generate content with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=30.0
        )
        
        result = response.text.strip()
        
        # For comprehensive analysis, try to extract structured data
        if analysis_type in ["comprehensive", "all"]:
            try:
                # Try to extract structured data from the response
                structured_data = {
                    "main_content": "",
                    "key_points": [],
                    "context": "",
                    "implications": "",
                    "related_topics": []
                }
                
                # Simple parsing logic - in a real app, you might use more sophisticated NLP
                sections = result.split("\n\n")
                current_section = None
                
                for section in sections:
                    if "Main Content:" in section or "Refined Text:" in section:
                        current_section = "main_content"
                        structured_data["main_content"] = section.split(":", 1)[1].strip()
                    elif "Key Points:" in section:
                        current_section = "key_points"
                        points_text = section.split(":", 1)[1].strip()
                        structured_data["key_points"] = [p.strip() for p in points_text.split("\n") if p.strip()]
                    elif "Context:" in section:
                        current_section = "context"
                        structured_data["context"] = section.split(":", 1)[1].strip()
                    elif "Implications:" in section:
                        current_section = "implications"
                        structured_data["implications"] = section.split(":", 1)[1].strip()
                    elif "Related Topics:" in section:
                        current_section = "related_topics"
                        topics_text = section.split(":", 1)[1].strip()
                        structured_data["related_topics"] = [t.strip() for t in topics_text.split("\n") if t.strip()]
                    elif current_section and section.strip():
                        # Append to current section if we're in the middle of one
                        if current_section == "main_content":
                            structured_data["main_content"] += "\n" + section
                        elif current_section == "context":
                            structured_data["context"] += "\n" + section
                        elif current_section == "implications":
                            structured_data["implications"] += "\n" + section
                
                return {
                    "text": result,
                    "structured_data": structured_data
                }
            except Exception:
                # If structured parsing fails, just return the text
                return {"text": result}
        
        return {"text": result}
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request to AI service timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

# API endpoints
@app.post("/tidy-text/", response_model=TextResponse)
async def tidy_text(input: TextInput):
    try:
        start_time = time.time()
        
        # Process text with Gemini API
        result = await process_text_with_gemini(input.text, input.language, input.analysis_type)
        
        processing_time = time.time() - start_time
        
        response_data = {
            "result": result["text"],
            "processing_time": processing_time,
            "model_used": "gemini-1.5-flash"
        }
        
        # Add structured analysis if available
        if "structured_data" in result:
            response_data["analysis"] = result["structured_data"]
        
        return TextResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Text Tidy API is running",
        "endpoints": {
            "tidy_text": "POST /tidy-text/ - Process and translate text",
            "health": "GET /health - Check API health"
        },
        "analysis_types": [
            "comprehensive - Full analysis with key points, context, and implications",
            "summary - Concise summary of the main points",
            "sentiment - Analysis of emotional tone and sentiment",
            "entities - Extraction of key entities (people, organizations, etc.)",
            "all - Combination of all analysis types"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000
    )