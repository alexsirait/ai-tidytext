# Text Tidy API

A FastAPI-based service that uses Google's Gemini AI to refine, translate, and analyze text with comprehensive insights.

## Features

- Text refinement and translation using Google's Gemini AI
- Multiple analysis types:
  - Comprehensive analysis with key points, context, and implications
  - Concise text summarization
  - Sentiment and emotional tone analysis
  - Entity extraction (people, organizations, locations, etc.)
  - Combined analysis with all features
- Structured data extraction for easy integration
- Rate limiting to prevent abuse
- Caching for improved performance
- Comprehensive error handling
- Async processing for better scalability
- Health check endpoint

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Running the API

Start the server with:
```
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /tidy-text/

Process, translate, and analyze text.

**Request Body:**
```json
{
  "text": "Your text to process",
  "language": "en",  // Optional, defaults to "en"
  "analysis_type": "comprehensive"  // Optional, defaults to "comprehensive"
}
```

**Analysis Types:**
- `comprehensive`: Full analysis with key points, context, and implications
- `summary`: Concise summary of the main points
- `sentiment`: Analysis of emotional tone and sentiment
- `entities`: Extraction of key entities (people, organizations, etc.)
- `all`: Combination of all analysis types

**Response:**
```json
{
  "result": "Processed and analyzed text with detailed insights",
  "processing_time": 1.23,
  "model_used": "gemini-1.5-flash",
  "analysis": {
    "main_content": "Refined and translated version of the text",
    "key_points": [
      "Key point 1",
      "Key point 2",
      "Key point 3"
    ],
    "context": "Relevant background information",
    "implications": "Significance and potential impact",
    "related_topics": [
      "Related topic 1",
      "Related topic 2"
    ]
  }
}
```

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

### GET /

Root endpoint with API information.

**Response:**
```json
{
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
```

## Rate Limiting

The API is rate-limited to 10 requests per minute per IP address.

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (e.g., empty text, invalid analysis type)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error
- 504: Gateway Timeout (AI service timeout) 