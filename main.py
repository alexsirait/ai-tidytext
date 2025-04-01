from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

# Konfigurasi API Key Gemini
GEMINI_API_KEY = "AIzaSyDlwUdJBjSPOghzrgsUhLfs54Md5ywn2Dc"
genai.configure(api_key=GEMINI_API_KEY)

# Model untuk input request body
class TextInput(BaseModel):
    text: str

# Fungsi untuk merapikan teks menggunakan Gemini API
def process_text_with_gemini(input_text: str):
    try:
        # Inisialisasi model Gemini (gunakan model yang tersedia, misalnya 'gemini-1.5-flash')
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt untuk merapikan teks dan menerjemahkan ke bahasa Inggris
        prompt = f"""
            AI Instructions:
            - Refine the provided text and translate it into English seamlessly.
            - Deliver a moderately detailed response structured with concise, clear bullet points.
            - Emphasize key details from the input while excluding irrelevant or redundant information.
            - Ensure the entire response is in English, with no introductory fluff.

            Input text: '{input_text}'
        """
        
        # Panggil API Gemini untuk memproses teks
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Error processing text with Gemini API: {str(e)}")

# Endpoint API
@app.post("/tidy-text/")
async def tidy_text(input: TextInput):
    try:
        if not input.text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Proses teks dengan Gemini API
        result = process_text_with_gemini(input.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint untuk testing
@app.get("/")
async def root():
    return {"message": "API is running. Use POST /tidy-text/ with a JSON body containing 'text'"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)