import os
import shutil
import tempfile
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# --- 🚨 Import the RAG logic from the new file 🚨 ---
from rag_logic import setup_and_invoke_rag_chain 
# --- 🚨 No need for LangChain/Groq imports here anymore! 🚨 ---


# --- FastAPI App Setup ---
app = FastAPI(title="Document Summarizer Backend")

# Configure CORS
origins = [
    "http://localhost",
    "http://127.0.0.1:8000", # If you are running the frontend via a specific host/port
    "http://127.0.0.1:3000", # Example for Live Server extension
    "*" # Use '*' temporarily for development, but be specific in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End FastAPI App Setup ---

@app.post("/summarize_uploaded_document/")
async def summarize_document_endpoint(
    question: Annotated[str, Form(description="The question to ask about the document.")],
    file: UploadFile = File(description="The PDF document to be summarized.")
):
    """
    Handles file upload, delegates RAG processing, and cleans up the temporary file.
    """
    
    # --- 0. Input Validation ---
    if file.content_type not in ['application/pdf', 'text/plain']:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF and TXT files are accepted."
        )

    tmp_path = None
    
    try:
        # --- 1. Save the file temporarily ---
        suffix = ".pdf" if file.content_type == 'application/pdf' else ".txt"
        
        # The 'with' block ensures the temporary file is created and closed, 
        # preventing Windows locking issues (WinError 32).
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            shutil.copyfileobj(file.file, tmp_file)

        # --- 2. Delegate Logic to rag_logic.py ---
        # The function call now goes to the external file
        result = setup_and_invoke_rag_chain(tmp_path, question)
        
        return {"answer": result}
        
    except Exception as e:
        print(f"Error during processing: {e}")
        # Return a 500 status to the client with the error detail
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred: {e}"
        )
        
    finally:
        # --- 3. Clean up the temporary file ---
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            # print(f"Cleaned up temporary file: {tmp_path}")