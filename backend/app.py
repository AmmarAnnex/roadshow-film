from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import shutil
from pathlib import Path
from utils.video_processor import CinemaProcessor

app = FastAPI(title="Roadshow Film API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = CinemaProcessor()

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    settings: str = Form(default='{"intensity":0.8,"exposure":0.5,"color":0.7}')
):
    """Process uploaded video with cinema transformation"""
    try:
        # Parse settings
        video_settings = json.loads(settings)
        
        # Validate file type
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, f"input_{video.filename}")
        
        # Save uploaded video
        with open(input_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Process video
        output_path = processor.transform_video(
            input_path, 
            intensity=video_settings.get('intensity', 0.8),
            exposure=video_settings.get('exposure', 0.5),
            color=video_settings.get('color', 0.7)
        )
        
        # Return processed video
        return FileResponse(
            output_path,
            media_type='video/mp4',
            filename='roadshow-enhanced.mp4',
            background=lambda: shutil.rmtree(temp_dir, ignore_errors=True)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": processor.model is not None,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Roadshow Film API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)