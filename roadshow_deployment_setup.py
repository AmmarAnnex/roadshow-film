#!/usr/bin/env python3
"""
Roadshow Film - Complete Deployment Setup Script
Generates all files needed for Vercel frontend + Railway backend deployment
"""

import os
import json
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the complete project structure"""
    print("🏗️ Creating project structure...")
    
    directories = [
        "roadshow-film",
        "roadshow-film/frontend",
        "roadshow-film/frontend/pages",
        "roadshow-film/frontend/pages/api",
        "roadshow-film/frontend/components",
        "roadshow-film/frontend/styles",
        "roadshow-film/frontend/public",
        "roadshow-film/backend",
        "roadshow-film/backend/model",
        "roadshow-film/backend/models",
        "roadshow-film/backend/utils",
        "roadshow-film/.github",
        "roadshow-film/.github/workflows"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def create_frontend_files():
    """Create all frontend files"""
    print("⚛️ Creating frontend files...")
    
    # package.json
    package_json = {
        "name": "roadshow-film",
        "version": "1.0.0",
        "description": "Transform iPhone footage to cinema camera characteristics",
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint"
        },
        "dependencies": {
            "next": "^14.0.0",
            "react": "^18.0.0",
            "react-dom": "^18.0.0",
            "axios": "^1.6.0",
            "@vercel/blob": "^0.15.0",
            "framer-motion": "^10.16.0"
        },
        "devDependencies": {
            "eslint": "^8.0.0",
            "eslint-config-next": "^14.0.0"
        }
    }
    
    with open("roadshow-film/frontend/package.json", "w") as f:
        json.dump(package_json, f, indent=2)
    
    # next.config.js
    next_config = '''/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['sharp']
  },
  api: {
    bodyParser: {
      sizeLimit: '100mb',
    },
    responseLimit: '100mb',
  },
  env: {
    BACKEND_URL: process.env.RAILWAY_STATIC_URL || 'http://localhost:8000'
  }
}

module.exports = nextConfig'''
    
    with open("roadshow-film/frontend/next.config.js", "w") as f:
        f.write(next_config)
    
    # Main page (index.js)
    index_js = '''import { useState } from 'react';
import { motion } from 'framer-motion';
import FilmLens from '../components/FilmLens';
import Controls from '../components/Controls';
import Preview from '../components/Preview';

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processedVideo, setProcessedVideo] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [controls, setControls] = useState({
    intensity: 0.8,
    exposure: 0.5,
    color: 0.7
  });

  const handleFileUpload = async (file) => {
    setUploadedFile(file);
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('video', file);
      formData.append('settings', JSON.stringify(controls));
      
      const response = await fetch('/api/process-video', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const result = await response.blob();
        setProcessedVideo(URL.createObjectURL(result));
      } else {
        console.error('Processing failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="p-8 text-center">
        <motion.h1 
          className="text-6xl font-light mb-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          roadshow <span className="text-orange-400">film</span>
        </motion.h1>
        <p className="text-xl text-gray-400">from mobile to motion picture</p>
      </header>
      
      <main className="container mx-auto px-8">
        <FilmLens 
          onFileUpload={handleFileUpload}
          isProcessing={isProcessing}
        />
        
        <Controls 
          values={controls}
          onChange={setControls}
        />
        
        {processedVideo && (
          <Preview 
            original={uploadedFile}
            processed={processedVideo}
          />
        )}
      </main>
    </div>
  );
}'''
    
    with open("roadshow-film/frontend/pages/index.js", "w") as f:
        f.write(index_js)
    
    # API route
    api_route = '''import { NextRequest, NextResponse } from 'next/server';

export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const backendUrl = process.env.RAILWAY_STATIC_URL || 'http://localhost:8000';
    
    const response = await fetch(`${backendUrl}/process`, {
      method: 'POST',
      body: req.body,
      headers: {
        'Content-Type': req.headers['content-type'],
      },
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    const processedVideo = await response.buffer();
    
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Disposition', 'attachment; filename="roadshow-enhanced.mp4"');
    res.send(processedVideo);
    
  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ error: 'Processing failed', details: error.message });
  }
}'''
    
    with open("roadshow-film/frontend/pages/api/process-video.js", "w") as f:
        f.write(api_route)
    
    # Components
    create_components()
    create_styles()
    
    print("✅ Frontend files created")

def create_components():
    """Create React components"""
    
    # FilmLens component
    film_lens = '''import { useCallback } from 'react';
import { motion } from 'framer-motion';

export default function FilmLens({ onFileUpload, isProcessing }) {
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileUpload(files[0]);
    }
  }, [onFileUpload]);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      onFileUpload(file);
    }
  };

  return (
    <motion.div
      className="border-2 border-dashed border-orange-400 rounded-lg p-12 text-center mb-8"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      {isProcessing ? (
        <div className="space-y-4">
          <div className="animate-spin w-8 h-8 border-2 border-orange-400 border-t-transparent rounded-full mx-auto"></div>
          <p>Transforming your footage...</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="text-6xl">🎬</div>
          <h3 className="text-2xl font-light">Drop your video here</h3>
          <p className="text-gray-400">or click to browse</p>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
            id="file-input"
          />
          <label
            htmlFor="file-input"
            className="inline-block px-6 py-3 bg-orange-400 text-black rounded-lg cursor-pointer hover:bg-orange-300 transition-colors"
          >
            Choose File
          </label>
        </div>
      )}
    </motion.div>
  );
}'''
    
    with open("roadshow-film/frontend/components/FilmLens.js", "w") as f:
        f.write(film_lens)
    
    # Controls component
    controls = '''import { motion } from 'framer-motion';

export default function Controls({ values, onChange }) {
  const handleChange = (key, value) => {
    onChange({
      ...values,
      [key]: value
    });
  };

  const controls = [
    { key: 'intensity', label: 'Cinema Intensity', min: 0, max: 1, step: 0.1 },
    { key: 'exposure', label: 'Exposure Fix', min: 0, max: 1, step: 0.1 },
    { key: 'color', label: 'Color Grade', min: 0, max: 1, step: 0.1 }
  ];

  return (
    <motion.div
      className="bg-gray-900 rounded-lg p-6 mb-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <h3 className="text-xl font-light mb-6">Cinema Controls</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {controls.map(({ key, label, min, max, step }) => (
          <div key={key} className="space-y-2">
            <label className="block text-sm text-gray-400">{label}</label>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={values[key]}
              onChange={(e) => handleChange(key, parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="text-right text-sm text-orange-400">
              {(values[key] * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}'''
    
    with open("roadshow-film/frontend/components/Controls.js", "w") as f:
        f.write(controls)
    
    # Preview component
    preview = '''import { motion } from 'framer-motion';

export default function Preview({ original, processed }) {
  return (
    <motion.div
      className="bg-gray-900 rounded-lg p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
    >
      <h3 className="text-xl font-light mb-6">Preview</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm text-gray-400 mb-2">Original</h4>
          <video
            src={URL.createObjectURL(original)}
            controls
            className="w-full rounded-lg"
          />
        </div>
        <div>
          <h4 className="text-sm text-gray-400 mb-2">Roadshow Enhanced</h4>
          <video
            src={processed}
            controls
            className="w-full rounded-lg"
          />
        </div>
      </div>
      <div className="mt-6 text-center">
        <a
          href={processed}
          download="roadshow-enhanced.mp4"
          className="inline-block px-6 py-3 bg-orange-400 text-black rounded-lg hover:bg-orange-300 transition-colors"
        >
          Download Enhanced Video
        </a>
      </div>
    </motion.div>
  );
}'''
    
    with open("roadshow-film/frontend/components/Preview.js", "w") as f:
        f.write(preview)

def create_styles():
    """Create CSS styles"""
    global_css = '''@tailwind base;
@tailwind components;
@tailwind utilities;

.slider::-webkit-slider-thumb {
  appearance: none;
  height: 20px;
  width: 20px;
  border-radius: 50%;
  background: #fb923c;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  height: 20px;
  width: 20px;
  border-radius: 50%;
  background: #fb923c;
  cursor: pointer;
  border: none;
}'''
    
    with open("roadshow-film/frontend/styles/globals.css", "w") as f:
        f.write(global_css)

def create_backend_files():
    """Create all backend files"""
    print("🐍 Creating backend files...")
    
    # requirements.txt
    requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1.78
numpy==1.24.3
moviepy==1.0.3
python-multipart==0.0.6
rawpy==0.18.1
scikit-image==0.21.0
Pillow==10.1.0
python-dotenv==1.0.0'''
    
    with open("roadshow-film/backend/requirements.txt", "w") as f:
        f.write(requirements)
    
    # main app.py
    app_py = '''from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
    uvicorn.run(app, host="0.0.0.0", port=port)'''
    
    with open("roadshow-film/backend/app.py", "w") as f:
        f.write(app_py)
    
    # Video processor
    video_processor = '''import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import tempfile
import os
from model.cinema_v1_4m import ExposureFixedColorTransform

class CinemaProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained v1.4m model"""
        try:
            model = ExposureFixedColorTransform().to(self.device)
            
            # Try multiple model paths
            model_paths = [
                Path("models/cinema_v1_4m_model.pth"),
                Path("model/cinema_v1_4m_model.pth"),
                Path("../models/cinema_v1_4m_model.pth")
            ]
            
            checkpoint = None
            for path in model_paths:
                if path.exists():
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    break
            
            if checkpoint is None:
                print("Warning: No model checkpoint found, using untrained model")
                return model
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Model loaded on {self.device}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def transform_video(self, input_path, intensity=0.8, exposure=0.5, color=0.7):
        """Transform video using cinema model"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Create output path
        output_path = input_path.replace('.mp4', '_roadshow.mp4')
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame, intensity, exposure, color)
                
                # Write frame
                out.write(processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing: {progress:.1f}%")
        
        finally:
            cap.release()
            out.release()
        
        print(f"✅ Video processed: {frame_count} frames")
        return output_path
    
    def process_frame(self, frame, intensity, exposure, color):
        """Process a single frame through the model"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0,1]
            frame_normalized = rgb_frame.astype(np.float32) / 255.0
            
            # Resize to model input size (768x768)
            original_size = frame_normalized.shape[:2]
            frame_resized = cv2.resize(frame_normalized, (768, 768))
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Apply model transformation
            with torch.no_grad():
                transformed_tensor = self.model(frame_tensor)
            
            # Convert back to numpy
            transformed_frame = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Resize back to original size
            transformed_frame = cv2.resize(transformed_frame, (original_size[1], original_size[0]))
            
            # Apply intensity blending
            blended_frame = self.blend_frames(frame_normalized, transformed_frame, intensity)
            
            # Apply additional effects
            final_frame = self.apply_effects(blended_frame, exposure, color)
            
            # Convert back to BGR and [0,255]
            final_frame = np.clip(final_frame * 255, 0, 255).astype(np.uint8)
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            
            return final_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame  # Return original frame on error
    
    def blend_frames(self, original, transformed, intensity):
        """Blend original and transformed frames"""
        return original * (1 - intensity) + transformed * intensity
    
    def apply_effects(self, frame, exposure, color):
        """Apply additional cinematic effects"""
        # Exposure adjustment
        if exposure != 0.5:
            exposure_factor = 0.8 + (exposure * 0.4)  # Range: 0.8 to 1.2
            frame = np.clip(frame * exposure_factor, 0, 1)
        
        # Color grading
        if color != 0.5:
            # Simple color temperature adjustment
            color_factor = 0.9 + (color * 0.2)  # Range: 0.9 to 1.1
            frame[:, :, 0] *= color_factor  # Adjust red channel
            frame[:, :, 2] *= (2 - color_factor)  # Inverse adjust blue channel
        
        return np.clip(frame, 0, 1)'''
    
    with open("roadshow-film/backend/utils/video_processor.py", "w") as f:
        f.write(video_processor)
    
    # Copy the cinema model (you'll need to do this manually)
    print("📝 Note: You'll need to copy your cinema_v1_4m.py file to backend/model/")
    
    print("✅ Backend files created")

def create_deployment_files():
    """Create deployment configuration files"""
    print("🚀 Creating deployment files...")
    
    # Dockerfile for Railway
    dockerfile = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]'''
    
    with open("roadshow-film/backend/Dockerfile", "w") as f:
        f.write(dockerfile)
    
    # Railway config
    railway_config = '''[build]
builder = "DOCKERFILE"
dockerfilePath = "backend/Dockerfile"

[deploy]
startCommand = "python app.py"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[env]
PORT = { default = "8000" }'''
    
    with open("roadshow-film/railway.toml", "w") as f:
        f.write(railway_config)
    
    # Vercel config
    vercel_config = {
        "version": 2,
        "builds": [
            {
                "src": "frontend/package.json",
                "use": "@vercel/next"
            }
        ],
        "routes": [
            {
                "src": "/(.*)",
                "dest": "frontend/$1"
            }
        ],
        "env": {
            "RAILWAY_STATIC_URL": "@railway-url"
        }
    }
    
    with open("roadshow-film/vercel.json", "w") as f:
        json.dump(vercel_config, f, indent=2)
    
    # GitHub Actions workflow
    github_workflow = '''name: Deploy to Railway

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install Railway CLI
      run: npm install -g @railway/cli
    
    - name: Deploy to Railway
      run: railway up --service roadshow-backend
      env:
        RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}'''
    
    with open("roadshow-film/.github/workflows/railway-deploy.yml", "w") as f:
        f.write(github_workflow)
    
    print("✅ Deployment files created")

def create_documentation():
    """Create README and documentation"""
    print("📚 Creating documentation...")
    
    readme = '''# Roadshow Film

Transform iPhone footage to cinema camera characteristics using machine learning.

## 🎬 Overview

Roadshow Film uses advanced machine learning to transform mobile phone footage into cinema-quality video that matches professional camera characteristics like ARRI, Blackmagic, and Sony cameras.

## 🚀 Quick Start

### Frontend (Vercel)
```bash
cd frontend
npm install
npm run dev
```

### Backend (Railway)
```bash
cd backend
pip install -r requirements.txt
python app.py
```

## 📦 Deployment

### Frontend on Vercel
1. Connect your GitHub repository to Vercel
2. Set build directory to `frontend`
3. Deploy automatically on push to main

### Backend on Railway
1. Connect your GitHub repository to Railway
2. Set root directory to `backend`
3. Railway will automatically detect the Dockerfile

## 🛠️ Tech Stack

- **Frontend**: Next.js, React, Tailwind CSS, Framer Motion
- **Backend**: FastAPI, PyTorch, OpenCV
- **ML Model**: Custom cinema transformation neural network
- **Deployment**: Vercel (Frontend) + Railway (Backend)

## 📁 Project Structure

```
roadshow-film/
├── frontend/          # Next.js frontend
├── backend/           # FastAPI backend
├── .github/           # GitHub Actions
└── deployment files
```

## 🎯 Features

- **Drag & Drop Upload**: Easy video file handling
- **Real-time Processing**: Fast ML inference
- **Cinema Controls**: Adjust intensity, exposure, color
- **Preview Comparison**: Before/after video preview
- **Professional Output**: High-quality enhanced video

## 🧪 Model Information

The core ML model is based on:
- **Architecture**: Residual color transformation network
- **Training**: 79 iPhone/Sony camera pairs
- **Input**: 768x768 RGB frames
- **Output**: Professional color-graded frames

## 🔧 Environment Variables

### Frontend (.env.local)
```
RAILWAY_STATIC_URL=your-railway-backend-url
```

### Backend (.env)
```
PORT=8000
MODEL_PATH=models/cinema_v1_4m_model.pth
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Contact

For questions and support, please open an issue on GitHub.
'''
    
    with open("roadshow-film/README.md", "w") as f:
        f.write(readme)
    
    # .gitignore
    gitignore = '''# Dependencies
node_modules/
*.egg-info/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
pip-log.txt
pip-delete-this-directory.txt

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Build outputs
.next/
out/
dist/
build/

# Logs
*.log
logs/

# Model files (too large for git)
*.pth
*.pt
*.onnx
models/
!models/.gitkeep

# Data files
data/
temp/
uploads/
*.mp4
*.mov
*.avi
*.dng
*.arw

# Testing
.coverage
.pytest_cache/
.tox/

# Railway
.railway/

# Vercel
.vercel/'''
    
    with open("roadshow-film/.gitignore", "w") as f:
        f.write(gitignore)
    
    # Create placeholder files
    Path("roadshow-film/backend/models/.gitkeep").touch()
    
    print("✅ Documentation created")

def copy_model_files():
    """Copy existing model files"""
    print("📋 Copying model files...")
    
    # Files to copy from current directory
    files_to_copy = [
        "cinema_v1_4m.py",
        "cinema_v14_4k.py", 
        "models/cinema_v1_4m_model.pth"
    ]
    
    copy_instructions = []
    
    for file_path in files_to_copy:
        source = Path(file_path)
        if source.exists():
            if "models/" in file_path:
                dest = Path(f"roadshow-film/backend/{file_path}")
            else:
                dest = Path(f"roadshow-film/backend/model/{file_path}")
            
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print(f"   ✅ Copied {file_path}")
        else:
            copy_instructions.append(f"   ❌ Manual copy needed: {file_path} → roadshow-film/backend/model/")
    
    if copy_instructions:
        print("📝 Manual copy instructions:")
        for instruction in copy_instructions:
            print(instruction)
    
    print("✅ Model files processed")

def create_deployment_scripts():
    """Create deployment helper scripts"""
    print("🔧 Creating deployment scripts...")
    
    # Deploy script
    deploy_script = '''#!/bin/bash
set -e

echo "🚀 Roadshow Film Deployment Script"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "roadshow-film/package.json" ] && [ ! -f "package.json" ]; then
    echo "❌ Run this script from the roadshow-film directory or its parent"
    exit 1
fi

# Navigate to project root
if [ -d "roadshow-film" ]; then
    cd roadshow-film
fi

echo "📦 Installing frontend dependencies..."
cd frontend
npm install

echo "🔨 Building frontend..."
npm run build

echo "🐍 Installing backend dependencies..."
cd ../backend
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Push to GitHub: git add . && git commit -m 'Initial commit' && git push origin main"
echo "2. Deploy frontend to Vercel: https://vercel.com/new"
echo "3. Deploy backend to Railway: https://railway.app/new"
echo ""
echo "📖 See README.md for detailed deployment instructions"'''
    
    with open("roadshow-film/deploy.sh", "w") as f:
        f.write(deploy_script)
    
    # Make executable
    os.chmod("roadshow-film/deploy.sh", 0o755)
    
    # Windows batch script
    deploy_bat = '''@echo off
echo 🚀 Roadshow Film Deployment Script
echo ==================================

if not exist "package.json" if not exist "roadshow-film\\package.json" (
    echo ❌ Run this script from the roadshow-film directory or its parent
    exit /b 1
)

if exist "roadshow-film" cd roadshow-film

echo 📦 Installing frontend dependencies...
cd frontend
call npm install

echo 🔨 Building frontend...
call npm run build

echo 🐍 Installing backend dependencies...
cd ..\\backend
pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo 🚀 Next steps:
echo 1. Push to GitHub: git add . ^&^& git commit -m "Initial commit" ^&^& git push origin main
echo 2. Deploy frontend to Vercel: https://vercel.com/new
echo 3. Deploy backend to Railway: https://railway.app/new
echo.
echo 📖 See README.md for detailed deployment instructions'''
    
    with open("roadshow-film/deploy.bat", "w") as f:
        f.write(deploy_bat)
    
    print("✅ Deployment scripts created")

def print_deployment_instructions():
    """Print final deployment instructions"""
    print("\n" + "="*60)
    print("🎉 ROADSHOW FILM PROJECT SETUP COMPLETE!")
    print("="*60)
    
    print(f"""
📁 Project created at: ./roadshow-film/

🚀 DEPLOYMENT STEPS:

1️⃣ GITHUB SETUP:
   cd roadshow-film
   git init
   git add .
   git commit -m "Initial Roadshow Film setup"
   git remote add origin https://github.com/AmmarAnnex/roadshow-cinema-model.git
   git push -u origin main

2️⃣ RAILWAY BACKEND DEPLOYMENT:
   • Go to https://railway.app/new
   • Connect your GitHub repository
   • Select "Deploy from GitHub repo"
   • Choose roadshow-cinema-model repository
   • Set root directory to: "backend"
   • Railway will auto-detect Dockerfile and deploy
   • Note the generated Railway URL (e.g., https://xxx.railway.app)

3️⃣ VERCEL FRONTEND DEPLOYMENT:
   • Go to https://vercel.com/new
   • Connect your GitHub repository
   • Select roadshow-cinema-model repository
   • Set root directory to: "frontend"
   • Add environment variable: RAILWAY_STATIC_URL = your-railway-url
   • Deploy automatically

4️⃣ FINAL SETUP:
   • Copy your trained model file to: roadshow-film/backend/models/
   • Update Railway with the model file
   • Test the deployment

📋 MANUAL TASKS NEEDED:
   ✅ Copy cinema_v1_4m.py to roadshow-film/backend/model/
   ✅ Copy your trained .pth model file to roadshow-film/backend/models/
   ✅ Update GitHub repository
   ✅ Deploy to Railway and Vercel

🔧 QUICK START:
   cd roadshow-film
   ./deploy.sh   (or deploy.bat on Windows)

📖 Full instructions in roadshow-film/README.md
""")

def main():
    """Main setup function"""
    print("🎬 ROADSHOW FILM - COMPLETE DEPLOYMENT SETUP")
    print("=" * 50)
    print("Creating production-ready deployment structure...\n")
    
    try:
        create_directory_structure()
        create_frontend_files()
        create_backend_files()
        create_deployment_files()
        create_documentation()
        copy_model_files()
        create_deployment_scripts()
        print_deployment_instructions()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()