# EL Image Efficiency Agent
A powerful Streamlit application for analyzing solar cell Electroluminescence (EL) images using machine learning and large language models. Classify efficiency levels, predict Power Conversion Efficiency (PCE), and get AI-powered insights about your solar cell quality.

# Features
• 🖼️ EL Image Analysis: Upload multiple solar cell EL images for batch processing
• 📊 Efficiency Classification: Automatically classify images as High/Low efficiency using ResNet features
• 🔬 PCE Prediction: Predict Power Conversion Efficiency for high-efficiency cells
• 🤖 AI-Powered Insights: Get detailed technical analysis using LLM (Local or SJTU API)
• 💬 Interactive Chat: Ask questions about your specific analysis results
• 📈 Batch Reporting: Generate comprehensive batch quality assessments
• 📥 Data Export: Download results as CSV for further analysis

# Quick Start
## Prerequisites
• Python 3.12
• 4GB+ RAM
• 2GB+ free disk space

# Installation
## 1. Clone the repository
git clone https://github.com/mahreenahmed/gcl.git
cd el-agent-app

## 2. Create virtual environment (recommended)
python -m venv el_env
source el_env/bin/activate

On Windows: 
el_env\Scripts\activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Download model files
Place the following files in the models/ directory:
■ best_classifier.onnx
■ weighted_random_forest_model.pkl
■ feature_scaler.pkl

## 5. Running the Application
streamlit run el_agent_app.py

The app will open in your browser at http://localhost:8501

# Usage Guide
## Step 1: Upload EL Images
• Click "Upload EL images" button
• Select multiple PNG/JPG/JPEG files
• Supported: Standard EL images of solar cells

## Step 2: Run Analysis
• Click "Classify & Summarize Batch"
• Wait for processing (typically 2-10 seconds per image)
• View efficiency classifications and PCE predictions

## Step 3: Get AI Insights
• Use pre-built analysis buttons:
◦ Defect Analysis: Identify issues in low-efficiency cells
◦ PCE Analysis: Evaluate performance metrics
◦ Batch Summary: Comprehensive quality report
◦ Next Steps: Practical recommendations

## Step 4: Ask Custom Questions
• Type specific questions in the chat interface
• Examples:
◦ "Which images show cracking patterns?"
◦ "Compare high vs low efficiency cells"
◦ "What could cause the dark areas in image3.jpg?"

# Understanding Results
## Efficiency Classification
• 🔆 High Efficiency: Uniform, bright luminescence (good quality)
• ⚠️ Low Efficiency: Dark areas, cracks, non-uniform patterns (defects)
## PCE Values
• 10-18%: Good performance range
• <10%: Poor performance
• "—": Not calculated for low-efficiency cells

# Quality Indicators
• High batch efficiency ratio: Good manufacturing consistency
• Tight PCE distribution: Excellent process control
• Low defect rate: Robust cell quality

# Security & Privacy
• Encrypted API Keys: SJTU API keys are encrypted for security
• No Data Storage: Uploaded images are processed in memory only
• Session-based: Chat history clears when browser closes

# Troubleshooting
## Common Issues
"Models not loaded" error
• Ensure model files are in models/ directory
• Check file permissions
• Verify ONNX runtime is installed
"API connection failed"
• Check internet connection
• Verify SJTU campus network access
• Confirm API key is correctly encrypted
"Image processing error"
• Ensure images are valid EL images
• Check image format (PNG/JPG/JPEG)
• Verify file size (<10MB recommended)
Slow performance
• Close other memory-intensive applications
• Reduce number of simultaneous uploads
• Use local model mode for faster responses
Performance Tips
• Batch size: Process 5-10 images at a time for optimal performance
• Image size: Resize large images (>5MB) before uploading
• Local mode: Use for faster results without advanced analysis

# Development
## Project Structure:
el-agent-app/
├── el_agent_app.py          # Main application           
├── requirements.txt         # Python dependencies
├── models/                  # Machine learning models
│   ├── best_classifier.onnx
│   ├── weighted_random_forest_model.pkl
│   └── feature_scaler.pkl
├── predictor.py            # ONNX classifier wrapper
└── README.md

# Citation
If you use this tool in your research, please cite:
bibtex
@software{el_agent_2025,
  title = {EL Image Efficiency Agent},
  author = {Dr. Mehreen Ahmed},
  year = {2025},
  url = {https://github.com/mahreenahmed/gcl/}
}
