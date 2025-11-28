# 🔍 EL Image Efficiency Agent

A powerful Streamlit application for analyzing solar cell Electroluminescence (EL) images using machine learning and large language models. Classify efficiency levels, predict Power Conversion Efficiency (PCE), and get AI-powered insights about your solar cell quality.

![EL Analysis](https://img.shields.io/badge/EL-Analysis-blue)
![Solar Cells](https://img.shields.io/badge/Solar-Cells-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)

## ✨ Features

- **🖼️ EL Image Analysis**: Upload multiple solar cell EL images for batch processing
- **📊 Efficiency Classification**: Automatically classify images as High/Low efficiency using ResNet features
- **🔬 PCE Prediction**: Predict Power Conversion Efficiency for high-efficiency cells
- **🤖 AI-Powered Insights**: Get detailed technical analysis using LLM (Local or SJTU API)
- **💬 Interactive Chat**: Ask questions about your specific analysis results
- **📈 Batch Reporting**: Generate comprehensive batch quality assessments
- **📥 Data Export**: Download results as CSV for further analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 4GB+ RAM
- 2GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mahreenahmed/gcl.git
   cd el-agent-app

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv el_env
   source el_env/bin/activate  # On Windows: el_env\Scripts\activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **Download model files**
Place the following files in the models/ directory:
- best_classifier.onnx
- weighted_random_forest_model.pkl
- feature_scaler.pkl

5. **Running the Application**
**streamlit run el_agent_app.py
**The app will open in your browser at http://localhost:8501

