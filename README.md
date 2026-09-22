# DemeologyAI

An AI-powered dermatology explorer that analyzes skin photos and generates a detailed clinical-style report using a vision model and a large language model.

## Features
- Drag-and-drop or click-to-upload image analysis
- AI-generated skin condition report (overview, causes, recommended routine, professional decision)
- Animated skin health score and scanning UI
- Light/dark theme toggle
- Responsive, self-contained frontend (no build step required)

## Tech Stack
- **Backend:** Flask, Flask-CORS
- **AI Vision:** PyTorch, HuggingFace Transformers (ViT-based skin classifier)
- **AI Report Generation:** Groq API (OpenAI-compatible SDK), `openai/gpt-oss-120b`
- **Frontend:** Vanilla HTML, CSS, JavaScript (single self-contained file)
- **Deployment:** Gunicorn, Render

## Project Status
> **Note:** This deployment currently runs in **stub mode** — the real ViT model weights (`model.safetensors`, `vit_skin_disease.pth`) are Git LFS-tracked files not included in this build. Predictions are randomly sampled from the model's 22 class labels for demonstration purposes, while the AI report generation (via Groq) is fully live and functional.

## Local Setup

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   # Windows
   venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
