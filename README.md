# Smart Summarizer (Prototype)

Smart Summarizer is a Streamlit web app that generates **concise abstractive summaries** and **key takeaways** from pasted text or uploaded `.txt` / `.pdf` files using **Google Gemini**. It handles **very large inputs (~100 pages)** via **chunking + map-reduce**.

## Features
- Paste text or upload `.txt`/`.pdf`
- Adjustable **target summary length** (50–2000 words)
- Choose **1–10 key takeaways**
- **Download** the summary as a `.txt` file
- Robust PDF extraction with PyPDF2
- Map-Reduce pipeline for long documents
- Graceful handling for very short or punctuation-light inputs

## Tech Stack
- Python 3.10+
- Streamlit (UI)
- PyPDF2 (PDF extraction)
- google-generativeai (Gemini API)
- python-dotenv (env variables)

## Project Structure
smart-summarizer/
├── app.py           # Streamlit app entry point
├── llm_client.py    # Gemini API client
├── summarizer.py    # Chunking + map-reduce summarization
├── pdf_utils.py     # PDF text extraction
├── prompts.py       # Prompt templates
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── .env             # Environment Variables


## Setup & Installation

1. **Clone / Create Project Folder**
```
mkdir smart-summarizer && cd smart-summarizer
```

2. **Create and activate a virtual environment (recommended)**
```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**
```
pip install -r requirements.txt
```

4. **Create .env file**
```
GEMINI_API_KEY=your_key_here
```
5. **Run the app**
```
streamlit run app.py
```

