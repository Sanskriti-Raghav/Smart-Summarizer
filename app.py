import io
import os
import streamlit as st

from dotenv import load_dotenv
from pdf_utils import extract_text_from_pdf
from summarizer import summarize_document

DEFAULT_MODEL = "gemini-1.5-flash"  # Free-tier friendly

st.set_page_config(page_title="Smart Summarizer", page_icon="🧠", layout="wide")

def read_txt_file(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fallbacks
        for enc in ("utf-16", "latin-1"):
            try:
                return file_bytes.decode(enc)
            except Exception:
                continue
    return ""

def main():
    st.title("🧠 Smart Summarizer")
    st.caption("Text Summarization Web App powered by Google Gemini")

    with st.sidebar:
        st.header("Controls")
        input_mode = st.radio("Input Method", ["Paste text", "Upload file"], index=0)
        target_words = st.number_input(
            "Target summary length (words)", min_value=50, max_value=2000, value=200, step=50
        )
        takeaways_count = st.selectbox(
            "Number of key takeaways", options=list(range(1, 11)), index=2
        )

        model = DEFAULT_MODEL

    raw_text = ""
    uploaded_filename = None

    if input_mode == "Paste text":
        raw_text = st.text_area("Paste your text below", height=300, placeholder="Paste text here...")
    else:
        up = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
        if up is not None:
            uploaded_filename = up.name
            file_bytes = up.read()
            if up.type == "text/plain" or uploaded_filename.lower().endswith(".txt"):
                raw_text = read_txt_file(file_bytes)
            else:
                # PDF
                with st.spinner("Extracting text from PDF..."):
                    raw_text = extract_text_from_pdf(io.BytesIO(file_bytes))

    if st.button("Summarize", type="primary"):
        if not raw_text or not raw_text.strip():
            st.warning("Please provide some text (paste or upload a valid .txt/.pdf).")
            return

        if len(raw_text.split()) < 20:
            st.warning("Input is very short. You may still proceed, but results might be trivial.")

        with st.spinner("Generating summary..."):
            # Ensure .env is loaded for GEMINI_API_KEY
            load_dotenv()
            try:
                result = summarize_document(
                    raw_text=raw_text,
                    model=model,
                    target_words=int(target_words),
                    takeaways_count=int(takeaways_count),
                )
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                return

        if not result or "summary" not in result or "key_takeaways" not in result:
            st.error("Unexpected response format from LLM. Please try again.")
            return

        st.subheader("Summary")
        st.write(result["summary"].strip())

        st.subheader("Key Takeaways")
        if isinstance(result["key_takeaways"], list) and result["key_takeaways"]:
            for i, point in enumerate(result["key_takeaways"], start=1):
                st.markdown(f"- {point}")
        else:
            st.info("No key takeaways returned.")

        # Download Summary
        file_base = (uploaded_filename or "summary").rsplit(".", 1)[0]
        download_name = f"{file_base}_summary.txt"
        st.download_button(
            label="⬇️ Download Summary (.txt)",
            data=result["summary"].strip().encode("utf-8"),
            file_name=download_name,
            mime="text/plain",
        )

    st.markdown("---")
    st.caption("Prototype • Handles large documents via chunking and map-reduce with Gemini")

if __name__ == "__main__":
    main()
