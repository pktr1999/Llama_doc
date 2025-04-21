import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from llama_cpp import Llama
import textwrap
import io
import os
import multiprocessing

# Optional: If tesseract is not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Extract text from PDF ===
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# === Extract text from image using OCR ===
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# === Chunk long text ===
def chunk_text(text, max_chars=1500):
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return textwrap.wrap(text, max_chars, break_long_words=False)

# === Ask LLaMA on one chunk ===
def ask_llama_question(chunk, question, llm):
    prompt = f"""You are an intelligent assistant. Use the below text to answer the question.

Text:
{chunk}

Question: {question}
Answer:"""
    response = llm(prompt, max_tokens=300, stop=["Question:", "\n"])
    return response["choices"][0]["text"].strip()

# === Ask LLaMA across chunks (with early stop) ===
def ask_question_over_chunks(text, question, llm):
    chunks = chunk_text(text)
    answers = []
    for chunk in chunks:
        answer = ask_llama_question(chunk, question, llm)
        answers.append(answer)

        # Early stop if a confident answer is found
        if len(answer) > 100 and "not mentioned" not in answer.lower():
            break

    return "\n---\n".join(answers)

# === Load local LLaMA model ===
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = r"C:\Users\Prasanna\.cache\huggingface\hub\models--TheBloke--Llama-2-7b-Chat-GGUF\blobs\08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa"
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=multiprocessing.cpu_count(),
        n_batch=512
    )

llm = load_model()

# === Streamlit UI ===
#st.set_page_config(page_title="LLaMA IDP", layout="wide")
st.title("📄🧠 Intelligent Document Processing using LLaMA")
st.markdown("Supports PDFs and Images (OCR)")

uploaded_file = st.file_uploader("📤 Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])
question = st.text_input("❓ Ask a question based on the document content:")

if uploaded_file and question:
    file_type = uploaded_file.type

    with st.spinner("🔍 Extracting text..."):
        if "pdf" in file_type:
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif "image" in file_type:
            extracted_text = extract_text_from_image(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    if not extracted_text.strip():
        st.warning("⚠️ No text could be extracted from the document.")
    else:
        with st.spinner("🧠 Thinking with LLaMA..."):
            final_answer = ask_question_over_chunks(extracted_text, question, llm)
            st.subheader("📌 Answer")
            st.write(final_answer)
