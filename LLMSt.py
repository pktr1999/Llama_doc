import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from Llama import load_model, ask_question_over_chunks, summarize_document
from VectorDBUtils import (
    generate_doc_id,
    store_document_entry,
    get_summary,
    get_answer
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    content = pdf_file.read()
    if not content:
        raise ValueError("Empty file")
    with fitz.open(stream=content, filetype="pdf") as doc:
        text = "".join([page.get_text() for page in doc])
    return content, text

# Extract text from image
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return image_file.read(), pytesseract.image_to_string(image)

# Cache model
@st.cache_resource
def get_model():
    model_path = r"C:\Users\Prasanna\.cache\huggingface\hub\models--TheBloke--Llama-2-7b-Chat-GGUF\blobs\08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa"
    return load_model(model_path)

# UI
st.title("📄 LLaMA-based Document Summarizer & QA")

uploaded_file = st.file_uploader("📤 Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
question = st.text_input("❓ Ask a question")
summarize = st.checkbox("📝 Generate Summary")

if uploaded_file:
    file_type = uploaded_file.type

    try:
        with st.spinner("📄 Extracting..."):
            if "pdf" in file_type:
                file_bytes, extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                file_bytes, extracted_text = extract_text_from_image(uploaded_file)

        doc_id = generate_doc_id(file_bytes)
        doc_name = uploaded_file.name

        llm = get_model()

        if summarize:
            with st.spinner("📚 Checking summary in DB..."):
                existing_summary = get_summary(doc_id)
                if existing_summary:
                    st.subheader("📝 Stored Summary")
                    st.write(existing_summary)
                else:
                    with st.spinner("🧠 Summarizing..."):
                        summary = summarize_document(extracted_text, llm)
                        store_document_entry(doc_id, doc_name, summary)
                        st.subheader("📝 New Summary")
                        st.write(summary)

        if question:
            with st.spinner("💬 Checking answer in DB..."):
                existing_answer = get_answer(doc_id, question)
                if existing_answer:
                    st.subheader("📌 Stored Answer")
                    st.write(existing_answer)
                else:
                    with st.spinner("🧠 Thinking..."):
                        answer = ask_question_over_chunks(extracted_text, question, llm)
                        summary = get_summary(doc_id) or summarize_document(extracted_text, llm)
                        store_document_entry(doc_id, doc_name, summary, question, answer)
                        st.subheader("📌 New Answer")
                        st.write(answer)

    except Exception as e:
        st.error(f"Error: {str(e)}")
