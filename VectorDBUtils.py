from astrapy.db import AstraDB
import os
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

# Load env vars
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_REGION = os.getenv("ASTRA_DB_REGION")

# Astra connection
db = AstraDB(
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=f"https://{ASTRA_DB_ID}-{ASTRA_DB_REGION}.apps.astra.datastax.com"
)
collection = db.collection("llama_documents")

# Helper: hash content to generate doc_id
def generate_doc_id(content_bytes):
    return hashlib.sha256(content_bytes).hexdigest()

# Store document
def store_document_entry(doc_id, doc_name, summary, question=None, answer=None):
    combined_text = f"{summary} {question or ''} {answer or ''}"

    payload = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "summary": summary,
        "question": question,
        "answer": answer,
        "text": combined_text,
        "type": "qa" if question else "summary"
    }

    # Prevent duplicates
    query = {"doc_id": doc_id, "question": question} if question else {"doc_id": doc_id, "type": "summary"}
    existing = collection.find(query)

    if isinstance(existing, list) and existing:
        print("✅ Document already exists in DB.")
        return

    collection.insert_one(payload)
    print("✅ Inserted to DB.")

# Get summary
def get_summary(doc_id):
    results = collection.find({"doc_id": doc_id, "type": "summary"})
    if isinstance(results, list) and results:
        return results[0].get("summary", "")
    return ""

# Get answer
def get_answer(doc_id, question):
    results = collection.find({"doc_id": doc_id, "question": question})
    if isinstance(results, list) and results:
        return results[0].get("answer", "")
    return ""
