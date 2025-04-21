import textwrap
from llama_cpp import Llama
import multiprocessing

# === Load LLaMA Model ===
def load_model(model_path):
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=multiprocessing.cpu_count(),
        n_batch=512
    )

# === Chunk text ===
def chunk_text(text, max_chars=1500):
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return textwrap.wrap(text, max_chars, break_long_words=False)

# === Ask question on a single chunk ===
def ask_llama_question(chunk, question, llm):
    prompt = f"""You are an intelligent assistant. Use the below text to answer the question.

Text:
{chunk}

Question: {question}
Answer:"""
    response = llm(prompt, max_tokens=300, stop=["Question:", "\n"])
    return response["choices"][0]["text"].strip()

# === Ask question across all chunks ===
def ask_question_over_chunks(text, question, llm):
    chunks = chunk_text(text)
    answers = []
    for chunk in chunks:
        answer = ask_llama_question(chunk, question, llm)
        answers.append(answer)
        if len(answer) > 100 and "not mentioned" not in answer.lower():
            break
    return "\n---\n".join(answers)

# === Summarize a single chunk ===
def summarize_chunk(chunk, llm):
    prompt = f"""Summarize the following document text in bullet points.

Text:
{chunk}

Summary:"""
    response = llm(prompt, max_tokens=300, stop=["Text:", "\n\n"])
    return response["choices"][0]["text"].strip()

# === Summarize full document ===
def summarize_document(text, llm):
    chunks = chunk_text(text)
    summaries = [summarize_chunk(chunk, llm) for chunk in chunks]
    return "\n\n".join(summaries)