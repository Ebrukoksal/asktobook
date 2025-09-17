import streamlit as st
import PyPDF2
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.title("Ask Questions About Your Book")

# 1. API Key
user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
if not user_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# 2. Book selection
book_options = ["Book 1: Interprocess Communications in LINUX", "Book 2: C# Network Programming", "Upload your own PDF"]
book_choice = st.radio("Choose a book or upload your own:", book_options)

def extract_pdf_pages(file):
    reader = PyPDF2.PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def build_index(pages):
    vectorizer = TfidfVectorizer(stop_words="english")
    embeddings = vectorizer.fit_transform(pages)
    return vectorizer, embeddings

def find_relevant_pages(query, vectorizer, embeddings, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = np.dot(query_vec, embeddings.T).toarray()[0]
    return np.argsort(scores)[::-1][:top_k]


uploaded_file = None
if book_choice == "Upload your own PDF":
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")
    if not uploaded_file:
        st.info("Please upload a PDF to get started.")
        st.stop()
    pdf_text_pages = extract_pdf_pages(uploaded_file)
else:
    if book_choice == "Book 1: Interprocess Communications in LINUX":
        with open("books/computer network programming kitap 1.pdf", "rb") as f:
            pdf_text_pages = extract_pdf_pages(f)
    elif book_choice == "Book 2: C# Network Programming":
        with open("books/computer network programming kitap 2.pdf", "rb") as f:
            pdf_text_pages = extract_pdf_pages(f)

vectorizer, embeddings = build_index(pdf_text_pages)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.chat_input("Ask about your book")
if prompt:
    with st.spinner("Thinking..."):
        client = OpenAI(api_key=user_api_key)
        relevant_pages_idx = find_relevant_pages(prompt, vectorizer, embeddings, top_k=2)
        context_text = "\n\n".join([f"Page {i+1}:\n{pdf_text_pages[i]}" for i in relevant_pages_idx])

        # Build the messages list from chat history
        messages = [{"role": "system", "content": "You are a helpful assistant. Use ONLY the information from the provided textbook pages to answer. Do NOT add external knowledge. Mention which page(s) the answer came from."}]

        for entry in st.session_state.chat_history:
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["answer"]})

        # Add the current user question, with context
        messages.append({"role": "user", "content": f"Textbook content:\n{context_text}\n\nQuestion: {prompt}\nAnswer:"})

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )

        answer_text = response.choices[0].message.content.strip()
        pages_used = [i+1 for i in relevant_pages_idx]

        # Add immediately to chat history
        st.session_state.chat_history.append({
            "question": prompt,
            "answer": answer_text,
            "pages_used": pages_used
        })

# --- Move chat display here so it always shows latest state ---
if st.session_state.chat_history:
    st.subheader("Chat History")
    for entry in st.session_state.chat_history:
        st.chat_message("user").write(f"**You:** {entry['question']}")
        st.chat_message("assistant").write(f"**Answer:** {entry['answer']}")

        # Display each source page/section with its relevant text
        st.markdown("**Sources:**")
        for idx in entry["pages_used"]:
            # idx is 1-based, so subtract 1 for list index
            section_text = pdf_text_pages[idx - 1].strip() if 0 <= idx - 1 < len(pdf_text_pages) else ""
            # Optionally, show only a snippet (e.g., first 300 chars)
            snippet = section_text[:300] + ("..." if len(section_text) > 300 else "")
            with st.expander(f"Section {idx}"):
                st.write(snippet)
