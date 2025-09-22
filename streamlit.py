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

# --- PDF yükleme ---
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

# --- Chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Kullanıcı input ---
prompt = st.chat_input("Ask about your book")
if prompt:
    st.session_state.chat_history.append({
        "question": prompt,
        "answer": None,
        "pages_used": []
    })

# --- Chat alanı (container ile dinamik alan) ---
chat_area = st.container()

with chat_area:
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, entry in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"**You:** {entry['question']}")
            if entry["answer"]:
                with st.chat_message("assistant"):
                    st.markdown(f"**Answer:** {entry['answer']}")
                    st.markdown("**Sources:**")
                    for idx in entry["pages_used"]:
                        if 0 <= idx-1 < len(pdf_text_pages):
                            section_text = pdf_text_pages[idx-1].strip()
                            snippet = section_text[:300] + ("..." if len(section_text) > 300 else "")
                            with st.expander(f"Page {idx}"):
                                st.write(snippet)

# --- Cevap üretme (rerun yok, sadece güncelleme) ---
for i, entry in enumerate(st.session_state.chat_history):
    if entry["answer"] is None:
        with st.spinner("Thinking..."):
            client = OpenAI(api_key=user_api_key)
            relevant_pages_idx = find_relevant_pages(entry["question"], vectorizer, embeddings, top_k=2)
            context_text = "\n\n".join([f"Page {j+1}:\n{pdf_text_pages[j]}" for j in relevant_pages_idx])

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use ONLY the information from the provided textbook pages to answer. Do NOT add external knowledge. Mention which page(s) the answer came from."}
            ]
            for prev in st.session_state.chat_history[:i]:
                messages.append({"role": "user", "content": prev["question"]})
                messages.append({"role": "assistant", "content": prev["answer"]})

            messages.append({"role": "user", "content": f"Textbook content:\n{context_text}\n\nQuestion: {entry['question']}\nAnswer:"})

            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                temperature=0.2,
                max_tokens=500
            )

            answer_text = response.choices[0].message.content.strip()
            pages_used = [j+1 for j in relevant_pages_idx]

            # Cevabı güncelle
            st.session_state.chat_history[i]["answer"] = answer_text
            st.session_state.chat_history[i]["pages_used"] = pages_used

            # Chat alanını tekrar çiz (sayfa komple yenilenmeden)
            with chat_area:
                with st.chat_message("assistant"):
                    st.markdown(f"**Answer:** {answer_text}")
                    st.markdown("**Sources:**")
                    for idx in pages_used:
                        if 0 <= idx-1 < len(pdf_text_pages):
                            section_text = pdf_text_pages[idx-1].strip()
                            snippet = section_text[:300] + ("..." if len(section_text) > 300 else "")
                            with st.expander(f"Page {idx}"):
                                st.write(snippet)
        break
