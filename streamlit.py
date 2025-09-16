import streamlit as st
import PyPDF2
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# from dotenv import load_dotenv  # REMOVE this line

# load_dotenv()  # REMOVE this line
# api_key = os.getenv("OPENAI_API_KEY")  # REMOVE this line

st.title("Ask Questions About Your PDF Book")

# NEW: Let user input their OpenAI API key
user_api_key = st.text_input("Enter your OpenAI API key:", type="password")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")
pdf_text_pages = []  

def extract_pdf_pages(file):
    reader = PyPDF2.PdfReader(file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages

def build_index(pages):
    vectorizer = TfidfVectorizer(stop_words="english")
    embeddings = vectorizer.fit_transform(pages)
    return vectorizer, embeddings

def find_relevant_pages(query, vectorizer, embeddings, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = np.dot(query_vec, embeddings.T).toarray()[0]
    top_pages = np.argsort(scores)[::-1][:top_k]
    return top_pages

if uploaded_file is not None:
    pdf_text_pages = extract_pdf_pages(uploaded_file)
    st.success(f"PDF uploaded! {len(pdf_text_pages)} pages extracted.")

    if pdf_text_pages:
        vectorizer, embeddings = build_index(pdf_text_pages)

        question = st.text_input("Ask a question about the book:")

        # Only proceed if user provided an API key
        if question and user_api_key:
            with st.spinner("Thinking..."):
                # Initialize OpenAI client with user key
                client = OpenAI(api_key=user_api_key)

                relevant_pages_idx = find_relevant_pages(question, vectorizer, embeddings, top_k=2)
                context_text = "\n\n".join([f"Page {i+1}:\n{pdf_text_pages[i]}" for i in relevant_pages_idx])

                system_prompt = f"""
You are an educational assistant. Use ONLY the information from the following textbook pages to answer the question.
Do NOT add external knowledge. Mention which page(s) the answer came from.

Textbook content:
{context_text}

Question: {question}
Answer:
"""

                response = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                ) 

                answer_text = response.choices[0].message.content.strip()
                pages_used = [i+1 for i in relevant_pages_idx]

                st.markdown(f"**Answer:** {answer_text}")
                st.markdown(f"**Source Pages:** {pages_used}")
        elif question and not user_api_key:
            st.warning("Please enter your OpenAI API key to ask questions.")
else:
    st.info("Please upload a PDF to get started.")
