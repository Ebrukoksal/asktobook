import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class TextbookTutor:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # HuggingFace embedding
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # ChromaDB setup
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="textbook")
        
    def load_textbook(self, pages):
        """Load textbook pages into ChromaDB with embeddings."""
        self.pages = pages
        for i, page in enumerate(pages):
            emb = self.embedder.encode([page])[0].tolist()
            self.collection.add(
                ids=[str(i)],
                embeddings=[emb],
                documents=[page],
                metadatas=[{"page": i+1}]
            )
        print(f"✅ Loaded {len(pages)} pages into ChromaDB")
        
    def _find_relevant_chunks(self, query, top_k=2):
        """Retrieve top-k relevant chunks from ChromaDB."""
        query_emb = self.embedder.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        return results
    
    def answer(self, query: str, top_k: int = 2):
        """Generate answer using retrieved chunks."""
        results = self._find_relevant_chunks(query, top_k=top_k)
        if not results["documents"]:
            return {"answer": "⚠️ No relevant information found.", "pages": []}
        
        docs = results["documents"][0]
        pages = [meta["page"] for meta in results["metadatas"][0]]
        context_text = "\n\n".join([f"Page {p}:\n{d}" for p, d in zip(pages, docs)])
        
        system_prompt = f"""
You are an educational assistant.
Use ONLY the following textbook content to answer the question.
Do NOT add external knowledge.

Textbook content:
{context_text}

Instructions:
1. Answer based only on the given pages.
2. Mention which page(s) the answer came from.
3. Keep the answer concise and accurate.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.2,
                max_tokens=500
            )
            answer = response.choices[0].message.content.strip()
            return {"answer": answer, "pages": pages}
        except Exception as e:
            return {"answer": f"⚠️ Error: {str(e)}", "pages": []}
