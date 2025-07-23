import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
import faiss
import os

# --- Configuration ---
# WARNING: Directly embedding the API key is NOT recommended for production.
# Use Streamlit secrets or environment variables for secure deployment.
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"  # Your API key directly here

if not API_KEY:
    st.error("Google API Key is missing. Please ensure it's set correctly.")
    st.stop()

genai.configure(api_key=API_KEY)
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini model: {e}")
    st.stop()

# --- Helper Functions for PDF Processing and RAG ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=100):
    """Splits text into smaller, overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts):
    """Generates embeddings for a list of texts using Gemini's embedding model."""
    embeddings = []
    for text in texts:
        try:
            response = genai.embed_content(model="models/embedding-001", content=text)
            embeddings.append(response['embedding'])
        except Exception as e:
            st.warning(f"Error generating embedding for text chunk: {text[:50]}... Error: {e}")
            embeddings.append(None)  # Append None for failed embeddings
    return embeddings

def build_faiss_index(chunks, embeddings):
    """Builds a FAISS index from embeddings."""
    valid_embeddings = [e for e in embeddings if e is not None]
    valid_chunks = [chunks[i] for i, e in enumerate(embeddings) if e is not None]

    if not valid_embeddings:
        st.error("No valid embeddings generated. Cannot build FAISS index.")
        return None, []

    embedding_dim = len(valid_embeddings[0])
    embeddings_np = np.array(valid_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    return index, valid_chunks

def retrieve_relevant_chunks(query, index, chunks, k=3):
    """Retrieves top-k most relevant chunks from the FAISS index."""
    if index is None or not chunks:
        st.warning("Knowledge base is empty. Please upload and process a PDF.")
        return []

    query_embedding = get_embeddings([query])
    if not query_embedding or query_embedding[0] is None:
        st.error("Failed to generate embedding for the query.")
        return []

    query_embedding_np = np.array(query_embedding[0]).astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding_np, k)
    relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    return relevant_chunks

def generate_response_with_rag(query, relevant_chunks):
    """Generates a response using Gemini, incorporating retrieved chunks as context."""
    if not relevant_chunks:
        return "No relevant information found in the knowledge base."

    context = "\n".join(relevant_chunks)
    prompt = f"""You are a helpful banking chatbot. Use the following context to answer the user's question concisely and accurately. If the answer is not in the context, state that you don't have enough information.

    Context:
    {context}

    User Question: {query}

    Answer:
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "I apologize, but I couldn't generate a response at this moment."

# --- Streamlit UI ---

st.set_page_config(
    page_title="Banking Chatbot RAG",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "A RAG-powered banking chatbot using Google Gemini and Streamlit."
    }
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stFileUploader label {
        font-size: 1.1rem;
        font-weight: bold;
        color: #333;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    h3 {
        color: #34495e;
        margin-top: 1.5rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-top: 15px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Banking Chatbot (RAG)")
st.markdown("<h3 style='text-align: center; color: #555;'>Upload banking documents and get instant answers!</h3>", unsafe_allow_html=True)

# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = False

# File Uploader Section
with st.container():
    st.subheader("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload your Banking Q&A PDF",
        type="pdf",
        help="Upload a PDF containing banking questions and answers to build the chatbot's knowledge base."
    )
    
    if uploaded_file:
        # Reset session state when a new file is uploaded
        st.session_state.faiss_index = None
        st.session_state.chunks = []
        st.session_state.knowledge_base_ready = False

        if st.button("Process PDF"):
            with st.spinner("Extracting text and building knowledge base..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text.strip():
                    st.warning("No text could be extracted from the PDF. Ensure it's a searchable PDF.")
                else:
                    st.success("Text extracted from PDF.")
                    chunks = chunk_text(pdf_text)
                    st.info(f"Split PDF into {len(chunks)} chunks.")
                    
                    with st.spinner("Generating embeddings..."):
                        embeddings = get_embeddings(chunks)
                    if embeddings and any(embeddings):
                        faiss_index, processed_chunks = build_faiss_index(chunks, embeddings)
                        st.session_state.faiss_index = faiss_index
                        st.session_state.chunks = processed_chunks
                        st.session_state.knowledge_base_ready = True
                        st.success("Knowledge base built successfully! You can now ask questions.")
                    else:
                        st.error("Failed to generate embeddings. Knowledge base could not be built.")

# Chatbot Interaction Section
with st.container():
    st.subheader("2. Ask Your Question")
    if st.session_state.knowledge_base_ready:
        user_query = st.text_input(
            "Type your question about banking here:",
            key="user_query_input",
            placeholder="e.g., What are the requirements for a personal loan?"
        )

        if user_query:
            with st.spinner("Searching for answers..."):
                relevant_chunks = retrieve_relevant_chunks(user_query, st.session_state.faiss_index, st.session_state.chunks)
                
                if relevant_chunks:
                    response = generate_response_with_rag(user_query, relevant_chunks)
                    st.markdown("### 💬 Chatbot Response:")
                    st.info(response)
                    
                    with st.expander("🔍 See Retrieved Context"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"**Chunk {i+1}:**")
                            st.code(chunk, language="text")
                else:
                    st.warning("Could not find relevant information. Try rephrasing your question or uploading a different PDF.")
    else:
        st.info("Upload and process a PDF in step 1 to enable the chatbot.")

st.markdown("---")
st.caption("Powered by Google Gemini 1.5 Flash and Streamlit.")
