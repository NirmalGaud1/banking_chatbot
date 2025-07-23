import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
import faiss
import requests
from io import BytesIO
import time
from google.api_core import exceptions
from pdf2image import convert_from_bytes
import pytesseract

API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
DEFAULT_PDF_URL = "https://raw.githubusercontent.com/NirmalGaud1/banking_chatbot/main/Banking_Chatbot_Dataset.pdf"

if not API_KEY:
    st.error("Google API Key is missing.")
    st.stop()

genai.configure(api_key=API_KEY)
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Failed to initialize Gemini model.")
    st.stop()

def fetch_pdf_from_github(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        st.write(f"Debug: Fetched PDF from {url}")
        return BytesIO(response.content)
    except:
        st.error(f"Error fetching PDF from {url}")
        return None

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            st.warning("No text found. Attempting OCR...")
            pdf_file.seek(0)
            images = convert_from_bytes(pdf_file.read())
            text = ""
            for i, image in enumerate(images):
                st.write(f"Debug: OCR page {i+1}")
                text += pytesseract.image_to_string(image) + "\n"
        st.write(f"Debug: Extracted text length: {len(text)}")
        return text
    except:
        st.error("Error extracting text from PDF")
        return ""

def chunk_text(text, chunk_size=300, overlap=30):
    if not text:
        st.warning("No text to chunk")
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    st.write(f"Debug: Created {len(chunks)} chunks")
    return chunks

def get_embeddings(texts, max_retries=3, batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            for attempt in range(max_retries):
                try:
                    response = genai.embed_content(model="models/embedding-001", content=text)
                    embeddings.append(response['embedding'])
                    break
                except exceptions.ResourceExhausted:
                    st.warning(f"Rate limit for chunk {i+1}: {text[:50]}... Retry {attempt+1}/{max_retries}")
                    time.sleep(2 ** attempt)
                except:
                    st.warning(f"Error embedding chunk {i+1}: {text[:50]}...")
                    embeddings.append(None)
                    break
    valid_count = sum(1 for e in embeddings if e is not None)
    st.write(f"Debug: Generated {valid_count}/{len(texts)} embeddings")
    return embeddings

def build_faiss_index(chunks, embeddings):
    valid_embeddings = [e for e in embeddings if e is not None]
    valid_chunks = [chunks[i] for i, e in enumerate(embeddings) if e is not None]
    if not valid_embeddings:
        st.error("No valid embeddings")
        return None, []
    embedding_dim = len(valid_embeddings[0])
    embeddings_np = np.array(valid_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    st.write(f"Debug: FAISS index with {index.ntotal} vectors")
    return index, valid_chunks

def retrieve_relevant_chunks(query, index, chunks, k=3):
    if index is None or not chunks:
        st.warning("Empty knowledge base")
        return []
    query_embedding = get_embeddings([query])
    if not query_embedding or query_embedding[0] is None:
        st.error("Failed query embedding")
        return []
    query_embedding_np = np.array(query_embedding[0]).astype('float32').reshape(1, -1)
    try:
        D, I = index.search(query_embedding_np, min(k, len(chunks)))
        relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        st.write(f"Debug: Retrieved {len(relevant_chunks)} chunks for query")
        return relevant_chunks
    except:
        st.error("Error in FAISS search")
        return []

def generate_response_with_rag(query, relevant_chunks):
    if not relevant_chunks:
        return "No relevant information found. Try rephrasing."
    context = "\n".join([chunk[:300] for chunk in relevant_chunks])
    prompt = f"""Banking chatbot. Answer concisely using context. If no answer in context, say so.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    try:
        response = gemini_model.generate_content(prompt, generation_config={"max_output_tokens": 300})
        st.write("Debug: Response generated")
        return response.text.strip()
    except:
        st.error("Error generating response")
        return "Couldn't generate response"

st.set_page_config(
    page_title="GitHub PDF Banking Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; border-radius: 8px; border: none; cursor: pointer; }
    .stButton>button:hover { background-color: #45a049; }
    .stTextInput>div>div>input { border-radius: 8px; border: 1px solid #ccc; padding: 10px; }
    .stAlert { border-radius: 8px; }
    .stExpander { border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-top: 15px; background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 GitHub PDF Banking Chatbot")
st.markdown("<h3 style='text-align: center; color: #555;'>Query banking info from GitHub PDF</h3>", unsafe_allow_html=True)

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.chunks = []
    st.session_state.knowledge_base_ready = False

with st.container():
    st.subheader("1. Provide GitHub PDF URL")
    github_url = st.text_input(
        "Enter raw GitHub URL",
        value=DEFAULT_PDF_URL
    )
    
    if st.button("Process PDF"):
        st.session_state.faiss_index = None
        st.session_state.chunks = []
        st.session_state.knowledge_base_ready = False
        with st.spinner("Fetching and processing PDF..."):
            pdf_file = fetch_pdf_from_github(github_url)
            if pdf_file:
                pdf_text = extract_text_from_pdf(pdf_file)
                if not pdf_text.strip():
                    st.warning("No text extracted")
                else:
                    st.success(f"Text extracted: {len(pdf_text)} chars")
                    chunks = chunk_text(pdf_text)
                    with st.spinner("Generating embeddings..."):
                        embeddings = get_embeddings(chunks)
                    if embeddings and any(embeddings):
                        faiss_index, processed_chunks = build_faiss_index(chunks, embeddings)
                        st.session_state.faiss_index = faiss_index
                        st.session_state.chunks = processed_chunks
                        st.session_state.knowledge_base_ready = True
                        st.success("Knowledge base ready")
                    else:
                        st.error("Failed embeddings")
                        st.session_state.knowledge_base_ready = False

with st.container():
    st.subheader("2. Ask Your Question")
    if st.session_state.knowledge_base_ready:
        st.write(f"Debug: FAISS index size: {st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0}")
        st.write(f"Debug: Chunks: {len(st.session_state.chunks)}")
        user_query = st.text_input(
            "Type banking question",
            placeholder="e.g., What are interest rates for savings?"
        )
        if user_query:
            with st.spinner("Searching..."):
                relevant_chunks = retrieve_relevant_chunks(user_query, st.session_state.faiss_index, st.session_state.chunks)
                if relevant_chunks:
                    response = generate_response_with_rag(user_query, relevant_chunks)
                    st.markdown("### 💬 Response:")
                    st.info(response)
                    with st.expander("🔍 Context"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"Chunk {i+1}:")
                            st.code(chunk, language="text")
                else:
                    st.warning("No relevant info")
    else:
        st.info("Process PDF to enable chatbot")

st.markdown("---")
st.caption("Powered by Google Gemini 1.5 Flash and Streamlit")
