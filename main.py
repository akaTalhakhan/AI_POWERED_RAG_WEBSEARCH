import streamlit as st
import openai
import chromadb
import PyPDF2
import requests
from datetime import datetime
import uuid
import json
import time
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import urllib.parse
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="🤖 AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Custom Styling for Dark/Light Mode
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Dark/Light mode variables */
    :root {
        --primary-color: #6366f1;
        --primary-dark: #4f46e5;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --info-color: #3b82f6;
        --surface-light: #ffffff;
        --surface-dark: #1a1a1a;
        --surface-light-secondary: #f8fafc;
        --surface-dark-secondary: #2d2d2d;
        --text-light: #1e293b;
        --text-dark: #f1f5f9;
        --text-light-secondary: #64748b;
        --text-dark-secondary: #94a3b8;
        --border-light: #e2e8f0;
        --border-dark: #374151;
        --shadow-light: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-dark: 0 1px 3px 0 rgb(0 0 0 / 0.3), 0 1px 2px -1px rgb(0 0 0 / 0.3);
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Header styling */
    .main-header {
        background: var(--gradient-primary);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: var(--shadow-light);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid var(--border-light);
        padding: 0 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 24px;
        background: transparent;
        border-radius: 12px 12px 0 0;
        font-weight: 500;
        font-size: 16px;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--surface-light-secondary);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary-color);
        color: white !important;
        border-bottom: 3px solid var(--primary-dark);
    }
    
    /* Enhanced Card Components */
    .metric-card {
        background: var(--gradient-primary);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px -3px rgb(0 0 0 / 0.1), 0 4px 6px -2px rgb(0 0 0 / 0.05);
    }
    
    .status-card {
        background: var(--surface-light);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -3px rgb(0 0 0 / 0.1);
    }
    
    .source-card {
        background: var(--surface-light);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .source-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }
    
    .source-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -3px rgb(0 0 0 / 0.1);
        border-color: var(--primary-color);
    }
    
    /* Chat Container */
    .chat-container {
        background: var(--surface-light);
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
        min-height: 500px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        box-shadow: var(--shadow-light);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -3px rgb(0 0 0 / 0.2);
    }
    
    .stButton > button[kind="primary"] {
        background: var(--gradient-primary);
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: var(--gradient-primary);
        filter: brightness(1.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--surface-light);
        border-right: 1px solid var(--border-light);
    }
    
    /* Input Field Enhancements */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div > div {
        border-radius: 12px;
        border: 2px solid var(--border-light);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus,
    .stSelectbox > div > div > div:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border: 2px dashed var(--border-light);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-color);
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: var(--gradient-primary);
        border-radius: 8px;
    }
    
    /* Alert Messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: var(--shadow-light);
    }
    
    /* Search Results */
    .search-result {
        background: var(--surface-light);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .search-result:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-light);
        border-color: var(--primary-color);
    }
    
    .search-result::before {
        content: '🔍';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.2rem;
        opacity: 0.5;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        border-radius: 12px;
        background: var(--surface-light-secondary);
        border: 1px solid var(--border-light);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    /* Footer */
    .footer {
        background: var(--surface-light-secondary);
        border-top: 1px solid var(--border-light);
        padding: 2rem;
        margin-top: 3rem;
        border-radius: 16px;
        text-align: center;
        color: var(--text-light-secondary);
    }
    
    /* Dark Mode Overrides */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: var(--surface-dark);
            color: var(--text-dark);
        }
        
        .status-card,
        .source-card,
        .chat-container,
        .search-result {
            background: var(--surface-dark-secondary);
            border-color: var(--border-dark);
            color: var(--text-dark);
        }
        
        .css-1d391kg {
            background: var(--surface-dark-secondary);
            border-color: var(--border-dark);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: var(--surface-dark-secondary);
        }
        
        .streamlit-expanderHeader {
            background: var(--surface-dark-secondary);
            border-color: var(--border-dark);
            color: var(--text-dark);
        }
        
        .footer {
            background: var(--surface-dark-secondary);
            border-color: var(--border-dark);
            color: var(--text-dark-secondary);
        }
    }
    
    /* Custom animations */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .animate-slide-up {
        animation: slideInUp 0.6s ease-out;
    }
    
    .animate-fade-in {
        animation: fadeIn 0.4s ease-out;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-connected {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-disconnected {
        background: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    /* Hover effects for interactive elements */
    .hover-lift {
        transition: all 0.3s ease;
    }
    
    .hover-lift:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -3px rgb(0 0 0 / 0.1);
    }
    
    /* Loading states */
    .loading-shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .metric-card,
        .status-card,
        .source-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "api_initialized" not in st.session_state:
    st.session_state.api_initialized = False
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
if "search_results" not in st.session_state:
    st.session_state.search_results = []


class EnhancedRAGChatbot:
    def __init__(self):
        self.openai_client = None
        self.chroma_client = None
        self.collection = None
        self.serper_api_key = None

    def initialize_api(self, openai_key, serper_key=None):
        """Initialize OpenAI API and optionally Serper API"""
        try:
            # Initialize OpenAI
            self.openai_client = openai.OpenAI(api_key=openai_key)

            # Test the API key
            self.openai_client.models.list()

            # Initialize Serper API if provided
            if serper_key:
                self.serper_api_key = serper_key
                st.session_state.web_search_enabled = True

            # Initialize ChromaDB
            self.chroma_client = chromadb.Client()

            # Create or get collection
            try:
                self.collection = self.chroma_client.create_collection(
                    name="rag_documents", metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.collection = self.chroma_client.get_collection(
                    name="rag_documents"
                )

            return True
        except Exception as e:
            st.error(f"❌ Error initializing APIs: {str(e)}")
            return False

    def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search using Serper API"""
        if not self.serper_api_key:
            return []

        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query, "num": num_results})
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json",
            }

            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract organic results
                if "organic" in data:
                    for result in data["organic"]:
                        results.append(
                            {
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "link": result.get("link", ""),
                                "source": "web_search",
                            }
                        )

                return results
            else:
                st.error(f"Web search failed with status: {response.status_code}")
                return []

        except Exception as e:
            st.error(f"Web search error: {str(e)}")
            return []

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file with better error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)

            progress_bar = st.progress(0)
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
                progress_bar.progress((i + 1) / total_pages)

            progress_bar.empty()
            return text.strip()
        except Exception as e:
            st.error(f"❌ Error extracting text from PDF: {str(e)}")
            return None

    def extract_youtube_transcript(self, youtube_url):
        """Extract transcript from YouTube video with better URL parsing"""
        try:
            # Extract video ID from URL
            video_id = None
            if "youtube.com/watch?v=" in youtube_url:
                video_id = youtube_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in youtube_url:
                video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
            elif "youtube.com/embed/" in youtube_url:
                video_id = youtube_url.split("embed/")[1].split("?")[0]

            if not video_id:
                st.error("❌ Invalid YouTube URL format")
                return None

            # Get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry["text"] for entry in transcript])

            # Get video title using basic approach
            try:
                video_title = f"YouTube Video ({video_id})"
                return text, video_title
            except:
                return text, f"YouTube Video ({video_id})"

        except Exception as e:
            st.error(f"❌ Error extracting YouTube transcript: {str(e)}")
            return None, None

    def extract_website_content(self, url):
        """Extract content from website with better parsing"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove unwanted elements
            for element in soup(
                ["script", "style", "nav", "footer", "header", "aside"]
            ):
                element.decompose()

            # Get title
            title = soup.title.string if soup.title else "Website Content"

            # Get main content
            main_content = (
                soup.find("main") or soup.find("article") or soup.find("body")
            )
            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
            else:
                text = soup.get_text(separator=" ", strip=True)

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text, title

        except Exception as e:
            st.error(f"❌ Error extracting website content: {str(e)}")
            return None, None

    def generate_embedding(self, text):
        """Generate embedding using OpenAI's text-embedding-3-small model"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=text, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"❌ Error generating embedding: {str(e)}")
            return None

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into chunks for embedding"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def add_to_vector_db(self, text, source_type, source_info):
        """Add text to ChromaDB vector database"""
        try:
            chunks = self.chunk_text(text)
            processed_chunks = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, chunk in enumerate(chunks):
                # Update progress
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i + 1} of {len(chunks)}...")

                # Generate embedding using OpenAI
                embedding = self.generate_embedding(chunk)
                if embedding is None:
                    continue

                # Create unique ID
                doc_id = f"{source_type}_{source_info}_{i}_{uuid.uuid4().hex[:8]}"

                # Add to collection
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[
                        {
                            "source_type": source_type,
                            "source_info": source_info,
                            "chunk_index": i,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ],
                    ids=[doc_id],
                )
                processed_chunks += 1

            progress_bar.empty()
            status_text.empty()

            # Add to processed docs
            st.session_state.processed_docs.append(
                {
                    "type": source_type,
                    "name": source_info,
                    "chunks": processed_chunks,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            )

            st.success(
                f"✅ Successfully processed {processed_chunks} chunks from {source_type}: {source_info}"
            )
            return True
        except Exception as e:
            st.error(f"❌ Error adding to vector database: {str(e)}")
            return False

    def search_vector_db(self, query, n_results=5):
        """Search vector database for relevant chunks"""
        try:
            if not self.collection:
                return []

            # Generate query embedding using OpenAI
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                return []

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            st.error(f"❌ Error searching vector database: {str(e)}")
            return []

    def generate_response(self, user_query, include_web_search=False):
        """Generate response using RAG with optional web search"""
        try:
            # Search vector database
            relevant_chunks = self.search_vector_db(user_query)

            # Perform web search if enabled and requested
            web_results = []
            if include_web_search and self.serper_api_key:
                web_results = self.web_search(user_query)
                st.session_state.search_results = web_results

            # Prepare context
            context = "\n".join(relevant_chunks) if relevant_chunks else ""

            # Add web search results to context
            if web_results:
                web_context = "\n".join(
                    [
                        f"Web Result: {result['title']} - {result['snippet']}"
                        for result in web_results
                    ]
                )
                context = f"{context}\n\nWeb Search Results:\n{web_context}"

            # Prepare prompt
            system_prompt = f"""You are a helpful AI assistant with access to various data sources and real-time web search. 
            Use the following context to answer the user's question comprehensively. If the context doesn't contain 
            relevant information, provide a general response and mention that more specific data might be needed.

            Context from knowledge base:
            {context}

            User Question: {user_query}

            Please provide a comprehensive and helpful response. If you used web search results, mention the sources."""

            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            return response.choices[0].message.content

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."


# Initialize chatbot
def get_chatbot():
    if st.session_state.chatbot is None:
        st.session_state.chatbot = EnhancedRAGChatbot()
    return st.session_state.chatbot


chatbot = get_chatbot()

# Main Header
st.markdown("""
<div class="main-header animate-slide-up">
    <h1>🤖 AI Knowledge Assistant</h1>
    <p>Chat with your documents, websites, and the entire web using advanced AI</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")

    # API Keys Section with enhanced styling
    with st.expander("🔑 API Keys", expanded=not st.session_state.api_initialized):
        st.markdown("#### OpenAI Configuration")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable AI responses",
            placeholder="sk-...",
        )

        st.markdown("#### Web Search Configuration")
        serper_key = st.text_input(
            "Serper API Key (Optional)",
            type="password",
            help="Enter your Serper API key to enable web search",
            placeholder="Your Serper API key",
        )

        if openai_key:
            if st.button(
                "🚀 Initialize APIs", type="primary", use_container_width=True
            ):
                with st.spinner("🔄 Initializing APIs..."):
                    if chatbot.initialize_api(openai_key, serper_key):
                        st.session_state.vector_db = chatbot.chroma_client
                        st.session_state.collection = chatbot.collection
                        st.session_state.api_initialized = True
                        st.success("🎉 APIs initialized successfully!")
                        time.sleep(1)
                        st.rerun()
        else:
            st.warning("⚠️ OpenAI API Key Required")

    # Enhanced Status Section
    st.markdown("### 📊 System Status")

    # API Status with better visual indicators
    if st.session_state.api_initialized:
        st.markdown('<div class="status-indicator status-connected">🟢 OpenAI: Connected</div>', unsafe_allow_html=True)
        if st.session_state.web_search_enabled:
            st.markdown('<div class="status-indicator status-connected">🟢 Web Search: Enabled</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-warning">🟡 Web Search: Disabled</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-indicator status-disconnected">🔴 APIs: Not Connected</div>', unsafe_allow_html=True)

    # Enhanced Database Stats
    st.markdown("### 📈 Analytics")
    
    if st.session_state.collection:
        try:
            doc_count = st.session_state.collection.count()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.5rem;">{doc_count}</h3>
                    <p style="margin: 0; opacity: 0.9;">Document Chunks</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: var(--gradient-secondary);">
                    <h3 style="margin: 0; font-size: 1.5rem;">{len(st.session_state.processed_docs)}</h3>
                    <p style="margin: 0; opacity: 0.9;">Sources</p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="metric-card" style="background: var(--gradient-success);">
                <h3 style="margin: 0; font-size: 1.5rem;">0</h3>
                <p style="margin: 0; opacity: 0.9;">Ready to Start</p>
            </div>
            """, unsafe_allow_html=True)

    # Enhanced Quick Actions
    st.markdown("### ⚡ Quick Actions")

    # Clear All Data with better confirmation
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear All Data", use_container_width=True, help="Remove all processed documents"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); 
                    border-radius: 12px; padding: 1rem; margin: 1rem 0;">
            <p style="margin: 0; color: var(--error-color); font-weight: 500;">
                ⚠️ Are you sure? This action cannot be undone.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm", use_container_width=True, type="primary"):
                if st.session_state.collection:
                    try:
                        # Get all document IDs and delete them
                        all_docs = st.session_state.collection.get()
                        if all_docs["ids"]:
                            st.session_state.collection.delete(ids=all_docs["ids"])

                        # Clear processed docs list
                        st.session_state.processed_docs = []
                        st.session_state.confirm_clear = False
                        st.success("🎉 All data cleared successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing data: {str(e)}")
                        # Alternative approach - recreate collection
                        try:
                            # Delete and recreate collection
                            st.session_state.chroma_client.delete_collection("rag_documents")
                            st.session_state.collection = (
                                st.session_state.chroma_client.create_collection(
                                    name="rag_documents",
                                    metadata={"hnsw:space": "cosine"},
                                )
                            )
                            st.session_state.processed_docs = []
                            st.session_state.confirm_clear = False
                            st.success("🎉 All data cleared (collection recreated)!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e2:
                            st.error(f"Failed to clear data: {str(e2)}")
                            st.session_state.confirm_clear = False
                else:
                    st.session_state.confirm_clear = False

        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    if st.button("🔄 Reset Chat", use_container_width=True, help="Clear chat history"):
        st.session_state.chat_history = []
        st.session_state.search_results = []
        st.success("✅ Chat history cleared!")
        time.sleep(0.5)
        st.rerun()

    # Help Section
    st.markdown("### 💡 Tips")
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Add your OpenAI API key above
        2. Upload documents or add web sources
        3. Start chatting with your AI assistant
        
        **Features:**
        - 📄 PDF document processing
        - 🎥 YouTube transcript extraction
        - 🌐 Website content scraping
        - 🔍 Real-time web search
        - 💬 Intelligent conversation
        """)

# Check if API is initialized
if not st.session_state.api_initialized:
    st.markdown("""
    <div class="status-card animate-fade-in" style="text-align: center; padding: 3rem;">
        <h2>🚀 Welcome to AI Knowledge Assistant</h2>
        <p style="font-size: 1.1rem; margin: 1rem 0;">
            Configure your OpenAI API key in the sidebar to get started
        </p>
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); 
                    border-radius: 12px; padding: 1.5rem; margin: 2rem 0;">
            <p style="margin: 0; color: var(--primary-color); font-weight: 500;">
                💡 <strong>Pro Tip:</strong> Add your Serper API key to enable web search capabilities!
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Enhanced Tab Layout
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📚 Knowledge Base", "🔍 Web Search"])

with tab1:
    st.markdown("""
    <div class="animate-fade-in">
        <h3 style="margin-bottom: 1rem;">🎯 Intelligent Chat Interface</h3>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Chat options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("Ask questions about your uploaded documents or general topics")
    with col2:
        web_search_toggle = st.toggle(
            "🌐 Web Search",
            value=False,
            disabled=not st.session_state.web_search_enabled,
            help="Enable web search for current responses",
        )
    with col3:
        if st.session_state.chat_history:
            chat_export = "\n\n".join(
                [
                    f"**{msg['role'].title()}**: {msg['content']}"
                    for msg in st.session_state.chat_history
                ]
            )
            st.download_button(
                label="📥 Export",
                data=chat_export,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # Enhanced Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; opacity: 0.6;">
            <h3>👋 Hello! How can I help you today?</h3>
            <p>Start by asking a question or uploading some documents to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Chat input
    if prompt := st.chat_input("💭 Ask me anything...", key="chat_input"):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                response = chatbot.generate_response(
                    prompt, include_web_search=web_search_toggle
                )
                st.markdown(response)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat statistics
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Messages", len(st.session_state.chat_history))
        with col2:
            if st.session_state.search_results:
                st.metric("🔍 Web Results", len(st.session_state.search_results))
        with col3:
            user_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
            st.metric("❓ Questions", user_messages)

with tab2:
    st.markdown("""
    <div class="animate-fade-in">
        <h3 style="margin-bottom: 1rem;">📚 Knowledge Base Management</h3>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Data source upload interface
    st.markdown("#### 📤 Add New Data Sources")

    # Enhanced upload interface with better styling
    source_tabs = st.tabs(
        ["📄 PDF Documents", "🎥 YouTube Videos", "🌐 Websites", "📝 Text Input"]
    )

    with source_tabs[0]:
        st.markdown("""
        <div class="source-card">
            <h4 style="margin-top: 0;">📄 Upload PDF Documents</h4>
            <p>Extract and analyze content from PDF files</p>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple PDF documents to extract and analyze their content",
        )

        if pdf_files:
            st.success(f"📎 {len(pdf_files)} PDF(s) selected")

            if st.button(
                "🔄 Process All PDFs", key="process_pdfs", use_container_width=True
            ):
                for pdf_file in pdf_files:
                    with st.spinner(f"📖 Processing {pdf_file.name}..."):
                        text = chatbot.extract_text_from_pdf(pdf_file)
                        if text:
                            chatbot.add_to_vector_db(text, "PDF", pdf_file.name)
                st.rerun()

    with source_tabs[1]:
        st.markdown("""
        <div class="source-card">
            <h4 style="margin-top: 0;">🎥 Extract YouTube Transcripts</h4>
            <p>Get transcripts from YouTube videos for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        youtube_urls = st.text_area(
            "YouTube URLs (one per line)",
            placeholder="https://youtube.com/watch?v=...\nhttps://youtu.be/...",
            help="Enter YouTube URLs to extract transcripts",
            height=120
        )

        if youtube_urls:
            urls = [url.strip() for url in youtube_urls.split("\n") if url.strip()]
            st.info(f"🎬 {len(urls)} URL(s) entered")

            if st.button(
                "🔄 Process Videos", key="process_videos", use_container_width=True
            ):
                for url in urls:
                    with st.spinner(f"🎬 Processing {url}..."):
                        result = chatbot.extract_youtube_transcript(url)
                        if result and len(result) == 2:
                            text, title = result
                            chatbot.add_to_vector_db(text, "YouTube", title)
                st.rerun()

    with source_tabs[2]:
        st.markdown("""
        <div class="source-card">
            <h4 style="margin-top: 0;">🌐 Scrape Website Content</h4>
            <p>Extract content from web pages and articles</p>
        </div>
        """, unsafe_allow_html=True)
        
        website_urls = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://blog.example.com/post",
            help="Enter website URLs to extract content",
            height=120
        )

        if website_urls:
            urls = [url.strip() for url in website_urls.split("\n") if url.strip()]
            st.info(f"🌐 {len(urls)} URL(s) entered")

            if st.button(
                "🔄 Process Websites", key="process_websites", use_container_width=True
            ):
                for url in urls:
                    with st.spinner(f"🌍 Processing {url}..."):
                        result = chatbot.extract_website_content(url)
                        if result and len(result) == 2:
                            text, title = result
                            chatbot.add_to_vector_db(text, "Website", title)
                st.rerun()

    with source_tabs[3]:
        st.markdown("""
        <div class="source-card">
            <h4 style="margin-top: 0;">📝 Direct Text Input</h4>
            <p>Add text content directly to your knowledge base</p>
        </div>
        """, unsafe_allow_html=True)
        
        text_input = st.text_area(
            "Enter text directly",
            placeholder="Paste your text content here...",
            height=200,
            help="Enter text directly to add to knowledge base",
        )

        text_title = st.text_input(
            "Title for this text", 
            placeholder="Give your text a descriptive title"
        )

        if text_input and text_title:
            if st.button("🔄 Add Text", key="process_text", use_container_width=True):
                with st.spinner("📝 Processing text..."):
                    chatbot.add_to_vector_db(text_input, "Text", text_title)
                st.rerun()

    # Enhanced Display processed documents
    st.markdown("---")
    st.markdown("#### 📁 Processed Documents")

    if st.session_state.processed_docs:
        for i, doc in enumerate(st.session_state.processed_docs):
            st.markdown(f"""
            <div class="source-card hover-lift">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 0.5rem 0; color: var(--primary-color);">{doc['name']}</h4>
                        <div style="display: flex; gap: 1rem; align-items: center;">
                            <span style="background: rgba(99, 102, 241, 0.1); color: var(--primary-color); 
                                        padding: 0.25rem 0.75rem; border-radius: 16px; font-size: 0.875rem;">
                                📁 {doc['type']}
                            </span>
                            <span style="background: rgba(16, 185, 129, 0.1); color: var(--success-color); 
                                        padding: 0.25rem 0.75rem; border-radius: 16px; font-size: 0.875rem;">
                                📊 {doc['chunks']} chunks
                            </span>
                            <span style="color: var(--text-light-secondary); font-size: 0.875rem;">
                                🕒 {doc['timestamp']}
                            </span>
                        </div>
                    </div>
                    <div>
            """, unsafe_allow_html=True)
            
            if st.button(
                "🗑️ Remove",
                key=f"delete_{i}_{doc['name']}",
                help="Delete this document",
                type="secondary"
            ):
                try:
                    # Find and delete chunks related to this document
                    if st.session_state.collection:
                        # Get all documents with matching source_info
                        all_docs = st.session_state.collection.get()

                        # Find IDs of chunks belonging to this document
                        ids_to_delete = []
                        for j, metadata in enumerate(all_docs["metadatas"]):
                            if (
                                metadata.get("source_info") == doc["name"]
                                and metadata.get("source_type") == doc["type"]
                            ):
                                ids_to_delete.append(all_docs["ids"][j])

                        # Delete the chunks
                        if ids_to_delete:
                            st.session_state.collection.delete(ids=ids_to_delete)
                            st.success(f"✅ Deleted {len(ids_to_delete)} chunks")

                    # Remove from processed docs
                    st.session_state.processed_docs.remove(doc)
                    st.success(f"✅ Removed {doc['name']} from knowledge base")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting document: {str(e)}")
                    # Still remove from processed docs list
                    st.session_state.processed_docs.remove(doc)
                    st.rerun()
            
            st.markdown("""
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="text-align: center; padding: 3rem;">
            <h3>📭 No documents processed yet</h3>
            <p>Upload some content using the tabs above to get started!</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="animate-fade-in">
        <h3 style="margin-bottom: 1rem;">🔍 Web Search Results</h3>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.web_search_enabled:
        st.markdown("""
        <div class="status-card" style="text-align: center; padding: 3rem;">
            <h3>⚠️ Web Search Disabled</h3>
            <p>Please add your Serper API key in the sidebar to enable web search functionality.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="background: rgba(16, 185, 129, 0.05); border-color: var(--success-color);">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">🌐</span>
                <div>
                    <h4 style="margin: 0; color: var(--success-color);">Web Search Enabled</h4>
                    <p style="margin: 0; color: var(--text-light-secondary);">
                        Search the web for real-time information
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced Manual search interface
        st.markdown("#### 🔎 Manual Web Search")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter your search query...",
                help="Search the web for real-time information",
            )
        with col2:
            num_results = st.selectbox("Results", [3, 5, 10], index=1)

        if search_query:
            if st.button("🔍 Search Web", use_container_width=True, type="primary"):
                with st.spinner("🌐 Searching the web..."):
                    results = chatbot.web_search(search_query, num_results)
                    st.session_state.search_results = results
                st.rerun()

    # Enhanced Display search results
    if st.session_state.search_results:
        st.markdown("#### 📋 Latest Search Results")

        for i, result in enumerate(st.session_state.search_results):
            st.markdown(f"""
            <div class="search-result">
                <h4 style="margin: 0 0 0.5rem 0; color: var(--primary-color);">
                    {i + 1}. {result['title']}
                </h4>
                <p style="margin: 0 0 1rem 0; line-height: 1.6;">
                    {result['snippet']}
                </p>
                <a href="{result['link']}" target="_blank" 
                   style="color: var(--primary-color); text-decoration: none; font-weight: 500;">
                    🔗 Read More →
                </a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="text-align: center; padding: 2rem;">
            <h4>🔍 No search results yet</h4>
            <p>Use the chat with web search enabled or perform a manual search above.</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div class="footer">
    <div style="max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; background: var(--gradient-primary); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🚀 AI Knowledge Assistant
        </h3>
        <p style="margin: 0 0 1rem 0;">
            Powered by <strong>OpenAI GPT-3.5</strong> • <strong>ChromaDB</strong> • <strong>Serper API</strong>
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
            <span>📄 PDF Processing</span>
            <span>🎥 YouTube Integration</span>
            <span>🌐 Web Scraping</span>
            <span>🔍 Real-time Search</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)