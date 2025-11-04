# 🤖 AI-Powered RAG Web Search Assistant

An intelligent knowledge assistant that combines Retrieval-Augmented Generation (RAG) with real-time web search capabilities. Chat with your documents, websites, YouTube videos, and the entire web using advanced AI.

## ✨ Features

### 🔍 Multi-Source Knowledge Integration
- **Document Processing**: Upload and process PDF files
- **Web Content Extraction**: Extract content from any website URL
- **YouTube Transcripts**: Process YouTube video transcripts automatically
- **Real-time Web Search**: Search the web for up-to-date information using Serper API

### 🧠 Advanced AI Capabilities
- **RAG (Retrieval-Augmented Generation)**: Combines your knowledge base with AI responses
- **Vector Database**: ChromaDB for efficient similarity search
- **OpenAI Integration**: GPT-3.5-turbo for intelligent responses
- **Semantic Search**: Find relevant information using embeddings

### 🎨 Modern User Interface
- **Streamlit-based**: Clean, responsive web interface
- **Dark/Light Mode**: Automatic theme adaptation
- **Real-time Chat**: Interactive conversation interface
- **Progress Tracking**: Visual feedback for all operations
- **Analytics Dashboard**: Track processed documents and system status

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Serper API key (optional, for web search)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/akaTalhakhan/AI_POWERED_RAG_WEBSEARCH.git
   cd AI_POWERED_RAG_WEBSEARCH
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 🔧 Configuration

### API Keys Setup

1. **OpenAI API Key** (Required)
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Enter it in the sidebar configuration panel

2. **Serper API Key** (Optional)
   - Get your API key from [Serper.dev](https://serper.dev/)
   - Enables real-time web search functionality

### Environment Variables (Optional)
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

## 📖 Usage Guide

### 1. Initialize APIs
- Enter your OpenAI API key in the sidebar
- Optionally add Serper API key for web search
- Click "Initialize APIs"

### 2. Add Knowledge Sources

#### Upload Documents
- Go to the "Document Upload" tab
- Upload PDF files
- Wait for processing to complete

#### Add Website Content
- Go to the "Web Content" tab
- Enter any website URL
- The system will extract and process the content

#### Process YouTube Videos
- Go to the "YouTube" tab
- Enter a YouTube video URL
- Automatic transcript extraction and processing

### 3. Chat with Your Knowledge Base
- Use the "Chat" tab for conversations
- Ask questions about your uploaded content
- Enable web search for real-time information
- View sources and references for each response

## 🏗️ Architecture

### Core Components

```
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .venv/                 # Virtual environment
└── README.md              # This file
```

### Technology Stack
- **Frontend**: Streamlit with custom CSS styling
- **AI/ML**: OpenAI GPT-3.5-turbo, text-embedding-3-small
- **Vector Database**: ChromaDB
- **Web Search**: Serper API
- **Document Processing**: PyPDF2, BeautifulSoup4, YouTube Transcript API

### Data Flow
1. **Input Processing**: Documents/URLs → Text extraction
2. **Chunking**: Large texts → Manageable chunks
3. **Embedding**: Text chunks → Vector embeddings
4. **Storage**: Embeddings → ChromaDB vector database
5. **Retrieval**: User query → Relevant chunks via similarity search
6. **Generation**: Context + Query → AI response

## 🔍 Features Deep Dive

### Document Processing
- **PDF Support**: Extracts text from multi-page PDFs
- **Progress Tracking**: Real-time processing updates
- **Chunking Strategy**: Overlapping chunks for better context retention
- **Metadata Storage**: Source tracking and timestamps

### Web Content Extraction
- **Smart Parsing**: Removes navigation, ads, and irrelevant content
- **Multiple Formats**: Supports various website structures
- **Error Handling**: Graceful handling of inaccessible content
- **Title Extraction**: Automatic page title detection

### Vector Search
- **Semantic Similarity**: Find relevant content regardless of exact keyword matches
- **Configurable Results**: Adjust number of retrieved chunks
- **Cosine Similarity**: Optimized for text embeddings
- **Fast Retrieval**: Efficient similarity search

### Web Search Integration
- **Real-time Results**: Up-to-date information from the web
- **Source Attribution**: Clear indication of web vs. knowledge base sources
- **Configurable Results**: Adjust number of search results
- **Error Resilience**: Fallback to knowledge base if web search fails

## 🎨 User Interface

### Modern Design
- **Responsive Layout**: Works on desktop and mobile
- **Custom Styling**: Professional gradient themes
- **Interactive Elements**: Hover effects and animations
- **Status Indicators**: Clear system status visualization

### Navigation
- **Tabbed Interface**: Organized feature access
- **Sidebar Configuration**: Easy API key management
- **Progress Feedback**: Visual progress for all operations
- **Error Messages**: Clear error reporting and guidance

## 🔒 Security & Privacy

### API Key Handling
- **Secure Input**: Password-masked input fields
- **Session Storage**: Keys stored only in session state
- **No Persistence**: Keys not saved to disk

### Data Privacy
- **Local Processing**: Documents processed locally
- **Vector Storage**: Only embeddings stored, not raw text
- **No External Storage**: All data remains in your environment

## 🚀 Performance Optimization

### Efficient Processing
- **Chunking Strategy**: Optimized chunk sizes for better retrieval
- **Batch Processing**: Efficient embedding generation
- **Progress Tracking**: User feedback during long operations
- **Error Recovery**: Graceful handling of processing failures

### Memory Management
- **Session State**: Efficient state management
- **Vector Database**: Optimized storage and retrieval
- **Streaming**: Large file processing without memory issues

## 🛠️ Troubleshooting

### Common Issues

#### API Connection Problems
```
Error: Invalid API key
Solution: Verify your OpenAI API key is correct and has sufficient credits
```

#### Document Processing Failures
```
Error: PDF extraction failed
Solution: Ensure PDF is not password-protected or corrupted
```

#### Web Search Not Working
```
Error: Web search failed
Solution: Check Serper API key and internet connection
```

### Debug Mode
Enable debug information by adding to your environment:
```env
STREAMLIT_LOGGER_LEVEL=debug
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug mode
streamlit run main.py --logger.level=debug
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT and embedding models
- **ChromaDB** for the vector database solution
- **Streamlit** for the web application framework
- **Serper** for web search API services

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/akaTalhakhan/AI_POWERED_RAG_WEBSEARCH/issues)
- **Discussions**: [GitHub Discussions](https://github.com/akaTalhakhan/AI_POWERED_RAG_WEBSEARCH/discussions)
- **Email**: [Contact Developer](taalhakhaan7@gmail.com)

## 🔮 Roadmap

### Upcoming Features
- [ ] Support for more document formats (Word, Excel, PowerPoint)
- [ ] Multi-language support
- [ ] Advanced search filters
- [ ] Export conversation history
- [ ] API endpoint for programmatic access
- [ ] Integration with more AI models
- [ ] Collaborative knowledge bases
- [ ] Advanced analytics and insights

### Version History
- **v1.0.0** - Initial release with core RAG functionality
- **v1.1.0** - Added web search integration
- **v1.2.0** - Enhanced UI and user experience
- **v1.3.0** - Performance optimizations and bug fixes

---

**Made  by [Talha Khan](https://github.com/akaTalhakhan)**

*Transform your documents and web content into an intelligent, searchable knowledge base with the power of AI.*