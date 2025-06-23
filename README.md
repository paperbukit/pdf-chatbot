# 📚 PDF Chatbot with Groq Llama 3


A beautiful, modern multi-PDF chatbot powered by [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), and [Groq Llama 3](https://console.groq.com/). Upload PDFs, ask questions, and get instant answers with OCR, translation, and chat history export.

---

## ✨ Features
- **Multi-PDF Upload:** Chat with multiple PDFs at once
- **OCR Support:** Extract text from scanned PDFs
- **Translation:** Instantly translate extracted text (toggle in sidebar)
- **Groq Llama 3 LLM:** Fast, free, and powerful open-source LLM via Groq API
- **Chat History:** View and export your Q&A session
- **Modern UI:** Clean, responsive, and user-friendly

---

## 🚀 Quickstart
1. **Clone the repo:**
   ```bash
   git clone https://github.com/paperbukit/pdf-chatbot.git
   cd pdf-chatbot
   ```
2. **Create a virtual environment (Python 3.11 recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Add your Groq API key:**
   - Get a free key at [Groq Console](https://console.groq.com/keys)
   - Add it to `.streamlit/secrets.toml`:
     ```toml
     GROQ_API_KEY = "your-groq-api-key"
     ```
5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Live Demo
Check out the live demo: [PDF Chatbot](https://paperbukit-pdf-chatbot.streamlit.app/)

---

## 🖼️ Screenshot

![App Screenshot](Screenshot%202025-06-24%20030035.png)

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM:** Groq Llama 3 (llama3-8b-8192)
- **Embeddings:** Sentence Transformers (MiniLM)
- **OCR:** Tesseract
- **Translation:** Deep Translator
- **Vector DB:** FAISS

---

## ⚡ Plugins & Customization
- Toggle OCR and translation in the sidebar
- Easily swap LLMs or add new plugins (translation, summarization, etc.)

---

> Made with ❤️ using Streamlit, LangChain, and Groq