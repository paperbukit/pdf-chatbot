import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq  # <-- add this import
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from deep_translator import GoogleTranslator

# 📌 Tesseract path
# Set Tesseract path dynamically for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Set Tesseract path for Streamlit Cloud

# Streamlit UI setup
st.set_page_config(page_title="PDF Chatbot with History", layout="wide")

# Clean, minimal styling
st.markdown("""
<style>
    /* Clean, minimal styling */
    .stButton > button {
        border-radius: 4px;
    }
    
    /* Simple code blocks for sources */
    code {
        padding: 2px 5px;
        background-color: #f0f0f0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 PDF Chatbot")

# Ensure translate_text function is defined and accessible
def translate_text(text, target_language="en"):
    """
    Translate the given text to the target language.

    Args:
        text (str): The text to translate.
        target_language (str): The language code to translate to (default is English).

    Returns:
        str: Translated text.
    """
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# Initialize session state
for key, val in {
    "history": [],
    "vectorstore": None,
    "chain": None,
    "processed": False,
    "saved_files": [],
    "last_question": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Initialize conversational memory with explicit output key
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.session_state.saved_files = uploaded_files

# PDF preview/thumbnail display
if uploaded_files:
    st.markdown("### 📄 PDF Previews")
    
    cols = st.columns(min(3, len(uploaded_files)))
    
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name not in st.session_state:
            st.session_state[uploaded_file.name] = uploaded_file.read()

        file_content = st.session_state[uploaded_file.name]
        if file_content:  # Ensure the file is not empty
            with cols[i % 3]:
                doc = fitz.open(stream=file_content, filetype="pdf")
                first_page = doc[0]
                pix = first_page.get_pixmap(dpi=100)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                st.image(img, caption=uploaded_file.name, use_container_width=True)
                st.caption(f"{doc.page_count} pages | {uploaded_file.size//1024} KB")
        else:
            with cols[i % 3]:
                st.warning(f"The file {uploaded_file.name} is empty and cannot be processed.")

# Sidebar plugin system
st.sidebar.title("Plugin Settings")
ocr_enabled = st.sidebar.checkbox("Enable OCR", value=True, key="ocr_toggle")
translation_enabled = st.sidebar.checkbox("Enable Translation", value=False, key="translation_toggle")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Mistral AI")

# Process PDFs only once
if st.session_state.saved_files and not st.session_state.processed:
    all_docs = []
    
    with st.spinner("� Reading PDFs..."):
        progress_bar = st.progress(0)
        total_files = len(st.session_state.saved_files)
        
        for idx, uploaded_file in enumerate(st.session_state.saved_files):
            progress_bar.progress((idx) / total_files)
            
            file_content = st.session_state[uploaded_file.name]  # Reuse stored content
            if file_content:  # Ensure the file is not empty
                doc = fitz.open(stream=file_content, filetype="pdf")
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        content = text
                    elif ocr_enabled:  # Apply OCR only if enabled
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        content = pytesseract.image_to_string(img)
                    else:
                        content = ""  # Skip OCR if disabled

                    if translation_enabled and content.strip():
                        content = translate_text(content)  # Apply translation if enabled

                    all_docs.append({
                        "text": content,
                        "metadata": {"source": uploaded_file.name, "page": i + 1}
                    })
        
        progress_bar.progress(1.0)

    with st.spinner("✂️ Chunking..."):
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        split_docs = []
        for doc in all_docs:
            chunks = splitter.split_text(doc["text"])
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc["metadata"]))

    with st.spinner("🔍 Embedding..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

    # Use Groq LLM
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama3-8b-8192"  # or "mixtral-8x7b-32768" for larger model
    )
    st.session_state.chain = load_qa_chain(llm, chain_type="stuff")
    st.session_state.processed = True
    st.success("✅ PDFs processed!")

# Update chain to use ConversationalRetrievalChain with explicit output key
if st.session_state.vectorstore:
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama3-8b-8192"
    )
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=st.session_state.memory,
        return_source_documents=True,  # Ensure source documents are included in the result
        output_key="answer"  # Explicitly set the output key for memory
    )

# Text input
question = st.text_input("Ask your question:")

# Prevent duplicate answering
if question and question != st.session_state.last_question:
    st.session_state.last_question = question

    with st.spinner("🤖 Answering with memory..."):
        result = st.session_state.chain({"question": question})
        st.session_state.history.append({
            "question": question,
            "answer": result["answer"],
            "sources": [(doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "?")) for doc in result["source_documents"]]
        })

# Display history
if st.session_state.history:
    st.markdown("### 🔄 Chat History")
    
    for i, entry in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"**Q{i}:** {entry['question']}")
        st.markdown(f"**A{i}:** {entry['answer']}")
        
        for src, pg in entry["sources"]:
            st.markdown(f"📄 `{src} - Page {pg}`")
        
        st.markdown("---")

    # Export chat history
    history_text = ""
    for i, entry in enumerate(st.session_state.history, start=1):
        history_text += f"Q{i}: {entry['question']}\n"
        history_text += f"A{i}: {entry['answer']}\n"
        for src, pg in entry["sources"]:
            history_text += f"Source: {src} - Page {pg}\n"
        history_text += "---\n"

    st.download_button(
        label="Download Chat History",
        data=history_text,
        file_name="chat_history.txt",
        mime="text/plain",
        key="download_button"
    )
