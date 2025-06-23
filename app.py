import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# 📌 Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Streamlit UI setup
st.set_page_config(page_title="PDF Chatbot with History", layout="wide")
st.title("📚 Multi-PDF Chatbot (Mistral + OCR + Sources + History)")

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

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.session_state.saved_files = uploaded_files

# PDF preview/thumbnail display
# Store file content in session state to avoid multiple reads
if uploaded_files:
    st.markdown("### 📄 PDF Previews")
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state:
            st.session_state[uploaded_file.name] = uploaded_file.read()

        file_content = st.session_state[uploaded_file.name]
        if file_content:  # Ensure the file is not empty
            doc = fitz.open(stream=file_content, filetype="pdf")
            first_page = doc[0]
            pix = first_page.get_pixmap(dpi=100)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            st.image(img, caption=uploaded_file.name, use_container_width=True)
        else:
            st.warning(f"The file {uploaded_file.name} is empty and cannot be processed.")

# Process PDFs only once
if st.session_state.saved_files and not st.session_state.processed:
    all_docs = []
    with st.spinner("📖 Reading PDFs..."):
        for uploaded_file in st.session_state.saved_files:
            file_content = st.session_state[uploaded_file.name]  # Reuse stored content
            if file_content:  # Ensure the file is not empty
                doc = fitz.open(stream=file_content, filetype="pdf")
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        content = text
                    else:
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        content = pytesseract.image_to_string(img)

                    all_docs.append({
                        "text": content,
                        "metadata": {"source": uploaded_file.name, "page": i + 1}
                    })

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

    st.session_state.chain = load_qa_chain(Ollama(model="mistral"), chain_type="stuff")
    st.session_state.processed = True
    st.success("✅ PDFs processed!")

# Text input
question = st.text_input("Ask your question:")

# Prevent duplicate answering
if question and question != st.session_state.last_question:
    st.session_state.last_question = question

    # Combine previous answers for context
    context = "\n".join([entry['answer'] for entry in st.session_state.history])

    with st.spinner("🤖 Answering with context..."):
        docs = st.session_state.vectorstore.similarity_search(question, k=3)
        answer = st.session_state.chain.run(input_documents=docs, question=f"{context}\n{question}")

        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "sources": [(doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "?")) for doc in docs]
        })

# Display history
if st.session_state.history:
    st.markdown("### 💬 Chat History")
    for i, entry in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"**Q{i}:** {entry['question']}")
        st.markdown(f"**A{i}:** {entry['answer']}")
        for src, pg in entry["sources"]:
            st.markdown(f"📄 `{src} - Page {pg}`")
        st.markdown("---")

# Export chat history
if st.session_state.history:
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
