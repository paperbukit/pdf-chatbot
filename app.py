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

# 🧠 Streamlit UI
st.set_page_config(page_title="PDF Chatbot with History", layout="wide")
st.title("📚 Multi-PDF Chatbot (Mistral + OCR + Sources + History + Memory)")

# 🧠 Session state
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    with st.spinner("📖 Reading PDFs..."):
        for uploaded_file in uploaded_files:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
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
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

    llm = Ollama(model="mistral")
    chain = load_qa_chain(llm, chain_type="stuff")

    st.success("✅ PDFs processed!")

    # 🧠 Helper: Create context from last N Q&A
    def get_chat_context(history, current_q):
        context = ""
        for item in history[-3:]:  # last 3 Q&A pairs
            context += f"Q: {item['question']}\nA: {item['answer']}\n"
        context += f"Q: {current_q}\n"
        return context

    question = st.text_input("Ask your question:")

    if question:
        with st.spinner("🤖 Answering..."):
            docs = vectorstore.similarity_search(question, k=3)
            full_prompt = get_chat_context(st.session_state.history, question)
            answer = chain.run(input_documents=docs, question=full_prompt)

            # Store in history
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "sources": [(doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "?")) for doc in docs]
            })

# 📜 Show chat history
if st.session_state.history:
    st.markdown("### 💬 Chat History")
    for i, entry in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"**Q{i}:** {entry['question']}")
        st.markdown(f"**A{i}:** {entry['answer']}")
        for src, pg in entry["sources"]:
            st.markdown(f"📄 `{src} - Page {pg}`")
        st.markdown("---")

# 📤 Export Chat History (No rerun)
if st.session_state.history:
    with st.form("download_form"):
        submit = st.form_submit_button("📥 Download Chat History")
        if submit:
            history_text = ""
            for i, entry in enumerate(st.session_state.history, start=1):
                history_text += f"Q{i}: {entry['question']}\n"
                history_text += f"A{i}: {entry['answer']}\n"
                for src, pg in entry['sources']:
                    history_text += f"Source: {src} - Page {pg}\n"
                history_text += "\n---\n"

            st.download_button(
                label="Click to download",
                data=history_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )
