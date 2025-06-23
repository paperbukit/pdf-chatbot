import streamlit as st
import fitz  # PyMuPDF

st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")
st.title("📄 PDF Q&A Chatbot")
st.markdown("Upload a PDF and start asking questions (coming soon).")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")
