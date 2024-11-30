import os
import streamlit as st
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import google.generativeai as genai
import json

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Invoice Processing System", layout="wide")

# Configure environment and generative AI
os.environ["USE_TORCH"] = "1"
genai.configure(api_key="AIzaSyCTpICzzPNhtSCSpqHgjRiILaDWu-0C7oo")

# Initialize Doctr OCR model
@st.cache_resource
def load_ocr_model():
    return ocr_predictor(pretrained=True)

ocr_model = load_ocr_model()

# Define helper functions
def process_pdf(file):
    """Process the uploaded PDF and extract text using Doctr OCR."""
    doc = DocumentFile.from_pdf(file)
    results = ocr_model(doc)
    recognized_lines = []
    for block in results.pages[0].blocks:
        for line in block.lines:
            line_text = ' '.join(word.value for word in line.words)
            recognized_lines.append(line_text)
    return recognized_lines

def extract_invoice_data(ocr_text):
    """Use Google Generative AI to extract structured invoice information."""
    prompt = f"""
        Respond with JSON only, without explanations or additional text.
        Extract supplier name, invoice number, invoice date, total amount,
        total tax percentage(not null give 0% instead) from the OCR processed text.
        No explanation, just json, no backquotes or comments.
        If some fields are unrecognizable, just fill with context or null.
        Verify the total amount with the total in words, words is final.
        NOTE:
        1. Sometimes the total amount may have a prefix of rupee symbol that is being recognized as '2'
        2. For total tax, add up components like SGST, CGST but careful of duplicates
        3. If tax amount is given you calculate the percentage from the total
        The OCR processed Text: {ocr_text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Streamlit app layout
def main():
    st.title("📄 Invoice Processing System")
    st.write("Upload an invoice PDF to extract structured data using OCR and AI.")

    # File uploader
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded_file:
        with st.spinner("Processing your invoice..."):
            # Process the PDF
            ocr_text = process_pdf(uploaded_file)
            st.subheader("🔍 OCR Extracted Text")
            st.text_area("Extracted Text", value="\n".join(ocr_text), height=300)

            # Generate structured invoice data
            invoice_data = extract_invoice_data(ocr_text)
            st.subheader("🔍 AI Extracted Raw Response")
            st.text(invoice_data)
            invoice_json = json.loads(invoice_data)

            # Display structured data
            st.subheader("📊 Extracted Invoice Data")
            st.json(invoice_json)

# Run the Streamlit app
if __name__ == "__main__":
    main()
