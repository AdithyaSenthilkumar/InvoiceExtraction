import os
import streamlit as st
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import google.generativeai as genai
import json
import concurrent.futures

# Set Streamlit page configuration
st.set_page_config(page_title="Batch Invoice Processing System", layout="wide")

# Configure environment and generative AI
os.environ["USE_TORCH"] = "1"
genai.configure(api_key="AIzaSyCTpICzzPNhtSCSpqHgjRiILaDWu-0C7oo")

# Initialize Doctr OCR model
@st.cache_resource
def load_ocr_model():
    return ocr_predictor(pretrained=True)

ocr_model = load_ocr_model()

# Helper functions
def process_pdf(file):
    """Process the uploaded PDF and extract text using Doctr OCR."""
    doc = DocumentFile.from_pdf(file)
    results = ocr_model(doc)
    recognized_lines = []
    for block in results.pages[0].blocks:
        for line in block.lines:
            line_text = ' '.join(word.value for word in line.words)
            recognized_lines.append(line_text)
    return "\n".join(recognized_lines)

def extract_invoice_data(ocr_text):
    """Use Google Generative AI to extract structured invoice information."""
    prompt = f"""
        Respond with JSON only, without explanations or additional text.
        Extract supplier name, supplier gst, buyer name, buyer gst invoice number, invoice date, total amount.
        total tax percentage with split(CGST,IGST,SGST)(not null, give 0% instead) from the OCR processed text with % postfix.
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
    
    # Display raw response for debugging
    raw_response = response.text
    st.write("Debug: Raw AI Response", raw_response)
    
    # Clean and parse JSON
    try:
        cleaned_response = raw_response.strip('```json').strip('```').strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing failed: {e}")
        st.write("Debug: Cleaned Response for Manual Validation", cleaned_response)
        return {"error": "Failed to parse JSON", "raw_response": cleaned_response}


def process_single_invoice(file):
    """Process a single invoice and return structured data."""
    ocr_text = process_pdf(file)
    raw_response = extract_invoice_data(ocr_text)
    if "error" in raw_response:
        return {"file_name": file.name, **raw_response}
    return {"file_name": file.name, **raw_response}

# Streamlit app layout
def main():
    st.title("📄 Batch Invoice Processing System")
    st.write("Upload multiple invoice PDFs to extract structured data in parallel.")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing your invoices..."):
            # Process invoices in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_single_invoice, uploaded_files))
            
            # Display results in a table
            st.subheader("📊 Extracted Invoice Data")
            st.write("Processed data is displayed in the table below.")
            if results:
                st.dataframe(results)

# Run the Streamlit app
if __name__ == "__main__":
    main()
