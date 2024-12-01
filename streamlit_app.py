import os
import streamlit as st
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import google.generativeai as genai
import json
from concurrent.futures import ThreadPoolExecutor

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Batch Invoice Processing System", layout="wide")

# Configure environment and generative AI
os.environ["USE_TORCH"] = "1"
genai.configure(api_key="AIzaSyCTpICzzPNhtSCSpqHgjRiILaDWu-0C7oo")

# Initialize Doctr OCR model
def load_ocr_model():
    return ocr_predictor(pretrained=True)

ocr_model = load_ocr_model()

# Define helper functions
def process_pdf(file):
    """Process the uploaded PDF and extract text using Doctr OCR."""
    try:
        doc = DocumentFile.from_pdf(file)
        results = ocr_model(doc)
        recognized_lines = []
        for block in results.pages[0].blocks:
            for line in block.lines:
                line_text = ' '.join(word.value for word in line.words)
                recognized_lines.append(line_text)
        return "\n".join(recognized_lines)
    except Exception as e:
        return f"Error processing PDF: {e}"

def extract_invoice_data(ocr_text):
    """Use Google Generative AI to extract structured invoice information."""
    prompt = f"""
        Respond with JSON only, without explanations or additional text.
        Extract supplier name, supplier gst, buyer name, buyer gst invoice number, invoice date, total amount.
        total tax percentage with split (CGST, IGST, SGST) (not null, give 0% instead) from the OCR processed text with % postfix.
        No explanation, just json, no backquotes or comments.
        If some fields are unrecognizable, just fill with context or null.
        Verify the total amount with the total in words, words is final.
        NOTE:
        1. Sometimes the total amount may have a prefix of a rupee symbol that is being recognized as '2'.
        2. For total tax, add up components like SGST, CGST but be careful of duplicates.
        3. If tax amount is given, calculate the percentage from the total.
        The OCR processed Text: {ocr_text}
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        raw_response = response.text.strip()

        # Clean the response
        json_str = raw_response.replace('```json', '').replace('```', '').strip()

        # Parse JSON
        parsed_data = json.loads(json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing failed: {e}", "raw_response": raw_response}
    except Exception as e:
        return {"error": f"Failed to process AI response: {e}"}

def process_invoice(file):
    """Process a single invoice and extract data."""
    ocr_text = process_pdf(file)
    return extract_invoice_data(ocr_text)

# Streamlit app layout
def main():
    st.title("📄 Batch Invoice Processing System")
    st.write("Upload multiple invoice PDFs to extract structured data using OCR and AI.")

    # File uploader for batch processing
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing invoices..."):
            # Process invoices in parallel
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_invoice, uploaded_files))

            # Display results
            st.subheader("📊 Extracted Invoice Data")
            if results:
                for idx, result in enumerate(results):
                    st.write(f"**Invoice {idx + 1}:**")
                    if "error" in result:
                        st.error(result.get("error"))
                        st.write("Debug Raw Response:", result.get("raw_response", "No raw response available"))
                    else:
                        st.json(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
