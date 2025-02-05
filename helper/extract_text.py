import pypdf
import os

def extract_text_from_pdf(pdf_path="data/book.pdf"):
    if not os.path.exists(pdf_path):
        print(f"{pdf_path} not found")
        return
    if os.path.exists("generated/extracted_text.txt"):
        print("generated/extracted_text.txt already exists")
        return
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        
    with open("generated/extracted_text.txt", "w") as f:
        f.write(text)
        
    print("Extracted text created and saved to generated/extracted_text.txt")

extract_text_from_pdf()
