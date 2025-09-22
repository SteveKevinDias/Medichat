#Reading a pdf file

from pypdf import PdfReader
from typing import List,Optional
from io import BytesIO

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)
    texts = ' '
    for page in reader.pages:
        texts += page.extract_text() or ' '
    return texts


