import os
from langchain_core.documents import Document
from typing import List
import docx
import pypdf as pdf

def text_to_doc(text: str, source: str) -> Document:
    return Document(page_content=text, metadata={"source": source})

def load_txt(path: str) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        return [text_to_doc(text, os.path.abspath(path))]

def load_docx(path:str) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs)
    return [text_to_doc(text, os.path.abspath(path))]

def load_pdf(path:str) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError
    pdf_data = pdf.PdfReader(path)
    docs = []
    for i, page in enumerate(pdf_data.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(text_to_doc(text, f"{os.path.abspath(path)}:page:{i + 1}"))
    return docs

def load_document(path:str) -> list[Document] | None:
    if not os.path.exists(path):
        raise FileNotFoundError
    allowed_extensions = [".txt", ".docx", ".pdf"]
    file_extension= os.path.splitext(path)[1].lower()
    if file_extension not in allowed_extensions:
        raise ValueError(f"File extension {file_extension} not allowed")
    try:
        if file_extension == ".txt":
            return load_txt(path)
        elif file_extension == ".docx":
            return load_docx(path)
        elif file_extension == ".pdf":
            return load_pdf(path)
    except Exception as e:
        raise RuntimeError(f"Exception raised when trying to load the file: {e}") from e
