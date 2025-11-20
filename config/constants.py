import os
from pathlib import Path

#paths
BASE_DIR = Path(__file__).resolve.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploaded_docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
TEMP_DIR + DATA_DIR / "temp"

for directory in [DATA_DIR, UPLOAD_DIR, CHROMA_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


SUPPORTED_EXTENSIONS = {
        ".pdf", ".txt", ".doc", ".docx", ".xlsx", 
        ".pptx", ".ppt", ".xls", ".csv", ".json", 
        ".md", ".html", ".xml"
        }

MAX_FILES_SIZE = 10 * 1024 * 1024
CHROMA_COLLECTION_NAME = "rag_documents"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 200
MAX_WEB_RESULTS = 5
MAX_WIKIPIDIA_RESULTS = 4
