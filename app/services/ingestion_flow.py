import uuid
import pandas as pd
from typing import List

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.vector_store import VectorStoreManager
from app.core.dependencies import vector_store


import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io

def ingest_wikipedia_url(url: str = "https://en.wikipedia.org/wiki/Madagascar") -> List[Document]:
    """Fetches and parses a Wikipedia page, returning a list of Documents."""
    headers = {
        'User-Agent': 'RAG_Madagascar_Project'
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    content_div = soup.find(id="mw-content-text").find(class_="mw-parser-output")

    for sup in content_div.find_all("sup", class_="reference"):
        sup.decompose()
        
    for span in content_div.find_all("span", class_="mw-editsection"):
        span.decompose()

    documents = []
    current_h2 = "Introduction"
    current_h3 = ""
    TABLE_CHUNK_SIZE = 10
    STOP_SECTIONS = ["see also", "notes", "references", "external links"]

    for element in soup.find_all(['h2', 'h3', 'p', 'table']):
        if element.name == 'h2':
            text = element.get_text(strip=True)
            clean_header = re.sub(r'\[.*?\]', '', text).strip()
            
            if clean_header.lower() in STOP_SECTIONS:
                print(f"Arrêt de l'ingestion à la section : {clean_header}")
                break
            if text and text not in ["Contents", "Navigation menu", "Personal tools"]:
                current_h2 = text
                current_h3 = "" 
                
        elif element.name == 'h3':
            current_h3 = element.get_text(strip=True)

        elif element.name == 'p':
            text = element.get_text(strip=True)
            if len(text) > 50 and not text.startswith("This article is about"):
                metadata = {
                    "source": url,
                    "h2_section": current_h2,
                    "h3_section": current_h3,
                    "content_type": "text"
                }
                documents.append(Document(page_content=text, metadata=metadata))

        elif element.name == 'table':
            try:
                html_string = str(element)
                df = pd.read_html(io.StringIO(html_string))[0]
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(str(c) for c in col if c).strip() for col in df.columns]
                
                headers_list = df.columns.tolist()
                
                for i in range(0, len(df), TABLE_CHUNK_SIZE):
                    chunk_df = df.iloc[i : i + TABLE_CHUNK_SIZE]
                    chunk_lines = []
                    
                    for _, row in chunk_df.iterrows():
                        row_elements = []
                        for col in headers_list:
                            val = str(row[col])
                            if val and val.lower() not in ['nan', 'none']:
                                clean_col = re.sub(r'\[.*?\]', '', str(col)).strip()
                                clean_val = re.sub(r'\[.*?\]', '', val).strip()
                                row_elements.append(f"{clean_col}: {clean_val}")
                        
                        if row_elements:
                            chunk_lines.append(" | ".join(row_elements))
                    
                    if chunk_lines:
                        table_content = "Extract from the table:\n" + "\n".join(chunk_lines)
                        
                        metadata = {
                            "source": url,
                            "h2_section": current_h2,
                            "h3_section": current_h3,
                            "content_type": "table"
                        }
                        documents.append(Document(page_content=table_content, metadata=metadata))
                    
            except ValueError as ve:
                pass
            except Exception as e:
                print(f"Erreur tableau ignorée dans {current_h2}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    final_documents = []
    for doc in documents:
        if doc.metadata["content_type"] == "text":
            chunks = text_splitter.split_documents([doc])
            final_documents.extend(chunks)
        else:
            final_documents.append(doc)
            
    # Add a unique ID to each document
    for doc in final_documents:
        doc.metadata["id"] = str(uuid.uuid4())

    print(f"Nombre total de chunks prêts pour Qdrant : {len(final_documents)}")
    return final_documents

def insert_documents_in_vector_store(documents: List[Document]):
    """Orchestrates the ingestion of documents into the vector store."""
    if not documents:
        return []

    db_manager = VectorStoreManager(vector_store)
    inserted_ids = db_manager.insert_documents(documents)
    
    return inserted_ids
