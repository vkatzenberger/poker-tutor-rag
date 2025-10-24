from dotenv import load_dotenv
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
import pdfplumber
import pandas as pd
import tiktoken
import logging

# GLOBAL VARIABLES

# Load .env variables
load_dotenv()

# Set up 'PDF_FOLDER' in the .env file which contains the relevant PDFs
PDF_FOLDER = os.getenv("PDF_FOLDER")
os.makedirs(PDF_FOLDER, exist_ok=True)

# Set up 'TABLE_FOLDER' in the .env file which contains the extracted tables
TABLE_FOLDER = os.getenv("TABLE_FOLDER")
os.makedirs(TABLE_FOLDER, exist_ok=True)

# Set up 'STATUS_FILE' in the .env file which keep records of current statuses
STATUS_FILE = os.getenv("STATUS_FILE")

# Set up 'EMBEDDING_MODEL' in the .env file which is the model used for embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Set up 'DB_DIR' in the .env file which is the directory for the database
DB_DIR = os.getenv("DB_DIR")

# Set up 'LOG_FILE' in the .env file which defines the log file
LOG_FILE = os.getenv("LOG_FILE")

# Default logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def load_vectorstore():
    # Function for loading vectorstore
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embedding)

# Loading the vectorstore database
vectorstore = load_vectorstore()

# Helper functions for extracting tables

def clean_table(table_df):

    # Remove empty columns
    table_df = table_df.dropna(axis=1, how='all')

    # Convert to string and remove if only None values
    for column in table_df.columns:
        if all(str(value).strip() == 'None' for value in table_df[column]):
            table_df = table_df.drop(columns=[column])

    # Reset index
    table_df = table_df.reset_index(drop=True)

    # First row as header
    if len(table_df) > 1:
        table_df.columns = table_df.iloc[0]
        table_df = table_df.iloc[1:]

    # Reset again
    table_df = table_df.reset_index(drop=True)

    return table_df

def tables_as_placeholder(pdf_page):
    
    page_text = pdf_page.extract_text()
    # Get table locations
    table_parts = [table.bbox for table in pdf_page.find_tables()]  

    for table_num, area in enumerate(table_parts, start=1):
        # Extract table text
        table_text = pdf_page.crop(area).extract_text()  
        if table_text:
            page_text = page_text.replace(table_text, f"[TABLE {table_num}]") 

    return page_text


def tables_to_markdown(pdf_page):
    tables_md = []
                  
    # Get the tables from the pdf
    tables = pdf_page.extract_tables()
    
    if tables:
        for table in tables:
            if table:
                table_df = pd.DataFrame(table)
                if not table_df.empty:
                    # Clean the table
                    table_df = clean_table(table_df)
                    # Convert to markdown
                    table_md = table_df.to_markdown(index=False)
                    tables_md.append(table_md)
    
    return "\n".join(tables_md) if tables_md else None


# PDF class definition
class PDF:
    def __init__(self, file_name, embedding_model, vectorstore):
        # Protected attributes to prevent direct modifications
        self._file_name = file_name
        self._embedding_model = embedding_model
        self._vectorstore = vectorstore
        self._status = {
            "loaded": False,
            "cleaned": False,
            "chunked": False,
            "annotated": False,
            "vectorized": False
        }
        self.pages = []
        self.tables = []
        self.chunks = []

    # Property functions for getting values from protected attributes
    @property
    def file_name(self):
        return self._file_name

    @property
    def status(self):
        return self._status.copy()
    
    @property
    def pdf_path(self):
        return os.path.join(PDF_FOLDER, self._file_name)
    
    def update_status(self, new_status):
        self._status = new_status
                
    def load(self, pdf_loader = 'PyPDFLoader'):
        # Function for loading the PDF file
        try:
            if pdf_loader == 'PyPDFLoader':
                # Load with PyPDFLoader
                self.log("PDF load with PyPDFLoader")
                loader = PyPDFLoader(self.pdf_path)
                
                for page in loader.load():
                    page.metadata["type"] = "text"
                    self.pages.append(page)                
                
            elif pdf_loader == 'pdfplumber':                
                # Load with pdfplumber - complex method
                self.log("PDF load with pdfplumber")
                
                with pdfplumber.open(self.pdf_path) as pdf:
                    
                    for i, page in enumerate(pdf.pages):
                        # Get the text from the page and put placeholders for tables
                        page_text = tables_as_placeholder(page)
                        
                        if page_text:
                            
                            self.pages.append(Document(
                                page_content=page_text,
                                metadata={
                                    "page": i,
                                    "page_label": i + 1,
                                    "source": self.pdf_path,
                                    "type": "text"
                                }
                            ))
                        
                        # Get the tables from the page and convert to markdown format
                        page_tables = tables_to_markdown(page)
                         
                        if page_tables:
                                
                            name = self.file_name.rstrip(".pdf")
                            txt_file = os.path.join(TABLE_FOLDER, f"{name}_page_{i+1}_table.txt")
                            
                            # Write the tables into the file
                            with open(txt_file, "w", encoding="utf-8") as f:
                                f.write(page_tables)
                            
                            self.tables.append(Document(
                                page_content=page_tables,
                                metadata={
                                    "page": i,
                                    "page_label": i + 1,
                                    "source": self.pdf_path,
                                    "type": "table"
                                }
                            ))
                    
                    print(f"Extracted {len(self.pages)} text pages and {len(self.tables)} tables.")                        
                
                                 
            self._status["loaded"] = True
            self.log("PDF load done!")
            
        except Exception as e:
            logging.error(f"Error with loading the PDF: {e}")
            return
    
    def clean(self):
        # Function for cleaning the text
        if not self._status["loaded"]:
            raise ValueError("Please load the PDF before cleaning.")
        
        try:
            for page in self.pages:
                # This is fixing hyphenated words
                page.page_content = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page.page_content)
                # This removes \n characters
                page.page_content = ' '.join(page.page_content.split())
                # This removes special characters
                page.page_content = re.sub(r'[^\x00-\x7F]+', ' ', page.page_content)

            self._status["cleaned"] = True
            self.log(f"PDF cleaning done!")
            
        except Exception as e:
            logging.error(f"Error with cleaning the PDF: {e}")
            return
                    
    
    def chunk(self, chunk_type = 'token'):
        # Function for chunking the text
        if not self._status["cleaned"]:
            raise ValueError("Please load and clean the PDF before chunking.")
        
        try:
            # Character-based chunking - we are not using it currently
            if chunk_type == 'char':
                chunk_size = 2048
                chunk_overlap = 300    
                char_splitter = CharacterTextSplitter(
                    separator = ". ", 
                    chunk_size = chunk_size, 
                    chunk_overlap = chunk_overlap
                )
                self.chunks = char_splitter.split_documents(self.pages)
                
            # Token-based chunking - that is the default
            elif chunk_type == 'token':
                chunk_size = 512
                chunk_overlap = 128
                
                # OpenAI tiktoken tokenizer
                token_encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL).name
                    
                token_splitter = TokenTextSplitter(
                    encoding_name = token_encoding, 
                    chunk_size = chunk_size, 
                    chunk_overlap = chunk_overlap
                )
                self.chunks = token_splitter.split_documents(self.pages)
                
            else:
                raise ValueError(f"Choose correct chunk type: char or token based approach")
            
            self._status["chunked"] = True
            self.log(f"PDF chunked into {len(self.chunks)} chunks!")
            
        except Exception as e:
            logging.error(f"Error with chunking the PDF: {e}")
            return        
        
    def annotate(self):
        # Function for annotation
            if not self._status["chunked"]:
                raise ValueError("Please load, clean and chunk the PDF before annotating.")

            # Annotation part - if required
            self._status["annotated"] = True
            self.log("PDF annotated!")
    
    def vectorize(self):
        # Function for store into vectorstore
        if not self._status["annotated"]:
            raise ValueError("Please load, clean, chunk and annotate the PDF before vectorizing.")            
        
        try:
            documents = self.chunks + self.tables
            print("\nDocuments to be stored in Vectorstore")
            for doc in documents:
                print(doc.metadata)            
            
            self._vectorstore.add_documents(documents)
            
            self._status["vectorized"] = True
            self.log("PDF vectorized and stored!")
            
        except Exception as e:
            logging.error(f"Error with vectorizing and storing the PDF: {e}")
            return         
            
    def log(self, message):
        logging.info(f"[{self._file_name}] {message}")        