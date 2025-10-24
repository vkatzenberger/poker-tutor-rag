import os
import streamlit as st
import json
import logging
from pdf_class import PDF, EMBEDDING_MODEL, STATUS_FILE, PDF_FOLDER, TABLE_FOLDER, LOG_FILE, vectorstore

# Default logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Helper function, not related to PDF managing
def load_status():
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as file:
                return json.load(file)    
    except Exception as e:
        logging.error(f"Error with status file loading: {e}")
    return {} 


# PDFManager class definition
class PDFManager:
    def __init__(self):
        
        # Load status file
        saved_status = load_status()
        
        if "file_status" not in st.session_state:
            st.session_state.file_status = saved_status
        if "pages" not in st.session_state:
            st.session_state.pages = {}
        if "tables" not in st.session_state:
            st.session_state.tables = {}            
        if "chunks" not in st.session_state:
            st.session_state.chunks = {}
            
        # Store PDF objects    
        self.files = {}
        
        # Initialization of session state for all pdf files
        pdf_files = os.listdir(PDF_FOLDER)
        for file_name in pdf_files:
            if file_name not in st.session_state.file_status:
                st.session_state.file_status[file_name] = {
                    "loaded": False,
                    "cleaned": False,
                    "chunked": False,
                    "annotated": False,
                    "vectorized": False
                }
    
    def get_pdf(self, file_name):
        # If found already in the cache, return the object
        if file_name in self.files:
            return self.files[file_name]
        
        # If there is no pdf object, create and store it
        pdf = PDF(file_name, EMBEDDING_MODEL, vectorstore)
        self.files[file_name] = pdf
        
        # If there is file status in session state, we update it
        if file_name in st.session_state.file_status:
            pdf.update_status(st.session_state.file_status[file_name])
        
        return pdf
    
    
    def load_and_store(self, file_name, pdf_loader):
        # PDF processing and storing into session state
        pdf = self.get_pdf(file_name)
        pdf.load(pdf_loader)
        st.session_state.pages[file_name] = pdf.pages
        st.session_state.tables[file_name] = pdf.tables
        st.session_state.file_status[file_name]["loaded"] = True
        
    def clean_and_store(self, file_name):
        pdf = self.get_pdf(file_name)
        pdf.pages = st.session_state.pages[file_name]
        pdf.clean()
        st.session_state.pages[file_name] = pdf.pages
        st.session_state.file_status[file_name]["cleaned"] = True
        
    def chunk_and_store(self, file_name):
        pdf = self.get_pdf(file_name)
        pdf.pages = st.session_state.pages[file_name]
        pdf.chunk()
        st.session_state.chunks[file_name] = pdf.chunks
        st.session_state.file_status[file_name]["chunked"] = True

    def annotate(self, file_name):
        pdf = self.get_pdf(file_name)
        pdf.annotate()
        st.session_state.file_status[file_name]["annotated"] = True
        
    def vectorize_chunks(self, file_name):
        pdf = self.get_pdf(file_name)
        pdf.chunks = st.session_state.chunks[file_name]
        pdf.vectorize()
        st.session_state.file_status[file_name]["vectorized"] = True
        # Save status after vectorization
        self.save_status(file_name)
    
       
    def save_status(self, file_name):
        # Function for saving status into status file 
        try:
            if not st.session_state.file_status[file_name]["vectorized"]:
                logging.error(f"PDF {file_name} is not vectorized!")
                return

            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, "r") as file:
                    all_status = json.load(file)
            else:
                all_status = {}

            all_status[file_name] = st.session_state.file_status[file_name]

            with open(STATUS_FILE, "w") as file:
                json.dump(all_status, file, indent=4)            
            
        except Exception as e:
            logging.error(f"Error saving status file: {e}")
    
          
    def clear_all(self):
        # Clear all status and new vectorstore
        try:
            if os.path.exists(STATUS_FILE):
                os.remove(STATUS_FILE)             
            
            for file in os.listdir(TABLE_FOLDER):
                file_path = os.path.join(TABLE_FOLDER, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    
            vectorstore.reset_collection() 
              
            st.session_state.file_status = {}
            st.session_state.pages = {}
            st.session_state.chunks = {}
            st.session_state.tables = {}
            
            
            # We reinit PDFManager, and automatically store in session state
            st.session_state.pdf_manager = PDFManager()    

            logging.info("Cleared all documents.")
                      
        except Exception as e:
            logging.error(f"Error with clearing: {e}")