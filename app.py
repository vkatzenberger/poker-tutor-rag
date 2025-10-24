from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pdf_class import PDF_FOLDER, vectorstore
from pdf_manager import PDFManager
from chat_class import PokerTutor

st.set_page_config(
    page_title="Poker Tutor", 
    page_icon="♠️",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS

st.markdown(
    """
    <style>        
        .block-container {
            max-width: 50% !important;
            text-align: center;
        }

        .appview-container .main {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        h1, h2, h3, h4, h5, h6 {
            text-align: center !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# GLOBAL VARIABLES

# Load .env variables
load_dotenv()

# Set up 'OPENAI_API_KEY' in the .env file which contains the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up 'OPENAI_MODEL' in the .env file which is the preferred language model
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# HELPER FUNCTIONS

# For setup completion
def setup_complete():
    st.session_state.setup_complete = True
    
# Selecting all PDFs to process
def select_all_pdfs():
    for file_name in pdf_files:
        st.session_state.selected_pdfs[file_name] = True

# Selecting none PDFs to process
def select_none_pdfs():
    for file_name in pdf_files:
        st.session_state.selected_pdfs[file_name] = False

# Set up a spinner for longer functions
def spinner(progress, function):
    with st.spinner(progress):
        function()
    st.success(f"{progress} done!")

# Process all documents with a progress bar
def process_all_with_progress():
    
    pdf_files = os.listdir(PDF_FOLDER)
    # Get the selected pdfs
    selected_pdfs = [pdf_file for pdf_file in pdf_files if st.session_state.selected_pdfs.get(pdf_file, True)]
    
    # Set progress bar to zero
    progress = st.progress(0)
    
    full = len(selected_pdfs)

    for i, file_name in enumerate(selected_pdfs):
        try:
            # Step by step processing for each pdf
            pdf_manager.load_and_store(file_name, pdf_loader)
            pdf_manager.clean_and_store(file_name)
            pdf_manager.chunk_and_store(file_name)
            pdf_manager.annotate(file_name)
            pdf_manager.vectorize_chunks(file_name)

        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
            
        # Updating progress bar, first visually, and then with label
        progress.progress((i + 1) / full, f"{file_name} ({i + 1} / {full})")

    st.success("Process all documents done!")


# STREAMLIT INTERFACE

# Create PDFManager class
if "pdf_manager" not in st.session_state:
    st.session_state.pdf_manager = PDFManager()

pdf_manager = st.session_state.pdf_manager

st.markdown("<h1>♠️ Poker Tutor</h1>", unsafe_allow_html=True)

# Session state to track setup completion
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
    
# SETUP PHASE
if not st.session_state.setup_complete:
            
    # List files
    pdf_files = os.listdir(PDF_FOLDER)
    if st.session_state.file_status:
        pdfs = pd.DataFrame.from_dict(st.session_state.file_status, orient="index")
    else:
        pdfs = pd.DataFrame()    

    # Side bar to Select PDFs
    with st.sidebar:
        
        st.subheader("Select PDFs to Process")
        
        # For keeping session_state for selected_pdfs
        if "selected_pdfs" not in st.session_state:
            st.session_state.selected_pdfs = {}
            
         # Keeping checkbox states
        for file_name in pdf_files:
            if file_name not in st.session_state.selected_pdfs:
                # This checks and changes value if the PDF is already processed
                is_processed = st.session_state.file_status.get(file_name, {}).get("vectorized", False)
                st.session_state.selected_pdfs[file_name] = not is_processed
                    
        col1, col2 = st.columns([1,1])
        
        with col1:
            st.button("Select All", key="select_all", on_click=select_all_pdfs)
        with col2:
            st.button("Select None", key="select_none", on_click=select_none_pdfs)
                    

        for file_name in pdf_files:
            is_processed = st.session_state.file_status.get(file_name, {}).get("vectorized", False)
            st.session_state.selected_pdfs[file_name] = st.checkbox(
                file_name,
                value=st.session_state.selected_pdfs[file_name],
                key=f"selected_pdfs_{file_name}",
                disabled=is_processed
            )

    
    # PDF table in Streamlit
    st.subheader("Documents")
    st.dataframe(pdfs, height=250, use_container_width=True)
    
    # Clear All button
    st.button("Clear All", key="clear_all", on_click=pdf_manager.clear_all)   

    # PDF selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_pdf = st.selectbox("Choose a pdf:", pdf_files if pdf_files else ["No pdfs"])
    
    # Loader selection
    pdf_loaders = ["PyPDFLoader", "pdfplumber"]
    
    with col2:
        pdf_loader = st.selectbox("Select Loader:", pdf_loaders)

    # PDF processing buttons
    if selected_pdf and selected_pdf != "No pdfs":
        
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 2])

        col1.button("Load", key=f"load_{selected_pdf}", 
                    on_click=lambda: spinner("Loading", lambda: pdf_manager.load_and_store(selected_pdf, pdf_loader)), 
                    disabled=st.session_state.file_status[selected_pdf]["loaded"]
                    )

        col2.button("Clean", key=f"clean_{selected_pdf}", 
                    on_click=lambda: spinner("Cleaning", lambda: pdf_manager.clean_and_store(selected_pdf)), 
                    disabled=not st.session_state.file_status[selected_pdf]["loaded"] or
                    st.session_state.file_status[selected_pdf]["cleaned"]
                    )

        col3.button("Chunk", key=f"chunk_{selected_pdf}", 
                    on_click=lambda: spinner("Chunking", lambda: pdf_manager.chunk_and_store(selected_pdf)), 
                    disabled=not st.session_state.file_status[selected_pdf]["cleaned"] or
                    st.session_state.file_status[selected_pdf]["chunked"])

        col4.button("Annotate", key=f"annotate_{selected_pdf}", 
                    on_click=lambda: spinner("Annotating", lambda: pdf_manager.annotate(selected_pdf)), 
                    disabled=not st.session_state.file_status[selected_pdf]["chunked"] or
                    st.session_state.file_status[selected_pdf]["annotated"])

        col5.button("Vectorize", key=f"vectorize_{selected_pdf}", 
                    on_click=lambda: spinner("Cleaning", lambda: pdf_manager.vectorize_chunks(selected_pdf)), 
                    disabled=not st.session_state.file_status[selected_pdf]["annotated"] or
                    st.session_state.file_status[selected_pdf]["vectorized"])
        
        col6.button("Process All Documents", key="process_all", 
                    on_click=lambda: process_all_with_progress()
                    ) 
      

    # Prompt modifiers

    st.subheader("Settings")
        
    if "name" not in st.session_state:
            st.session_state["name"] = ""
    if "style" not in st.session_state:
            st.session_state["style"] = "Normal"
    if "focus" not in st.session_state:
            st.session_state["focus"] = "Poker Basics"
    if "mode" not in st.session_state:
            st.session_state["mode"] = "Documents"
    
    col1, col2 = st.columns([3, 1])
    
    # Add name
    with col1:        
        st.session_state["name"] = st.text_input(label = "Name", max_chars = 50, value = st.session_state["name"], placeholder = "Enter your name")


    # Mode selection
    mode_options = {
        "rag": ":book: **Only RAG**",
        "general": ":brain: **General Knowledge**"
    }

    with col2:
        mode = st.radio(
            label="Select Mode:",
            options=list(mode_options.keys()),
            format_func=lambda key: mode_options[key],
            index=0
        )
       
    st.session_state["mode"] = mode

    # Style and Focus selection
    col1, col2 = st.columns([1, 1])

    with col1:
        st.session_state["style"] = st.selectbox(
                                label="Style: ", 
                                options=["Normal", "Explain", "Summarize", "Step by Step"],
                                index=["Normal", "Explain", "Summarize", "Step by Step"].index(st.session_state["style"])
                                )

    with col2:
        st.session_state["focus"] = st.selectbox(
                                label="Focus: ", 
                                options=["Poker Basics", "Expected Value", "Bluffing"],
                                index=["Poker Basics", "Expected Value", "Bluffing"].index(st.session_state["focus"])
                                )
        
    st.button("I'm Ready", on_click=setup_complete)    
    
    
    
# CHAT PHASE (SETUP COMPLETED) 
if st.session_state.setup_complete:
    
    # Create Chat class
    chat = PokerTutor(vectorstore, OPENAI_MODEL)
    
    # Display chat history for current session
    chat.display_chat()
    
    # Get new input
    chat.get_user_input()
    
    