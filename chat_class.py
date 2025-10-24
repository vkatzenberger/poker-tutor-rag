import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from textwrap import dedent

# Helper function, not related to Chat class
def remove_path(filepath):
    return os.path.basename(filepath)

# PokerTutor Chat class definition
class PokerTutor:
    def __init__(self, vectorstore, model_name, seed=365, max_tokens=500):
        self.vectorstore = vectorstore
        self.chat = ChatOpenAI(model_name=model_name, seed=seed, max_tokens=max_tokens)
        self.setup_memory()
        self.prompt_template = self.create_prompt()

    def setup_memory(self):
        # Function for set up memory
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            
            # Get the settings
            name = st.session_state.get("name") or "User"
            style = st.session_state.get("style", "Normal")
            focus = st.session_state.get("focus", "General")
            mode = st.session_state.get("mode", "Documents")
            
            # Configure style settings
            style_settings = {
            "Normal": "Answer concisely with relevant information",
            "Explain": "Explain it like I am seven years old",
            "Summarize": "Provide a short, clear summary of the topic",
            "Step by Step": "Break it down into logical step-by-step explanations"
            }
            
            style_prompt = style_settings.get(style, style_settings["Normal"])
            
            # Strict RAG Documents instructions
            if mode == "rag":
                mode_instructions = dedent("""
                - **Answer ONLY using the provided context for questions.**
                - If the answer is **NOT** in the context, say "I don't know. I don't have enough information from the documents."
                - **Do NOT generate** information beyond the given sources.
                """)
            # General Knowledge instructions
            elif mode == "general":
                mode_instructions = dedent("""
                - **Prefer the provided context** when answering questions.
                - If no relevant information is found, use general poker knowledge **ONLY if highly confident**.
                - If unsure, say "I don't know."
                - **Do NOT generate** speculative or misleading information.
                """)             
                
            # The full system starting prompt
            system_prompt = f"""
            You are a **poker strategy assistant** helping user {name} with poker-related questions.

            ### Instructions:
            - **Response Style:** {style_prompt}
            - **Focus:** Center answers on {focus} topics
            - Follow the selected style and focus consistently.
            - Maintain conversation context across answers and ensure consistency.

            {mode_instructions}

            - Each source has a **name - page followed by colon** and the actual information.
            - Always reference sources using **square brackets**, e.g., [source1.pdf].
            - If multiple sources apply, reference all relevant ones.

            {name} will ask poker-related questions.
            """
            st.session_state.memory.chat_memory.add_message(SystemMessage(content=system_prompt))
            
            # Welcome message
            if not any(isinstance(msg, AIMessage) for msg in st.session_state.memory.chat_memory.messages):
                st.session_state.memory.chat_memory.add_message(
                    AIMessage(content=f"Hello {name}! How can I assist you with your poker questions today?")
                )

    def create_prompt(self):
        # Default prompt template
        prompt_template = PromptTemplate.from_template("""
            You are a poker strategy assistant. Follow the same instructions provided at the start of this chat.
                                                        
            Question:
            {question}

            Context:
            {context}

            History:
            {history}
        """)
        
        return prompt_template

    def retrieve_context(self, prompt):
        # Function for retrieval
        
        # Retriever for tables
        table_retriever = self.vectorstore.similarity_search_with_score(prompt, k=1, filter={"type": "table"})

        table_docs = []
        table_score_max = 0.40

        print("\nRetrieved Tables")
        for doc, score in table_retriever:
            # Evaluating results
            if score <= table_score_max:
                doc.metadata["score"] = score 
                print(doc.metadata)
                table_docs.append(doc)

        # Retriever for texts
        text_retriever = self.vectorstore.similarity_search_with_score(prompt, k=3, filter={"type": "text"})

        text_docs = []
        text_score_max = 0.50

        print("\nRetrieved Text")
        for doc, score in text_retriever:
            # Evaluating results
            if score <= text_score_max:
                doc.metadata["score"] = score 
                print(doc.metadata)
                text_docs.append(doc)

        # Combine retrieved tables and texts
        retrieved_docs = table_docs + text_docs

        context = "\n\n".join(self.format_context(doc) for doc in retrieved_docs) or "No relevant sources."

        return context
      
    
    def display_chat(self):
        # Chat history using Langchain memory
        for message in st.session_state.memory.chat_memory.messages:
        
            role = ""
            if isinstance(message, HumanMessage):
                role = "user" 
            elif isinstance(message, AIMessage): 
                role = "assistant"   
            else:
            # Optional, display system prompt for testing, otherwise just skip with continue
            #    role = "system"
                continue
            
            with st.chat_message(role):
                st.markdown(message.content)  

    def get_user_input(self):
        # Function for getting user input
        prompt = st.chat_input("Type your message here")
        
        if not prompt:
            return

        # Store user message in memory
        st.session_state.memory.chat_memory.add_message(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve context from vector store
        context = self.retrieve_context(prompt)

        # Get history
        history = "\n\n".join(self.format_history(message) for message in st.session_state.memory.chat_memory.messages)

        chain = (
            {"history": RunnablePassthrough(), "context": RunnablePassthrough(), "question": RunnablePassthrough()} 
            | self.prompt_template
            | self.chat
            | StrOutputParser()
        )

        # AI response
        with st.chat_message("assistant"):
            response = st.write_stream(chain.stream({"history": history, "context": context, "question": prompt}))

        # Store AI response in memory
        st.session_state.memory.chat_memory.add_message(AIMessage(content=response))

        # Show context
        with st.expander("Retrieved Context"):
            st.text(context)
            
    @staticmethod
    def format_context(doc):
        source = remove_path(doc.metadata.get('source', 'Unknown'))
        page_label = doc.metadata.get('page_label', 'Unknown')
        return f"[{source} - Page {page_label}]:  {doc.page_content}\n"

    @staticmethod
    def format_history(message):
        return f"{message.__class__.__name__}: {message.content}"