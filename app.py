import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
st.set_page_config(page_title="LGH Thoracic Guide", page_icon="🫁")
st.title("🫁 Thoracic Resident Assistant")

# --- API KEY SETUP ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("⚠️ OpenAI API Key not found. Please set it in secrets.toml or Streamlit Cloud.")
    st.stop()

# --- BRAIN 1: PROTOCOLS (Loads at startup) ---
@st.cache_resource
def load_protocol_brain():
    folder_path = "protocols"
    all_splits = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return None

    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not files:
        return None

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)
    
    if all_splits:
        embeddings = OpenAIEmbeddings()
        vector = FAISS.from_documents(all_splits, embeddings)
        return vector
    return None

# --- BRAIN 2: TEXTBOOK (Loads only when requested) ---
@st.cache_resource
def load_textbook_brain():
    folder_path = "library" # Separate folder for textbooks
    all_splits = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return None

    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not files:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)
    
    if all_splits:
        embeddings = OpenAIEmbeddings()
        vector = FAISS.from_documents(all_splits, embeddings)
        return vector
    return None

# --- INITIALIZATION ---
protocol_vector = load_protocol_brain()

# Setup Protocol Chain
if protocol_vector:
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Strict prompt that forces a specific "NOT_FOUND" code if it doesn't know
    protocol_prompt = ChatPromptTemplate.from_template("""
    You are a strict resident assistant. 
    Search the following context from the Service Protocols.
    If the answer is clearly in the context, answer it.
    If the answer is NOT in the context, you must reply with exactly: "NOT_IN_PROTOCOL".
    Do not make up an answer.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    protocol_chain = create_retrieval_chain(protocol_vector.as_retriever(), create_stuff_documents_chain(llm, protocol_prompt))
else:
    st.warning("⚠️ No Service Guides found in 'protocols' folder.")

# --- CHAT INTERFACE & LOGIC ---

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am ready. Ask me a protocol question."}]
if "pending_textbook_search" not in st.session_state:
    st.session_state["pending_textbook_search"] = None # Holds the question waiting for textbook search

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- MAIN LOGIC LOOP ---
if prompt := st.chat_input("Enter clinical scenario..."):
    # 1. Add user query to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Search Protocols First
    if protocol_vector:
        with st.spinner("Checking Service Protocols..."):
            response = protocol_chain.invoke({"input": prompt})
            answer = response['answer']
        
        # 3. Check if Protocol failed
        if "NOT_IN_PROTOCOL" in answer:
            # Don't answer yet. Set state to "Waiting for user confirmation"
            st.session_state["pending_textbook_search"] = prompt
            
            # Create a system message asking for permission
            ask_msg = "⚠️ This is not in the official Service Guide. Would you like to search the 'TRSA 2011' textbook?"
            st.session_state.messages.append({"role": "assistant", "content": ask_msg})
            st.chat_message("assistant").write(ask_msg)
            # We force a rerun to show the buttons (implemented below)
            st.rerun()
            
        else:
            # Found in protocol -> Show answer
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
            # Clear any pending searches since we found it
            st.session_state["pending_textbook_search"] = None

# --- TEXTBOOK CONFIRMATION BUTTONS ---
# This block only appears if we are waiting for a "Yes"
if st.session_state["pending_textbook_search"]:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📖 Yes, Search Textbook"):
            query = st.session_state["pending_textbook_search"]
            
            # Load Textbook Brain (Only happens now!)
            with st.spinner("Loading Textbook (This happens once)..."):
                textbook_vector = load_textbook_brain()
            
            if textbook_vector:
                # Create Textbook Chain
                llm = ChatOpenAI(model="gpt-4o-mini")
                textbook_prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the textbook context below.
                Start your answer with: "**[SOURCE: TRSA Textbook]**" so the user knows this is not a hospital protocol.

                <context>
                {context}
                </context>

                Question: {input}
                """)
                textbook_chain = create_retrieval_chain(textbook_vector.as_retriever(), create_stuff_documents_chain(llm, textbook_prompt))
                
                with st.spinner("Reading Textbook..."):
                    response = textbook_chain.invoke({"input": query})
                    answer = response['answer']
                
                # Append Answer
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state["pending_textbook_search"] = None # Reset state
                st.rerun()
            else:
                st.error("Textbook file not found in 'library' folder.")
                
    with col2:
        if st.button("❌ No, Cancel"):
            st.session_state.messages.append({"role": "assistant", "content": "Okay, search cancelled."})
            st.session_state["pending_textbook_search"] = None
            st.rerun()