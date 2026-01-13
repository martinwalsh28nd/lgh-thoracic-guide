import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="LGH Thoracic Guide", page_icon="🫁")
st.title("🫁 Thoracic Resident Assistant")

# --- API KEY SETUP ---
# Ideally, we will move this to a secure "Secrets" file later.
os.environ["OPENAI_API_KEY"] = "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" 

# --- THE BRAIN (CACHED) ---
# This function runs ONLY once when the app starts, not every time you click.
@st.cache_resource
def load_and_process_pdfs():
    folder_path = "protocols"
    all_docs = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) # Create it if it's missing to avoid errors
        return None

    # Loop through every file in the 'protocols' folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not files:
        return None

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
        all_docs.extend(docs)
    
    # Create the search index
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(all_docs, embeddings)
    return vector

# --- INITIALIZATION ---
vector_store = load_and_process_pdfs()

if vector_store is None:
    st.error("⚠️ No PDFs found! Please put your service guides in the 'protocols' folder.")
else:
    # Set up the logic chain
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful resident assistant for the Thoracic Surgery service.
    Answer the user's question based ONLY on the following context from our protocols.
    If the answer is not in the context, say "I don't find a specific protocol for that."
    
    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    st.success("✅ Service Protocols Loaded & Ready")

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "I've read the protocols. What clinical scenario can I help with?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt_input := st.chat_input("E.g., management of post-op air leak..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        st.chat_message("user").write(prompt_input)

        # Generate Response
        response_dict = retrieval_chain.invoke({"input": prompt_input})
        answer = response_dict['answer']

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)