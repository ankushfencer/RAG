import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter
import fitz  # PyMuPDF for PDF handling
import os
from dotenv import load_dotenv
import nltk
from langchain_groq import ChatGroq
from transformers import GPT2TokenizerFast  # For token counting

# Download NLTK data if necessary
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Define path to where PDF documents are stored
DOCUMENTS_PATH = "./pdf_documents"

# User Access Control with sample roles and access
# Define access control for different roles
# Each role is allowed access to certain files only
USER_ACCESS = {
    "admin": {"files": ["guidelines.pdf", "healthins.pdf", "IIScSociety.pdf"]},
    "researcher": {"files": ["healthins.pdf", "IIScSociety.pdf"]},
    "end_user": {"files": ["IIScSociety.pdf"]}
}

# Mock user database for demonstration purposes
# In practice, passwords should be stored securely (hashed)
USER_DB = {
    "admin_user": {"password": "admin_pass", "role": "admin"},
    "research_user": {"password": "research_pass", "role": "researcher"},
    "end_user": {"password": "end_pass", "role": "end_user"}
}

def initialize_api_keys():
    """Initialize API keys from environment variables or Streamlit secrets."""
    # Store API keys in session state for secure access throughout the app
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if 'GROQ_API_KEY' not in st.session_state:
        st.session_state.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if 'HUGGINGFACE_API_KEY' not in st.session_state:
        st.session_state.HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))
    return st.session_state.OPENAI_API_KEY, st.session_state.HUGGINGFACE_API_KEY, st.session_state.GROQ_API_KEY

# Authentication
def authenticate_user(username, password):
    """Authenticate user based on username and password."""
    user_info = USER_DB.get(username)
    if user_info and user_info["password"] == password:
        # Set session state based on verified role
        st.session_state["username"] = username
        st.session_state["role"] = user_info["role"]
        st.session_state["allowed_files"] = USER_ACCESS[user_info["role"]]["files"]
        st.success(f"Logged in as {username} with role {user_info['role']}")
        return True
    else:
        st.error("Authentication failed. Please check your username and password.")
        return False

# Load PDFs based on access
def load_pdfs():
    """Load PDF documents based on the authenticated user's access level."""
    accessible_files = st.session_state.get("allowed_files", [])
    documents = []

    # Loop through each accessible file and load its content
    for filename in accessible_files:
        filepath = os.path.join(DOCUMENTS_PATH, filename)
        if os.path.exists(filepath):
            with fitz.open(filepath) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text() 
                documents.append(text) # Add the document's text to the list
        else:
            st.warning(f"File {filename} not found.")
    return documents

# Chunk documents
def load_and_chunk_documents(documents):
    """Split loaded documents into chunks based on selected chunking method."""
    # Choose the chunking method based on user's selection
    if st.session_state.chunking_method == "Character-based":
        text_splitter = CharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
    elif st.session_state.chunking_method == "Sentence-based":
        text_splitter = NLTKTextSplitter()  

    # Split each document into chunks and store them in a list  
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

# Create embeddings and store in a role-specific vector database
def create_vector_database(chunks):
    """Create a vector database from document chunks based on role-specific embeddings."""

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    role_db_path = f"./chroma_db_{st.session_state['role']}"  # Create a role-specific database path
    db = Chroma.from_texts(chunks, embeddings, persist_directory=role_db_path)
    db.persist()
    return db

# Token counting utility
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_token_count(text):
    """Count the number of tokens in the given text."""
    return len(tokenizer.encode(text))

def fit_chunks_within_limit(chunks, token_limit=6000):
    """Limit the total tokens in chunks to fit within the token limit."""
    token_count = 0
    fitted_chunks = []
    for chunk in chunks:
        chunk_tokens = get_token_count(chunk)
        if token_count + chunk_tokens > token_limit:
            break # Stop if adding the next chunk exceeds the token limit
        fitted_chunks.append(chunk)
        token_count += chunk_tokens
    return fitted_chunks

# Initialize session state variables for settings and chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "use_mmr" not in st.session_state:
    st.session_state.use_mmr = False
if "use_open_source_llm" not in st.session_state:
    st.session_state.use_open_source_llm = True
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1500
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "chunking_method" not in st.session_state:
    st.session_state.chunking_method = "Character-based"
if "allowed_files" not in st.session_state:
    st.session_state["allowed_files"] = []

# Define a function to clear the chat
def clear_chat():
    st.session_state.messages = []

# Main UI
st.set_page_config(page_title="RAG Chat Interface", layout="wide")
st.title("RAG Chat Interface")

# Initialize API keys
openai_api_key, huggingface_api_key, groq_api_key = initialize_api_keys()

# Sidebar for Login and Logout
st.sidebar.title("User Authentication")
if "username" not in st.session_state:  # Display login fields if not logged in
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        authenticate_user(username, password)
else:
    # Display logout button if the user is logged in
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout", type="primary"):
        for key in ["username", "role", "allowed_files", "messages"]:
            st.session_state.pop(key, None)  # Clear only specific session keys
        st.rerun()  # Reload the app to reset

# Additional settings in the sidebar for retrieval and chunking options
if "username" in st.session_state:  # Show options only if logged in
    st.session_state.use_mmr = st.sidebar.checkbox("Enable MMR for retrieval")
    st.session_state.use_open_source_llm = st.sidebar.checkbox("Use Open-Source LLM")
    st.session_state.chunk_size = st.sidebar.slider("Chunk Size", min_value=500, max_value=3000, value=1500, step=100)
    st.session_state.chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    st.session_state.chunking_method = st.sidebar.selectbox("Chunking Method", options=["Character-based", "Sentence-based"])

# RAG System Execution with restricted access
if st.session_state.get("username"):
    documents = load_pdfs()  # Only load documents based on the allowed role
    if not documents:
        st.warning("No documents available for your role.")
    else:
        # Process documents into chunks and store in vector database
        chunks = load_and_chunk_documents(documents)
        db = create_vector_database(chunks)  # Create database with role-specific docs
        
        # Define prompt template with context and question
        template = """
        You are an intelligent assistant who only answers questions based on the given context from the documents.
        If the answer is not present in the provided context, respond with 
        "The answer is not available in the provided documents."

        Use the conversation history to understand the ongoing context and answer the user's query.

        Context: {context}

        Question: {question}
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        # Select LLM based on user's choice of open-source or proprietary model
        if st.session_state.use_open_source_llm:
            llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=groq_api_key)
        else:
            llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")

        # Set up memory to retain chat history
        memory = ConversationBufferMemory(memory_key="chat_history")

        # Configure the retrieval chain, optionally using Maximal Marginal Relevance (MMR)
        if st.session_state.use_mmr:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 2}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                verbose=True
            )
        else:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                verbose=True
            )

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input for questions
        user_input = st.chat_input("Your message here...")
        if user_input:
            # Store user message in chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").markdown(user_input)

            # Generate assistant response based on user input and retrieved context
            response = qa_chain({"question": user_input})
            assistant_response = response["answer"]

            # Store assistant response in chat history and display it
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.chat_message("assistant").markdown(assistant_response)

        # Display Clear Chat button only if there are messages
        if st.session_state.messages:
            if st.button("Clear Chat"):
                clear_chat()
                st.rerun() # Refresh the interface
else:
    st.warning("Please log in to access the RAG Chat Interface.")
