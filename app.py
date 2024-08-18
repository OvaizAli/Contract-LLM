import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
import os
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
CONTRACTS_DIR = "./Contracts"  # Directory containing contract documents
EMBEDDINGS_FILE = "embeddings.pkl"

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from TXT documents
def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        try:
            text += txt.read().decode("utf-8")
        except UnicodeDecodeError:
            text += txt.read().decode("latin1")
    return text

# Function to extract text from CSV documents
def get_csv_text(csv_docs):
    text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        text += df.to_string(index=False)
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not text_chunks:
        raise ValueError("No text chunks to create vectorstore.")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to save vectorstore embeddings
def save_embeddings(vectorstore):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(vectorstore, f)

# Function to load vectorstore embeddings
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return None

# Function to create a conversational chain
def get_conversation_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display conversation
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history[-1:]):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# Function to process documents and create embeddings
def process_documents():
    all_text = ""

    # Process all PDF files
    pdf_files = [os.path.join(CONTRACTS_DIR, file) for file in os.listdir(CONTRACTS_DIR) if file.endswith(".pdf")]
    if pdf_files:
        all_text += get_pdf_text(pdf_files)

    # Process all TXT files
    txt_files = [os.path.join(CONTRACTS_DIR, file) for file in os.listdir(CONTRACTS_DIR) if file.endswith(".txt")]
    if txt_files:
        all_text += get_txt_text(txt_files)

    # Process all CSV files
    csv_files = [os.path.join(CONTRACTS_DIR, file) for file in os.listdir(CONTRACTS_DIR) if file.endswith(".csv")]
    if csv_files:
        all_text += get_csv_text(csv_files)

    if not all_text:
        st.error("No text extracted from the documents.")
        return

    # Create and save embeddings
    text_chunks = get_text_chunks(all_text)
    if not text_chunks:
        st.error("No text chunks created from the extracted text.")
        return

    vectorstore = get_vectorstore(text_chunks)
    save_embeddings(vectorstore)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Contract Review LLM", page_icon=":brain:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.spinner("Loading or creating embeddings..."):
        vectorstore = load_embeddings()
        if not vectorstore:
            st.warning("No embeddings found. Processing documents...")
            process_documents()
            vectorstore = load_embeddings()

    if vectorstore:
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("Ready to Access your Contracts Now!")

    st.header("Contract Review LLM (PDF, TXT, CSV) :brain:")

    uploaded_file = st.file_uploader("Upload a contract file (PDF, TXT, CSV):", type=["pdf", "txt", "csv"])

    if st.button("Review Contract"):
        if uploaded_file:
            all_text = ""

            if uploaded_file.type == "application/pdf":
                all_text += get_pdf_text([uploaded_file])
            elif uploaded_file.type == "text/plain":
                all_text += get_txt_text([uploaded_file])
            elif uploaded_file.type == "text/csv":
                all_text += get_csv_text([uploaded_file])

            if not all_text:
                st.error("No text extracted from the uploaded file.")
                return

            # Improved query for reviewing the contract
            improved_query = "As a professional contract reviewer. Review each line of the contract. If you identify any issues related to that line, state the issue and provide the best possible correction for each issue identified."

            # Handle user input with the query
            handle_userinput(improved_query)

if __name__ == '__main__':
    main()
