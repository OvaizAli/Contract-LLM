
        # default_query = "Review each line of the contract and identify specific issues related to that contract. Provide the best possible correction for each issue too."

        # default_query = "Review each line of the contract to identify issues and suggest corrections where necessary. Please highlight any ambiguities, errors, or areas needing clarification, and propose improvements or amendments to ensure clarity, accuracy, and compliance with legal standards."




import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template

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
    embeddings = OpenAIEmbeddings()
    if not text_chunks:
        raise ValueError("No text chunks to create vectorstore.")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display conversation
def handle_userinput(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    st.empty()  # Hide initial UI elements

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Contract Review LLM", page_icon=":brain:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.conversation = None
        st.session_state.chat_history = None
        
        st.session_state.contracts_folder = "./Contracts"  # Adjust this to the directory where your contracts are stored
        st.session_state.uploaded_files = os.listdir(st.session_state.contracts_folder)
        st.session_state.all_text = ""

        if st.session_state.uploaded_files:
            with st.spinner("Processing Documents..."):
                pdf_files = [os.path.join(st.session_state.contracts_folder, file) for file in st.session_state.uploaded_files if file.endswith(".pdf")]
                if pdf_files:
                    st.session_state.all_text += get_pdf_text(pdf_files)

                txt_files = [os.path.join(st.session_state.contracts_folder, file) for file in st.session_state.uploaded_files if file.endswith(".txt")]
                if txt_files:
                    st.session_state.all_text += get_txt_text(txt_files)

                csv_files = [os.path.join(st.session_state.contracts_folder, file) for file in st.session_state.uploaded_files if file.endswith(".csv")]
                if csv_files:
                    st.session_state.all_text += get_csv_text(csv_files)

                if not st.session_state.all_text:
                    st.error("No text extracted from the uploaded documents.")
                    return

                # Get the text chunks
                text_chunks = get_text_chunks(st.session_state.all_text)
                if not text_chunks:
                    st.error("No text chunks created from the extracted text.")
                    return

                # Create vector store
                st.session_state.vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    st.subheader("Paste the contract content or upload a file for review")

    # Text area for pasting contract content
    contract_text = st.text_area("Content of the contract for review:", height=300)

    # File uploader for uploading contract files
    uploaded_file = st.file_uploader("Or upload a contract file (PDF, TXT, CSV):", type=["pdf", "txt", "csv"])

    if st.button("Review Contract"):
        all_text = ""
        
        if contract_text:
            all_text += contract_text
        
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                all_text += get_pdf_text([uploaded_file])
            elif uploaded_file.type == "text/plain":
                all_text += get_txt_text([uploaded_file])
            elif uploaded_file.type == "text/csv":
                all_text += get_csv_text([uploaded_file])

        if not all_text:
            st.error("No text provided or extracted from the uploaded file.")
            return

        # Improved query for reviewing the contract
        improved_query = "Review each line of the contract and identify specific issues related to that contract. Provide the best possible correction for each issue too."
        
        # Handle user input with the query
        handle_userinput(improved_query, st.session_state.conversation)

if __name__ == '__main__':
    main()
