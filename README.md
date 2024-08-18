# Contract Review LLM

A Streamlit application for reviewing contract documents in PDF, TXT, and CSV formats using a Large Language Model (LLM). This application extracts text from various document formats, creates embeddings, and allows for interactive conversations to review contract content.

## Features

- **Document Processing**: Extracts and processes text from PDF, TXT, and CSV files.
- **Text Chunking**: Splits extracted text into manageable chunks.
- **Vectorstore Creation**: Uses embeddings to create a searchable vectorstore.
- **Conversational AI**: Interacts with users to review contract content and provide feedback.
- **Streamlit Interface**: User-friendly web interface for document upload and interaction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OvaizAli/Contract-LLM.git
   cd contract-review-llm
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project directory with the following content:
   ```
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`) to interact with the application.

## Functions

- `get_pdf_text(pdf_docs)`: Extracts text from PDF documents.
- `get_txt_text(txt_docs)`: Extracts text from TXT documents.
- `get_csv_text(csv_docs)`: Extracts text from CSV documents.
- `get_text_chunks(text)`: Splits text into chunks using `CharacterTextSplitter`.
- `get_vectorstore(text_chunks)`: Creates a vectorstore from text chunks using FAISS and Hugging Face embeddings.
- `save_embeddings(vectorstore)`: Saves the vectorstore embeddings to a file.
- `load_embeddings()`: Loads the vectorstore embeddings from a file.
- `get_conversation_chain(vectorstore)`: Creates a conversational chain using the vectorstore and Hugging Face endpoint.
- `handle_userinput(user_question)`: Handles user input and displays conversation responses.
- `process_documents()`: Processes all documents in the `CONTRACTS_DIR`, creates embeddings, and saves them.
- `main()`: Main function to run the Streamlit app.

## File Structure

- `app.py`: Main Streamlit application script.
- `htmltemplates.py`: Contains HTML templates for styling chat messages.
- `requirements.txt`: List of Python dependencies.
- `.env`: Environment variables for API tokens and configuration.
- `Contracts/`: Directory containing contract documents (PDF, TXT, CSV).

## Dependencies

- `streamlit`
- `dotenv`
- `PyPDF2`
- `pandas`
- `langchain`
- `langchain_community`
- `langchain_huggingface`
- `FAISS`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain/langchain) for text processing and conversational chains.
- [Hugging Face](https://huggingface.co) for the embeddings and models.
- [Streamlit](https://streamlit.io) for the interactive web interface.
