from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

app = Flask(__name__)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = "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"

@app.route('/ask', methods=['POST'])
def ask_question():
    # Check if a PDF file was uploaded
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    pdf_file = request.files['pdf']

    # Read the PDF file and extract text
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Get the user's question from the request
    user_question = request.form.get('question')

    # Perform similarity search
    docs = knowledge_base.similarity_search(user_question)

    # Initialize OpenAI language model
    llm = OpenAI(api_key=OPENAI_API_KEY)

    # Load question-answering model
    chain = load_qa_chain(llm, chain_type="stuff")

    # Get the answer to the user's question
    response = chain.run(input_documents=docs, question=user_question)

    return jsonify({'answer': response}), 200

if __name__ == '__main__':
    app.run(debug=True)
