import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain_community.document_loaders.csv_loader import CSVLoader

# Set up the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCp9dlPhLqOphZ2dhiNEZ5qwUrGbr2nm1E"

# Define file path for the FAISS index
db_file_path = "FAISS_Index"

# Initialize embeddings and language model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Function to create a vector database from a data loader
def create_vector_db(loader):
    data = loader.load()
    db = FAISS.from_documents(data, embeddings)
    db.save_local(db_file_path)

# Function to create an FAQ retrieval chain
def create_faq_chain():
    db = FAISS.load_local(db_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(score_threshold=0.7)

    # Define prompt template
    prompt_template = PromptTemplate(
        template="""Given the following context and a question, generate an answer based on this context only.\n
        In the answer, provide as much text as possible from the 'response' section in the source document context without making many changes.\n
        If the answer is not found in the context, kindly state 'This question is not present in my database.' Do not make up an answer.\n
        CONTEXT: {context}\nQUESTION: {question}""",
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="Question",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return chain

# Function to load a CSV file
def load_csv(file_path):
    return CSVLoader(file_path=file_path)

# Load CSV data and create vector database
csv_file_path = "Ecomerese_data.csv"
loader = load_csv(csv_file_path)
create_vector_db(loader)
faq_chain = create_faq_chain()

# Streamlit UI setup
st.title("E-commerce FAQ Retrieval System🛒")
st.markdown(
    """
    <style>
        #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    This model is designed specifically for E-commerce purposes. It is trained using 200 Question-Answer pairs. 
    Asking questions beyond this scope might lead to inaccurate answers. 

    You can view all 200 questions [here](https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset) and ask similar or combined questions such as:

    - **"How can I create an account?"**
    - **"What payment methods do you accept?"**
    - **"How can I create an account and what payment methods do you accept?"**
    """,
    unsafe_allow_html=True
)

query = st.text_input("Enter your question:")
if st.button("🔮 Get Answer"):
    with st.spinner("🔄 Processing..."):
        result = faq_chain(query)
        if "result" in result:
            st.write(result["result"])
        else:
            st.write("Answer not found.")
