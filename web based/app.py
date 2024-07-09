import os
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import load_prompt
from langchain.schema import BaseOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()

# Define a custom output parser class
class SimpleStrOutputParser(BaseOutputParser):
    def parse(self, output: str) -> str:
        return output

# Initialize Streamlit app
st.title("RAG-based Q&A System with Memory")

# Set up the LLM and other components
@st.cache_resource
def initialize_components():
    llm = GoogleGenerativeAI(model="gemini-pro")
    loader = WebBaseLoader("https://jalammar.github.io/illustrated-transformer/")
    docs = loader.load()
    
    if not docs or not docs[0].page_content:
        raise ValueError("Failed to load documents or document content is empty.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    if not splits:
        raise ValueError("Document splitting resulted in an empty list.")
    
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever()
    
    return llm, retriever

llm, retriever = initialize_components()

# Set up the RAG chain
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | SimpleStrOutputParser()
)

# Set up the question-answering chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Initialize ConversationBufferWindowMemory
conversation_memory = ConversationBufferWindowMemory(k=2)

# Function to handle user input and conversation
def handle_conversation(user_input):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(user_input)
    
    # Get the response from the QA chain
    response = qa_chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
    
    # Update conversation memory with user input and AI response
    conversation_memory.save_context({"input": user_input}, {"output": response["output_text"]})
    
    # Update chat history
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        response["output_text"]
    ])
    
    return response["output_text"]

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Ask a question:", on_change=lambda: st.session_state["submitted"] = True, key="input")

if st.button("Submit") or st.session_state.get("submitted"):
    st.session_state["submitted"] = False
    if user_input:
        # Handle the conversation and get the response
        response = handle_conversation(user_input)
        
        # Display the response
        st.write("Response:", response)
        st.session_state.input = ""

# Chat history button
if st.button("Show Chat History"):
    st.write("Chat History:")
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write(f"Human: {message.content}")
        else:
            st.write(f"AI: {message}")




