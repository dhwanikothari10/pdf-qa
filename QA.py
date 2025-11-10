import streamlit as st
import os
#from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --------------------- SETUP ---------------------
#load_dotenv()

GOOGLE_API_KEY=st.secrets['GOOGLE_API_KEY']
HUGGINGFACEHUB_API_KEY=st.secrets['HUGGINGFACEHUB_API_KEY']

st.set_page_config(page_title="📘 PDF Question Answering App", layout="wide")

st.title("📘 PDF Question Answering using Gemini + LangChain")
st.markdown(
    "Upload a PDF, then ask questions about its content. "
    "The app uses *embeddings + FAISS + Gemini model* to answer based on the PDF."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.0)
    top_k = st.slider("Retriever Top K", 1, 10, 4)

# --------------------- FILE UPLOAD ---------------------
uploaded_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        # Save uploaded file temporarily
        pdf_path = os.path.join("temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and split
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = splitter.split_documents(docs)

        # Embeddings + FAISS index
        emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vs = FAISS.from_documents(splits, emb)
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

        # Prompt + LLM setup
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)

        def format_docs(docs): 
            return "\n\n".join(d.page_content for d in docs)

        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        chain = parallel | prompt | llm | StrOutputParser()

        st.success("✅ PDF processed successfully!")

        # --------------------- Q&A ---------------------
        st.subheader("💬 Ask a question from the PDF")
        question = st.text_input("Enter your question here:")

        if question:
            with st.spinner("Generating answer..."):
                answer = chain.invoke(question.strip())

            st.markdown("### 🧠 Answer:")
            st.write(answer)
else:
    st.info("👆 Upload a PDF file to begin.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain, FAISS, HuggingFace & Gemini")