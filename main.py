import os
import streamlit as st
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
import nltk

nltk.download("punkt")

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Sidebar input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# Initialize the ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    api_key="you_api_key",
    model_name="llama-3.1-70b-versatile",
)

if process_url_clicked:
    # Ensure URLs are not empty
    urls = [url for url in urls if url.strip()]
    if not urls:
        st.error("Please enter at least one valid URL.")
        st.stop()

    main_placeholder.text("Data Loading...Started...✅✅✅")

    try:
        # Synchronous data loading using UnstructuredURLLoader
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()  # Use the synchronous `load` method

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=250,
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("No valid text chunks could be created. Ensure the URLs have enough content.")
            st.stop()

        # Create embeddings and FAISS index
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    except Exception as e:
        st.error(f"Error processing URLs: {e}")
        st.stop()

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
