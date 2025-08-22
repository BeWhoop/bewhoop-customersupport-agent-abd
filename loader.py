from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from db import supabase_client


# No need to call this function once data is uploaded as it uploads kb data
def store_documents():
    # Loading KB Docs from notion_export folder
    loader = NotionDirectoryLoader("notion_export/")
    loaded_docs = loader.load()

    # Chunking loaded docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 40,
        length_function=len
    )

    docs = text_splitter.split_documents(loaded_docs)

    # Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", 
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # Store documents with embeddings
    storing_doc = SupabaseVectorStore.from_documents(
        docs,
        embedding_model,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents",
        chunk_size=500  # Number of documents to insert at once
    )
    
    if storing_doc:
        print("Data Successfully Uploaded")
    
    return storing_doc


if __name__ == "__main__":
    store_documents()