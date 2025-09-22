# to store data and do retrieval that is select till top k relevant documents
# here we use langchain vectorstore and FAISS as vector database
# langchain makes our life easy

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import List

def create_faiss_index(texts: List[str]) :
    """
    Create a FAISS index from the provided texts using the specified embedding model.

    Args:
        texts (List[str]): List of texts to be indexed.
        embedding_model_name (str): Name of the HuggingFace embedding model to use.

    Returns:
        FAISS: A FAISS vector store containing the indexed texts.
    """
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create the FAISS index
    return FAISS.from_texts(texts, embeddings)

def retrieve_relevant_docs(vector_store: FAISS, query: str, k: int = 4):
    """
    Retrieve the top k relevant documents from the FAISS vector store based on the query.

    Args:
        vector_store (FAISS): The FAISS vector store to search.
        query (str): The query string to search for.
        k (int): The number of top relevant documents to retrieve.

    Returns:
        List[str]: A list of the top k relevant documents.
    """
    # Perform similarity search
    return vector_store.similarity_search(query, k=k)
    
    
