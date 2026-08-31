from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import MODEL_PATH, MODEL_KWARDS, ENCODE_KWARDS


def create_vector_store(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,     
        model_kwargs=MODEL_KWARDS, 
        encode_kwargs=ENCODE_KWARDS
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store