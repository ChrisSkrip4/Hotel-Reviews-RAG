from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_data_from_huggingface(dataset_name, page_content_column): 

    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    data = loader.load()
    return data


def split_data(data):

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(data)

    return chunks


def prepare_data(dataset_name, page_content_column) :

    data = load_data_from_huggingface(dataset_name, page_content_column)
    chunks = split_data(data)

    return chunks

