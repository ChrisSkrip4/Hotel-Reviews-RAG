from langchain_classic.chains import RetrievalQA
from src.vector_store import create_vector_store
from src.data_acquisition import prepare_data
from src.llm import llm
from src.config import TOP_K



def rag_pipeline(dataset_name, page_content_column):

    documents = prepare_data(dataset_name, page_content_column)
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,  
        chain_type="stuff",            
    )

    return qa_chain
