# Hotel Reviews RAG

## Project Overview
The project uses RAG (Retrieval-Augmented Generation) technology to generate accurate and relevant responses based on hotel reviews. The system combines vector database retrieval with response generation via an LLM, relying on context from guest reviews.

## Data
Dataset that used in the project: https://huggingface.co/datasets/coeuslearning/hotel_reviews

## Technologies

| Компонент       | Технологія                     |
|-----------------|----------------------------------|
| Orchestration   | LangChain                        |
| LLM             | Google Gemini (gemini-3.5-flash) |
| Embeddings      | HuggingFace sentence-transformers|
| Vector Store    | FAISS                            |
| Data            | HuggingFace Datasets             |


## Project structure

List and description of the main files:

///
main.py                     # entry point: runs the RAG pipeline with a test query
|
notebooks/
├── eda.py                  # testing parameters
|
src/
├── data_acquisition.py     # uploading data + chunking
├── vector_store.py         # FAISS index
├── llm.py                  # Gemini
├── chain.py                # creating RAG pipeline
└── config.py               # setting 
///

## Results

- The system correctly finds relevant reviews for specific queries;
- With vague or generalized questions, the quality of the answer decreases;
- 'chunk_size=500' showed better results than 'chunk_size=300', because smaller chunks often broke the train of thought mid-sentence.

