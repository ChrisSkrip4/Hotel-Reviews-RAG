from src.chain import rag_pipeline
from src.config import DATASET_NAME, PAGE_CONTENT_COLUMN


def main():

    qa_chain = rag_pipeline(DATASET_NAME, PAGE_CONTENT_COLUMN)

    query = "Where are the small rooms?"
    result = qa_chain.invoke({"query": query})

    print("Answer:", result["result"])
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(f"- {doc.page_content}")


if __name__ == "__main__":
    main()