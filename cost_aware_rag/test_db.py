import os
from database import ingest_pdf, query_database 

pdf_path = "data/medical_data.pdf"

if __name__ == "__main__":
    if os.path.exists(pdf_path):
        print("Starting PDF Ingestion...")
        ingest_pdf(pdf_path)

        test_query = "What is the guideline for coding COVID-19?"
        print(f"\nTesting query: {test_query}")
        
        results = query_database(test_query, n_results=3)
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Page: {res['page_num']}")
            print(f"Score: {res['score']}")
            print(f"Text Preview: {res['text'][:200]}...")
    else:
        print(f"PDF not found at: {pdf_path}")
        
