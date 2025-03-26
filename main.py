# main.py
from rag import index_document, query_rag
import os


PDF_PATH = os.path.join("sample_data", "content.pdf")
def main():
    print("Welcome to the Black's Law Dictionary RAG System!")
    print(f"Using dictionary at: {PDF_PATH}")
    print("1. Index the dictionary")
    print("2. Ask a question")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            try:
                index_document(PDF_PATH)
                print("Dictionary indexed successfully!")
            except Exception as e:
                print(f"Error indexing dictionary: {e}")
        
        elif choice == "2":
            question = input("Enter your legal question: ")
            answer = query_rag(question)
            print(f"\nAnswer: {answer}")
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()