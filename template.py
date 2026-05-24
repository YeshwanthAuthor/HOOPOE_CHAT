import os
import pathlib

def create_project_structure():
    """Create the complete project folder structure with empty files"""
    
    # Define the folder structure
    structure = {
        "config": [
            "config.json",
            ".env"
        ],
        "data": [
            "uploaded_docs/",
            "sample_docs/"
        ],
        "vectorstores": [
            "faiss_index/",
            "pinecone_backup/"
        ],
        "modules": [
            "__init__.py",
            "loader.py",
            "splitter.py",
            "embedder.py",
            "retriever.py",
            "rag_chain.py",
            "memory.py",
            "utils.py"
        ],
        "app": [
            "streamlit_app.py",
            "gradio_app.py"
        ],
        "notebooks": [
            "RAG_Debugging_Notebook.ipynb"
        ],
        "": [  # Root directory files
            "requirements.txt",
            "README.md",
            "run_app.bat"
        ]
    }
    
    # Create all folders and files
    for folder, items in structure.items():
        # Create folder if it doesn't exist
        if folder:
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {folder}/")
        
        # Create files within the folder
        for item in items:
            file_path = os.path.join(folder, item) if folder else item
            
            # Handle directories (ending with /)
            if item.endswith('/'):
                os.makedirs(file_path, exist_ok=True)
                print(f"Created folder: {file_path}")
            else:
                # Create empty file
                pathlib.Path(file_path).touch()
                print(f"Created file: {file_path}")
    
    print("\nProject structure created successfully!")
    print("All files and folders are now ready.")

if __name__ == "__main__":
    create_project_structure()