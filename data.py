import os
import json
from pathlib import Path

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader

###THIS FILE MANAGES DATA/TEXT DIRECTORY AND PREPROCESSING FOR EMBEDDING
###KEEPS TRACK OF METADATA TO AVOID EMBEDDING TEXT MULTIPLE TIMES


###CURRENTLY ONLY SUPPORTS: .txt, .pdf
###CLASS MUST BE INITIALIZED ONLY ONCE PER RUN, IF DOCUMENTS ARE UPDATED CLASS MUST BE RUN AGAIN
###SINCE IT CURRENTLY SAVES DOCUMENT METADATA AND EMBEDS ON INITIALIZATION

class DBStore:
    """ Handles all logic around directory metadata and file management.
    """
    def __init__(self, dirpath, embedding_function):
        """ Initialize variables.

        On initialization:
            1. Checks for metadata
            2. Loads filenames under given directory path
            3. Loads documents from filenames that are not already in metadata

        Args:
            dirpath: path to document directory
            loadpath: path to existing vectordb, if None then none exists
            embedding_function: class 
            loadpath: path to load existing ChromaDB
        """
        self.dirpath = dirpath
        self.embedding = embedding_function
        self.filenames = self.get_files()
        self.documents = []
        self.splits = []
        self.vector_store = None
        self.load_true = False

        #Load vectordb and set load_true if persistant database exists
        self.check_and_load_metadata()

        #Load documents
        self.load_documents()

        #Split documents
        self.split_docs()

        #Embed documents
        self.embed_documents()

    def check_and_load_metadata(self):
        #Check for existing directory
        if os.path.exists(os.path.join(self.dirpath, 'index')):
            self.vector_store = Chroma(persist_directory=self.dirpath)
            self.vector_store.load()
            self.load_true = True
        else:
            print("No persistant directory found.")
        
        #TODO: Check for existing file metadata to add to existing directory
        return None
    
    def get_files(self):
        """ Return list of all filenames in directory and ensure they are a file
        """
        return [f for f in os.listdir(self.dirpath) if os.path.isfile(os.path.join(self.dirpath,f))]
    
    def save_metadata(self):
        """Save filenames list as json to prevent embedding twice
        """
        return None
    
    def load_documents(self):
        """Load in documents to document attribute.
        """
        #TODO: Check for 
        # loader = TextLoader(self.filenames)
        # self.documents = loader.load()
        print("Loading documents...")
        for file in self.filenames:
            if ".txt" in file.lower():
                loader = TextLoader(os.path.join(self.dirpath, file))
            elif ".pdf" in file.lower():
                loader = PyPDFLoader(os.path.join(self.dirpath, file))
            #temp = loader.load()
            self.documents.extend(loader.load())
        
        print("Documents length: " + str(len(self.documents)))
        return None
    
    def split_docs(self, chunk_size=1500, chunk_overlap=150):
        """Split documents with recursive character text splitter.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # for doc in self.documents:
        #     chunks = splitter.split_text(doc.page_content)
        #     self.splits.extend(chunks)
        self.splits = splitter.split_documents(self.documents)
        return None

    def embed_documents(self):
        """Embeds documents given embedding option as argument.
        """
        if self.load_true:
            print("Adding embedding to existing vector store...")
            #Here load existing DB and add new documents
            #TODO: This is bad, it will reembed old documents as long as we are not checking in load_documents
            self.vector_store.add(self.documents, embedding_function=self.embedding)
        else:
            print("Embedding documents...")
            self.vector_store = Chroma.from_documents(documents=self.splits, 
                                                      persist_directory=self.dirpath, 
                                                      embedding = self.embedding)

        return None


