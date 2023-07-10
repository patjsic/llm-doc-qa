import os
from data import DBStore

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

###Configure Environment Variables
os.environ["OPENAI_API_KEY"] = "<INSERT OPENAI ENDPOINT HERE>"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "<INSERT LANGCHAIN ENDPOINT HERE>"
os.environ["LANGCHAIN_API_KEY"] = '<INSERT LANGCHAIN API KEY HERE>'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(dirpath):
    """Main run logic
    """
    embedding = HuggingFaceEmbeddings()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301")
    db = DBStore(dirpath, embedding).vector_store

    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
    )

    retriever=db.as_retriever()

    # TODO: Build prompt for conversational QA
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    # qa = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=retriever,
    #     return_source_documents=True,
    #     # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    #     chain_type="stuff"
    # )

    while True:
        query = input("Input a question: ") #Always returns str
        results = qa({"question": query})
        print(results["answer"])
    return None

if __name__ == "__main__":
    main('./papers')