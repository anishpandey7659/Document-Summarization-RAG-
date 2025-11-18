# Document-Summarization-RAG-

# Also Due to technical issues i cant add rag_logic.py file so add by yourself

# rag_logic.py

    # 1. Imports needed for RAG logic
    
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_chroma import Chroma
    from langchain_groq import ChatGroq
    import uuid
    import os

    API_KEY = os.getenv("GROQ_API_KEY")
    # Initialize the LLM model (can be outside the function as it doesn't depend on the file)
    model = ChatGroq(
    groq_api_key=API_KEY, # Replace with your actual API key
    model="llama-3.1-8b-instant"
    )
    # Initialize the embeddings model (can be outside the function)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #1
    def setup_and_invoke_rag_chain(pdf_path: str, user_question: str):
    """
    Sets up the RAG chain for a specific document and invokes it with a question.
    (Your existing RAG logic, using the temporary pdf_path)
    """
    # [RAG logic function content remains the same]
    # 1. Load PDF from the temporary path
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    # 2. Split the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    docs = splitter.split_documents(document)

    # 🚨 FIX: Use a unique collection name every time 🚨
    unique_collection_name = f"doc_collection_{uuid.uuid4()}"
    # 3. Create Chroma vector store in memory
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=unique_collection_name  # Use the unique identifier
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}, search_type="mmr")

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 4. Define the RAG chain components
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    
    # Define the Prompt Template
    prompt = PromptTemplate(
        template="""
        You are an expert summarizer.
        Your job is to create a clear, detailed, and complete summary from the given context.
        Do NOT add outside information.
        Do NOT change facts
        If the context is insufficient, just say I don't know.

        Context:
        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )
    
    rag_chain = parallel_chain | prompt | model | parser
    
    result = rag_chain.invoke(user_question)
    return result


    if __name__ == '__main__':
    # This block allows you to test the logic independently
    # Example: print(setup_and_invoke_rag_chain("path/to/test.pdf", "What is the key finding?"))
    pass


