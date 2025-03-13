from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", cache_dir="./embedding_cache" )

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(script_dir, "../real_estate_db/vectorstore")
persist_directory = os.path.normpath(persist_directory)

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embed_model, collection_name="vectorstore")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever_lambda = RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"]))

def retrieval_grader(question, context):
    retrieval_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination. \n
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
    retrieval_result = retrieval_grader.invoke({"question": question, "document": context})
    return retrieval_result

# test
question = "“Người sử dụng đất” được hiểu như thế nào theo quy định của Luật Đất đai năm 2024?"
docs = retriever.invoke(question)
print(f"Retrieved {len(docs)} documents.")
print(f"Using vector database from: {persist_directory}")

for i, doc in enumerate(docs):
    doc_txt = doc.page_content
    print(f"Document {i + 1} Content:\n{doc_txt}")
    grading_result = retrieval_grader(question, doc_txt)
    print(f"Document {i + 1} Grading Result: {grading_result}\n")

