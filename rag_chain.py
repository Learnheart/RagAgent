from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from data_processing import text_splitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

def chunk_embedding(text):
    # text = text_splitter('data/news.json')
    chunk_embed = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_storage = Chroma.from_texts(texts=text, embedding=chunk_embed)

    retriever = vector_storage.as_retriever()
    return retriever

def call_llm(model='gpt-3.5-turbo'):
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. You have to answer in Vietnamese language. \nQuestion: {question} \nContext: {context} \nAnswer:"
    )
    return llm, prompt

def main():
    text = text_splitter('data')
    retriever = chunk_embedding(text)
    llm, prompt = call_llm(model='gpt-3.5-turbo')

    def retrieve_and_format(question):
        docs = retriever.invoke(question)
        context = "\n\n".join(
            [doc.page_content for doc in docs])
        return {"context": context, "question": question}

    rag_chain = RunnablePassthrough() | retrieve_and_format | prompt | llm

    question = "một n"
    answer = rag_chain.invoke(question)
    print(answer.content)

if __name__ == "__main__":
    main()




