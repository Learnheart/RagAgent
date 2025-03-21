from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FastEmbedEmbeddings
import pandas as pd

embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", cache_dir="../embedding_cache" )

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(script_dir, "../real_estate_db/vectorstore")
persist_directory = os.path.normpath(persist_directory)

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embed_model, collection_name="vectorstore")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever_lambda = RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"]))

def answer_generator(question):
    qa_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just refuse answer in polite and friendly. 
        Answer question in detailed, make sure references vietnamese's law that prove for your answer. 
        For example:
        Question: Phạm vi điều chỉnh và đối tượng áp dụng Luật Đất đai năm 2024 là gì?
        Answer: 
        Điều 1. Phạm vi điều chỉnh
        Luật này quy định về chế độ sở hữu đất đai, quyền hạn và trách nhiệm của Nhà nước đại diện chủ sở hữu toàn dân về đất đai và thống nhất quản lý về đất đai, chế độ quản lý và sử dụng đất đai, quyền và nghĩa vụ của công dân, người sử dụng đất đối với đất đai thuộc lãnh thổ của nước Cộng hòa xã hội chủ nghĩa Việt Nam.
        Điều 2. Đối tượng áp dụng
        1. Cơ quan nhà nước thực hiện quyền hạn và trách nhiệm đại diện chủ sở hữu toàn dân về đất đai, thực hiện nhiệm vụ thống nhất quản lý nhà nước về đất đai.
        2. Người sử dụng đất.
        3. Các đối tượng khác có liên quan đến việc quản lý, sử dụng đất đai.'
        Answer in professional in vietnamese.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    rag_chain = (
        {"question": lambda x: x["question"], "context": retriever_lambda}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke({"question": question})

def generate_answer_for_test_set(file, output):
    data = pd.read_json(file)
    answer_list = []
    for sample in range(len(data['question'])):
        try:
            answer = answer_generator(data['question'][sample])
            print(f" answer for {data['question'][sample]}: {answer}")
            answer_list.append(answer)
        except Exception as e:
            print(f"Generator meet error {e}")
            answer_list.append("error")
    # test_for_hallu_and_answer = {"question": data['question'], "answer": answer_list}
    data['answer'] = pd.DataFrame(answer_list)
    data.to_excel(output, index=False)
    return data

data = pd.read_json("../data/test data/router_test_data.json")
output = "../data/test data/test_for_hallu_and_answer.xlsx"
input = data[0:2]
print(input)
generate_answer_for_test_set(input, output)



