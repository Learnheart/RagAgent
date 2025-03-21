from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
import json
import pandas as pd
import argparse

embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                  cache_dir="./embedding_cache")

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(script_dir, "../real_estate_db/vectorstore")
persist_directory = os.path.normpath(persist_directory)

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embed_model, collection_name="vectorstore")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# retriever = vectorstore.as_retriever(
#     search_type="mmr",  # Maximum Marginal Relevance - cân bằng giữa liên quan và đa dạng
#     search_kwargs={
#         "k": 5,  # Lấy nhiều tài liệu hơn để có độ phủ tốt hơn
#         "fetch_k": 20,  # Lấy 20 tài liệu gần nhất ban đầu
#         "lambda_mult": 0.7,  # Hệ số cân bằng giữa liên quan (0.7) và đa dạng (0.3)
#         "filter": None  # Có thể thêm filter để lọc kết quả theo metadata
#     }
# )

# retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 3,  # Số lượng tài liệu trả về
#             "fetch_k": 9,  # Lấy nhiều hơn để sau đó lọc
#             "lambda_mult": 0.7})

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


def check_keywords(text, keywords):
    found = []
    for kw in keywords:
        if kw.lower() in text.lower():
            found.append(kw)
    return found

def test_retrieval_with_dataset(test_data_path='C:\\VNUIS_Workspace\\RagAgent\\data\\retrieval_data.json', output_file='../data/data test/retrieval_test_result.xlsx'):
    """Kiểm thử với bộ dữ liệu và tạo file kết quả"""
    # Đọc dữ liệu kiểm thử
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_cases = test_data.get('test_cases', [])
    results = []

    for test_case in test_cases:
        question = test_case['question']
        test_id = test_case.get('id', 'unknown')
        expected_keywords = test_case.get('expected_keywords', [])

        print(f"\nTesting: {test_id} - {question}")
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents.")

        for i, doc in enumerate(docs):
            doc_txt = doc.page_content
            doc_id = f"{test_id}_doc{i + 1}"

            grading_result = retrieval_grader(question, doc_txt)
            is_relevant = grading_result.get('score', 'no').lower() == 'yes'

            found_keywords = check_keywords(doc_txt, expected_keywords)
            keyword_ratio = f"{len(found_keywords)}/{len(expected_keywords)}"

            results.append({
                'test_id': test_id,
                'question': question,
                'doc_id': doc_id,
                'llm_relevance': is_relevant,
                'keywords_found': keyword_ratio,
                'found_keywords': ', '.join(found_keywords),
                'document_preview': doc_txt[:300] + '...' if len(doc_txt) > 300 else doc_txt,
                'manual_relevance': '',
                'notes': ''
            })

            print(f"Document {i + 1}: {'Relevant' if is_relevant else 'Not relevant'} (Keywords: {keyword_ratio})")

    # Lưu kết quả vào file CSV
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"\nĐã lưu kết quả vào file {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieval Grader Test')
    parser.add_argument('--data', type=str, default='C:\\VNUIS_Workspace\\RagAgent\\data\\retrieval_data.json', help='File dữ liệu kiểm thử')
    parser.add_argument('--output', type=str, default='../data/data test/retrieval_test_result.xlsx', help='File kết quả đầu ra')
    parser.add_argument('--question', type=str, help='Kiểm thử với một câu hỏi cụ thể')

    args = parser.parse_args()

    if args.question:
        question = args.question
        print(f"\nTesting single question: {question}")
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents.")

        for i, doc in enumerate(docs):
            doc_txt = doc.page_content
            print(f"\nDocument {i + 1} Content:\n{doc_txt[:500]}...")
            grading_result = retrieval_grader(question, doc_txt)
            print(f"Document {i + 1} Grading Result: {grading_result}")
    else:
        test_retrieval_with_dataset(args.data, args.output)