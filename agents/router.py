import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
import pandas as pd
import json
from tqdm import tqdm

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

def router_question(question):
    router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on real estate laws in Vietnam. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    question_router = router_prompt | llm | JsonOutputParser()
    router_result = question_router.invoke(question)
    return router_result


def test_with_dataset(dataset_path):
    # Đọc dataset từ file JSON
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    results = []

    for item in tqdm(dataset, desc="Testing router"):
        question = item["question"]
        expected_label = item["label"]

        try:
            result = router_question(question)
            predicted_label = result.get("datasource")

            results.append({
                "question": question,
                "expected": expected_label,
                "predicted": predicted_label,
                "correct": expected_label == predicted_label
            })
        except Exception as e:
            results.append({
                "question": question,
                "expected": expected_label,
                "predicted": "error",
                "correct": False,
                "error": str(e)
            })

    results_df = pd.DataFrame(results)
    #Accuracy
    accuracy = results_df["correct"].mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nDetailed Statistics:")
    print(f"Total questions: {len(results_df)}")
    print(f"Correct predictions: {results_df['correct'].sum()}")

    # Confusion matrix
    confusion = pd.crosstab(
        results_df["expected"],
        results_df["predicted"],
        rownames=["Expected"],
        colnames=["Predicted"]
    )
    print("\nConfusion Matrix:")
    print(confusion)

    # Save the result
    results_df.to_csv("router_test_results.csv", index=False)
    return results_df

def process():
    dataset_path = "data/router_test_data.json"
    # Test router với dataset
    results = test_with_dataset(dataset_path)

process()
# test
# question = "Hôm nay là ngày bao nhiêu"
# print(router_question(question))