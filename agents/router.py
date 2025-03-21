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


def router_question_zero_shot(question):
    """
    Phiên bản zero-shot của router (không có ví dụ)
    """
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


def router_question_few_shot(question):
    """
    Phiên bản few-shot của router (có các ví dụ minh họa)
    """
    router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on real estate laws in Vietnam. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search.

        Here are some examples:

        Question: "Thời tiết Hà Nội hôm nay thế nào?"
        Datasource: web_search

        Question: "Quyền sử dụng đất là gì theo luật đất đai?"
        Datasource: vectorstore

        Question: "Cách nấu phở bò tái ngon không bị tanh?"
        Datasource: web_search

        Question: "Tôi có thể chuyển đổi đất nông nghiệp sang đất ở được không?"
        Datasource: vectorstore

        Question: "Thủ tục sang tên sổ đỏ sau khi mua nhà?"
        Datasource: vectorstore

        Question: "Giá vàng hôm nay là bao nhiêu?"
        Datasource: web_search

        Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    question_router = router_prompt | llm | JsonOutputParser()
    router_result = question_router.invoke(question)
    return router_result


def test_with_dataset(dataset_path, mode='both'):
    """
    Kiểm tra router với dataset

    Args:
        dataset_path: Đường dẫn đến file dataset JSON
        mode: 'zero_shot', 'few_shot', hoặc 'both'
    """
    # Đọc dataset từ file JSON
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    zero_shot_results = []
    few_shot_results = []

    # Kiểm tra zero-shot nếu mode là 'zero_shot' hoặc 'both'
    if mode in ['zero_shot', 'both']:
        print("\nTesting with zero-shot learning...")
        for item in tqdm(dataset, desc="Testing zero-shot router"):
            question = item["question"]
            expected_label = item["label"]

            try:
                result = router_question_zero_shot(question)
                predicted_label = result.get("datasource")

                zero_shot_results.append({
                    "question": question,
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "correct": expected_label == predicted_label
                })
            except Exception as e:
                zero_shot_results.append({
                    "question": question,
                    "expected": expected_label,
                    "predicted": "error",
                    "correct": False,
                    "error": str(e)
                })

    # Kiểm tra few-shot nếu mode là 'few_shot' hoặc 'both'
    if mode in ['few_shot', 'both']:
        print("\nTesting with few-shot learning...")
        for item in tqdm(dataset, desc="Testing few-shot router"):
            question = item["question"]
            expected_label = item["label"]

            try:
                result = router_question_few_shot(question)
                predicted_label = result.get("datasource")

                few_shot_results.append({
                    "question": question,
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "correct": expected_label == predicted_label
                })
            except Exception as e:
                few_shot_results.append({
                    "question": question,
                    "expected": expected_label,
                    "predicted": "error",
                    "correct": False,
                    "error": str(e)
                })

    # Phân tích kết quả zero-shot
    if mode in ['zero_shot', 'both'] and zero_shot_results:
        zero_shot_df = pd.DataFrame(zero_shot_results)
        zero_shot_accuracy = zero_shot_df["correct"].mean() * 100

        print("\n=== Zero-Shot Results ===")
        print(f"Accuracy: {zero_shot_accuracy:.2f}%")
        print(f"Total questions: {len(zero_shot_df)}")
        print(f"Correct predictions: {zero_shot_df['correct'].sum()}")

        # Confusion matrix
        zero_shot_confusion = pd.crosstab(
            zero_shot_df["expected"],
            zero_shot_df["predicted"],
            rownames=["Expected"],
            colnames=["Predicted"]
        )
        print("\nZero-Shot Confusion Matrix:")
        print(zero_shot_confusion)

        # Lưu kết quả
        zero_shot_df.to_excel("../data/test data/router_test_zero_shot_results.xlsx", index=False)

    # Phân tích kết quả few-shot
    if mode in ['few_shot', 'both'] and few_shot_results:
        few_shot_df = pd.DataFrame(few_shot_results)
        few_shot_accuracy = few_shot_df["correct"].mean() * 100

        print("\n=== Few-Shot Results ===")
        print(f"Accuracy: {few_shot_accuracy:.2f}%")
        print(f"Total questions: {len(few_shot_df)}")
        print(f"Correct predictions: {few_shot_df['correct'].sum()}")

        # Confusion matrix
        few_shot_confusion = pd.crosstab(
            few_shot_df["expected"],
            few_shot_df["predicted"],
            rownames=["Expected"],
            colnames=["Predicted"]
        )
        print("\nFew-Shot Confusion Matrix:")
        print(few_shot_confusion)

        # Lưu kết quả
        few_shot_df.to_excel("../data/test data/router_test_few_shot_results.xlsx", index=False)

    if mode == 'both' and zero_shot_results and few_shot_results:
        comparison_data = []
        for i in range(len(dataset)):
            if i < len(zero_shot_results) and i < len(few_shot_results):
                zero_shot = zero_shot_results[i]
                few_shot = few_shot_results[i]
                if zero_shot["question"] == few_shot["question"]:
                    comparison_data.append({
                        "question": zero_shot["question"],
                        "expected": zero_shot["expected"],
                        "zero_shot_predicted": zero_shot["predicted"],
                        "zero_shot_correct": zero_shot["correct"],
                        "few_shot_predicted": few_shot["predicted"],
                        "few_shot_correct": few_shot["correct"],
                        "agreement": zero_shot["predicted"] == few_shot["predicted"]
                    })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            agreement_rate = comparison_df["agreement"].mean() * 100

            zero_better = comparison_df[
                (comparison_df["zero_shot_correct"] == True) & (comparison_df["few_shot_correct"] == False)]

            few_better = comparison_df[
                (comparison_df["zero_shot_correct"] == False) & (comparison_df["few_shot_correct"] == True)]

            print("\n=== Comparison Results ===")
            print(f"Agreement rate: {agreement_rate:.2f}%")
            print(f"Cases where zero-shot is correct but few-shot is wrong: {len(zero_better)}")
            print(f"Cases where few-shot is correct but zero-shot is wrong: {len(few_better)}")

            # Lưu kết quả so sánh
            comparison_df.to_csv("router_test_comparison.csv", index=False)

            if len(zero_better) > 0:
                print("\nTop examples where zero-shot is better:")
                for _, row in zero_better.head(5).iterrows():
                    print(f"Q: {row['question']}")
                    print(
                        f"Expected: {row['expected']}, Zero-shot: {row['zero_shot_predicted']}, Few-shot: {row['few_shot_predicted']}")
                    print()

            if len(few_better) > 0:
                print("\nTop examples where few-shot is better:")
                for _, row in few_better.head(5).iterrows():
                    print(f"Q: {row['question']}")
                    print(
                        f"Expected: {row['expected']}, Zero-shot: {row['zero_shot_predicted']}, Few-shot: {row['few_shot_predicted']}")
                    print()

    # Trả về kết quả dựa trên mode
    if mode == 'zero_shot' and zero_shot_results:
        return pd.DataFrame(zero_shot_results)
    elif mode == 'few_shot' and few_shot_results:
        return pd.DataFrame(few_shot_results)
    elif mode == 'both' and zero_shot_results and few_shot_results:
        return {
            'zero_shot': pd.DataFrame(zero_shot_results),
            'few_shot': pd.DataFrame(few_shot_results)
        }
    else:
        return None


def process():
    dataset_path = "../data/test data/router_test_data.json"

    results = test_with_dataset(dataset_path, mode='both')

    # results = test_with_dataset(dataset_path, mode='zero_shot')
    # results = test_with_dataset(dataset_path, mode='few_shot')


if __name__ == "__main__":
    process()
