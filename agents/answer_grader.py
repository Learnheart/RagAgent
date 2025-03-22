from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

def answer_grader(question, response):
    answer_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is useful to resolve a question. Give a binary score "yes" or "no" to indicate whether the answer is useful to resolve a question. Provide the binary score as a JSON with a single key "score" and no preamble or explanation.
        
        Here are some sample cases:
        Example 1:
        question: Có bao nhiêu loại đất?"
        answer: "Các loại đất nông nghiệp bao gồm: Đất trồng cây hằng năm, gồm đất trồng lúa và đất trồng cây hằng năm khác; Đất trồng cây lâu năm; Đất lâm nghiệp, gồm đất rừng đặc dụng, đất rừng phòng hộ, đất rừng sản xuất; Đất nuôi trồng thủy sản; Đất chăn nuôi tập trung; Đất làm muối và các loại đất công nghiệp khác"
        output: {{"score": "yes"}}
        
        Example 2:
        question: Có bao nhiêu loại đất?"
        answer: "Có rất nhiều loại đất tại Việt Nam"
        output: {{"score": "no"}}
        
        Example 3: 
        question: "Tranh chấp đất đai được giải quyết theo trình tự nào?"
        answer: "Tranh chấp đất đai chỉ cần hai bên tự thỏa thuận với nhau."
        output: {{"score": "no"}}
        
        Example: 4
        document: "Tranh chấp đất đai được giải quyết theo trình tự nào?"
        answer: "Theo Điều 202, 203 Luật Đất đai 2013, tranh chấp đất đai trước hết được hòa giải tại UBND cấp xã. Nếu không thành, có thể khởi kiện tại Tòa án hoặc yêu cầu UBND cấp có thẩm quyền giải quyết."
        output: {{"score": "yes"}}
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    grader = answer_grader_prompt | llm | JsonOutputParser()
    grader_result = grader.invoke({"question": question,"generation": response})
    return grader_result

def test_answer_grader(file, output):
    data = pd.read_excel(file)
    grader_list = []
    for sample in range(len(data)):
        try:
            grader_result = answer_grader(data['question'][sample], data['answer'][sample])
            print("----------------------------------------------")
            print(f"question {data['question'][sample]} \nanswer: {data['answer'][sample]}")
            print(f"grade: {grader_result['score']}")
            print("----------------------------------------------")
            grader_list.append(grader_result['score'])
        except Exception as e:
            print(f"Error {e}")
            grader_list.append("error result")

    data['grader_answer'] = grader_list
    data.to_excel(output, index=False)
    print(f"save in {output}")
    return grader_list

def classification_report(file):
    data = pd.read_excel(file)
    valid_row = data[data['grader_answer'] != 'error result']
    y_true = valid_row['label']
    y_pred = valid_row['grader_answer']

    label_list = ["yes", "no"]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=label_list, average="binary", pos_label="yes")
    recall = recall_score(y_true, y_pred, labels=label_list, average="binary", pos_label="yes")
    f1 = f1_score(y_true, y_pred, labels=label_list, average="binary", pos_label="yes")
    return accuracy, precision, recall, f1


file = "../data/benchmark/test_for_answer_grader.xlsx"
output = "../data/test_output/answer_grader_fewshot.xlsx"
test_answer_grader(file, output)

accuracy, precision, recall, f1 = classification_report("../data/test_output/answer_grader_fewshot.xlsx")
print(f"accuracy {accuracy}")
print(f"precision {precision}")
print(f"recall {recall}")
print(f"f1 score {f1}")

