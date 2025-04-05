from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

def check_hallucination(generation, documents):

    hallucination_grader_prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is **strictly grounded in and supported by** a set of referenced legal documents about **Vietnam Real Estate Law**. 
        If the answer accurately reflects information from the provided legal documents **without introducing unsupported details, fabricating legal articles, or misstating years**, grade it as `"yes"`. Otherwise, grade it as `"no"`.
        Provide the binary score as a JSON with a single key `"score"` and no preamble or explanation. 
        
        Example 1:
        document: "Điều 9. Phân loại đất\n1. Căn cứ vào mục đích sử dụng,\nđất đai được phân loại bao gồm nhóm đất nông nghiệp, nhóm đất phi nông nghiệp,\nnhóm đất chưa sử dụng. 2. Nhóm đất\nnông nghiệp bao gồm các loại đất sau đây: a) Đất trồng cây hằng năm, gồm đất trồng lúa và đất trồng cây hằng năm khác; b) Đất trồng cây lâu năm; c) Đất lâm nghiệp, gồm đất rừng\nđặc dụng, đất rừng phòng hộ, đất rừng sản xuất; d) Đất nuôi trồng thủy sản; đ) Đất chăn nuôi tập trung; e) Đất làm muối; g) Đất nông nghiệp khác. 3"
        answer: "Các loại đất nông nghiệp bao gồm: Đất trồng cây hằng năm, gồm đất trồng lúa và đất trồng cây hằng năm khác; Đất trồng cây lâu năm; Đất lâm nghiệp, gồm đất rừng đặc dụng, đất rừng phòng hộ, đất rừng sản xuất; Đất nuôi trồng thủy sản; Đất chăn nuôi tập trung; Đất làm muối và các loại đất công nghiệp khác"
        output: `{{"score": "yes"}}`
        
        Example 2:
        document: ". 2. Việc sử dụng đất kết\nhợp đa mục đích phải đáp ứng các yêu cầu sau đây: a) Không làm thay đổi loại đất theo phân loại đất quy định tại khoản 2 và khoản 3 điều 9 và đã được xác định tại các loại\ngiấy tờ quy định tại điều 10 của Luật này ; b) Không làm mất đi điều kiện cần thiết để trở\nlại sử dụng đất vào mục đích chính; c) Không ảnh hưởng đến quốc phòng, an ninh; d) Hạn chế ảnh hưởng đến bảo tồn hệ sinh thái\ntự nhiên, đa dạng sinh học, cảnh quan môi trường; đ) Không làm ảnh hưởng đến việc sử dụng đất của\ncác thửa đất liền kề; e) Thực hiện đầy đủ nghĩa vụ tài chính theo\nquy định; g) Tuân thủ pháp luật có liên quan. 3. Trường hợp đ ơn vị sự nghiệp công lập sử dụng đất xây dựng công trình sự\nnghiệp kết hợp với thương mại, dịch vụ thì phải chuyển sang thuê đất trả tiền thuê đất\nhằng năm đối với phần diện tích kết hợp đó. 4"
        answer: "Các loại đất nông nghiệp bao gồm: Đất trồng cây hằng năm, gồm đất trồng lúa và đất trồng cây hằng năm khác; Đất trồng cây lâu năm; Đất lâm nghiệp, gồm đất rừng đặc dụng, đất rừng phòng hộ, đất rừng sản xuất; Đất nuôi trồng thủy sản; Đất chăn nuôi tập trung; Đất làm muối và các loại đất công nghiệp khác"
        output: `{{"score": "no"}}`
        
        Example 3: 
        document: "Điều 5. Các\nloại bất động sản, dự án bất động sản đưa vào kinh doanh\n1. Nhà ở có sẵn và nhà ở hình\nthành trong tương lai. 2. Công\ntrình xây dựng có sẵn, công trình xây dựng hình thành trong tương lai, bao gồm\ncông trình xây dựng có công năng phục vụ mục đích giáo dục, y tế, thể thao, văn\nhóa, văn phòng, thương mại, dịch vụ, du lịch, lưu trú, công nghiệp và công\ntrình xây dựng có công năng phục vụ hỗn hợp. 3. Phần diện tích sàn xây dựng\ntrong công trình xây dựng theo quy định tại khoản 2 Điều này. 4. Quyền sử dụng đất đã có hạ tầng\nkỹ thuật trong dự án bất động sản. 5. Dự án bất động sản. 6. Chính phủ\nquy định chi tiết khoản 2 và khoản 3 Điều này."
        answer: "Điều 5 Luật Đất đai 2024 quy định các nguyên tắc sử dụng đất như sau:
            1. Đúng mục đích sử dụng đất.
            2. Bền vững, tiết kiệm, có hiệu quả đối với đất đai và tài nguyên trên bề mặt, trong lòng đất.
            3. Bảo vệ đất, bảo vệ môi trường, thích ứng với biến đổi khí hậu, không được lạm dụng thuốc bảo vệ thực vật, phân hóa học làm ô nhiễm, thoái hóa đất.
            4. Thực hiện quyền và nghĩa vụ của người sử dụng đất trong thời hạn sử dụng đất theo quy định của Luật này và quy định khác của pháp luật có liên quan; không xâm phạm quyền, lợi ích hợp pháp của người sử dụng đất liền kề và xung quanh."
        output: `{{"score": "no"}}`
        
        Example: 4
        document: "Điều 5.\nNguyên tắc sử dụng đất\n1. Đúng mục đích sử dụng đất. 2. Bền vững, tiết kiệm, có hiệu\nquả đối với đất đai và tài nguyên trên bề mặt, trong lòng đất. 3. Bảo vệ đất, bảo vệ môi trường,\nthích ứng với biến đổi khí hậu, không được lạm dụng thuốc bảo vệ thực vật, phân\nhóa học làm ô nhiễm, thoái hóa đất. 4. Thực hiện quyền và nghĩa vụ của\nngười sử dụng đất trong thời hạn sử dụng đất theo quy định của Luật này và quy\nđịnh khác của pháp luật có liên quan; không xâm phạm quyền, lợi ích hợp pháp của\nngười sử dụng đất liền kề và xung quanh."
        answer: "Điều 5 Luật Đất đai 2024 quy định các nguyên tắc sử dụng đất như sau:
            1. Đúng mục đích sử dụng đất.
            2. Bền vững, tiết kiệm, có hiệu quả đối với đất đai và tài nguyên trên bề mặt, trong lòng đất.
            3. Bảo vệ đất, bảo vệ môi trường, thích ứng với biến đổi khí hậu, không được lạm dụng thuốc bảo vệ thực vật, phân hóa học làm ô nhiễm, thoái hóa đất.
            4. Thực hiện quyền và nghĩa vụ của người sử dụng đất trong thời hạn sử dụng đất theo quy định của Luật này và quy định khác của pháp luật có liên quan; không xâm phạm quyền, lợi ích hợp pháp của người sử dụng đất liền kề và xung quanh."
        output: `{{"score": "yes"}}`
     
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

    return hallucination_grader.invoke({"generation": generation, "documents": documents})

def hallucination_testing(file, output):
    data = pd.read_excel(file)
    hallu_list = []
    for sample in range(len(data)):
        try:
            hallu_result = check_hallucination(data['answer'][sample], data['document_preview'][sample])
            print("---------------------------------------------------------------")
            print(f"retrieve {data['document_preview'][sample]} & answer {data['answer'][sample]}")
            print(f"\nhallu check: {hallu_result['score']}")
            print("---------------------------------------------------------------")
            hallu_list.append(hallu_result['score'])
        except Exception as e:
            print(f"Error {e}")
            hallu_list.append("error result")

    data['hallu_score'] = hallu_list
    data.to_csv(output, index=False)
    print(f"successfully saved file in {output}")
    return hallu_list


def classification_report(file):
    data = pd.read_excel(file)
    valid_row = data[data['hallu_score'] != 'error result']
    y_true = valid_row['label']
    y_pred = valid_row['hallu_score']

    label_list = ["yes", "no"]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=label_list, average="binary", pos_label="yes")
    recall = recall_score(y_true, y_pred, labels=label_list, average="binary", pos_label="yes")
    f1 = f1_score(y_true, y_pred, labels=label_list, average="binary", pos_label="yes")
    return accuracy, precision, recall, f1

# start = time.time()
# file = "../data/benchmark/test_for_hallucination_labelled.xlsx"
# output = "../data/test_output/hallucination_fewshot.csv"
# hallucination_testing(file, output)

# end = time.time()
# print(f"time for hallu: {end - start}")

data = pd.read_csv("../data/test_output/hallucination_fewshot.csv")
print(data[data['hallu_score'] == "error result"])
# accuracy, precision, recall, f1 = classification_report(data)
# print(f"accuracy {accuracy}")
# print(f"precision {precision}")
# print(f"recall {recall}")
# print(f"f1 score {f1}")

# time fewshot: 1564.1549611091614



