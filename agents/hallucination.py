from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
import os
import pandas as pd

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

def check_hallucination(generation, documents):

    hallucination_grader_prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
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
            print(f"\nhallu check: {hallu_result}")
            print("---------------------------------------------------------------")
            hallu_list.append(hallu_result)
        except Exception as e:
            print(f"Error {e}")
            hallu_list.append(e)

    df = pd.DataFrame(hallu_list)
    data['hallu_score'] = df['score']
    data.to_excel(output, index=False)
    print(f"successfully saved file in {output}")
    return hallu_list

file = "../data/benchmark/test_for_hallucination.xlsx"
output = "../data/test_output/hallucination_zeroshot.xlsx"
hallucination_testing(file, output)

