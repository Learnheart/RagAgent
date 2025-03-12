from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
import os

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

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

# test
docs = """

"""

response = """

"""
hallucination_grader_response = hallucination_grader.invoke({"documents": docs, "generation": response})
print(hallucination_grader_response)