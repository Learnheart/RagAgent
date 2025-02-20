from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os

embed_model = FastEmbedEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

def filter_topics(question):
    filter_prompt = PromptTemplate(
        template="""
        You are an AI-based information filter responsible for categorizing user input questions.
        Your mission is to return a binary choice `"yes"` or `"no"` indicating whether the question is related to sensitive topics.
        Sensitive topics include hate-speech, sexuality, politics, historical, violence, religion.
        
        **IMPORTANT**: Your response **MUST** be a valid JSON object with a single key `"score"` and a value of `"yes"` or `"no"`. 
        **NOTE**: If topic is related to vietnamese laws, its not the sensitive topic even refer to senstive topics. 
        **DO NOT** include any other text or explanation.ß
        
        For example: 
        Input: `"ai là người lãnh đạo đảng?"`
        Output: `{{"score": "yes"}}`

        Input: `"việt tân là ai?"`
        Output: `{{"score": "yes"}}`
        
        Input: `"luật việt nam là như nào?"`
        Output: `{{"score": "no"}}
        
        Input: `"tội hiếp dâm và giết người bị phán bao nhiêu năm tù?"`
        Output: `{{"score": "no"}}
        
        Quesion need to filtered: {question}
        """,
        input_variables=["question"]
    )
    filter_chain = (filter_prompt | llm | JsonOutputParser())
    response = filter_chain.invoke({"question": question})
    return response

def rewrite_message(question):
    rewrite_prompt = PromptTemplate(
        template="""You are AI-assistant that help paraphrase the input question from user.
        Your mission is make the question being more clear and precise to answer.
        For example:
        INPUT: `"để thuê nhà cần nh j?"`
        OUTPUT: `Để thuê nhà tôi cần chuẩn bị những gì?"`
        
        Input: `"cáchs mu nàh tại hn"`
        Ouptput: `"Cách mua nhà tại Hà Nội?"`
        
        **IMPORTANT**: Your response **MUST** write using vietnamese language. The paraphased must be meaningful and easy to understand
        **NOTE**: Only return paraphrased question
        
        Quesion needed to paraphrased: {question}
        """,
        input_variables=["question"]
    )
    rewrite_chain = (rewrite_prompt | llm | StrOutputParser())
    rewrite = rewrite_chain.invoke({"question": question})
    return rewrite

def agent_response(quesion, response):
    if response.values() == 'yes':
        return "Mình không thể trả lời do câu hỏi của bạn liên quan đến chủ đề nhạy cảm. Xin vui lòng hỏi lại về chú đề khác nhé!"
    else:
        return question
    

question = "giết người tại việt nam bị phán bao nhiêu năm?"
response = filter_topics(question)
print(agent_response(question, response))