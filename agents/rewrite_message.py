from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os

embed_model = FastEmbedEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

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

# test
question = "nàm sao để mua nhà tại hn"
response = rewrite_message(question)
print(response)