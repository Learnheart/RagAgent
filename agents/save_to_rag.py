from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from ..data_processing import chunk_text
from groq import Groq
from langchain_groq import ChatGroq
from langchain.schema import Document
import os

embed_model = SentenceTransformer("hiieu/halong_embedding")
groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name='Llama3-8b-8192', api_key=groq_api_key)

sentences = [
    "Bóng đá có lợi ích gì cho sức khỏe?",
    "Bóng đá giúp cải thiện sức khỏe tim mạch và tăng cường sức bền.",
    "Bóng đá là môn thể thao phổ biến nhất thế giới.",
    "Bóng đá có thể giúp bạn kết nối với nhiều người hơn."
]
embeddings = embed_model.encode(sentences)

similarities = embed_model.similarity(embeddings, embeddings)
print(similarities.shape)

