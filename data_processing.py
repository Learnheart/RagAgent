import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_json(file):
    with open(file, 'r') as file:
        articles = json.load(file)

    data = [article['content'] for article in articles]
    return data

def clean_text(file):
    text = file.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text 

def read_database(file) -> str:
    articles = read_json(file)
    clean_data = (clean_text(article) for article in articles)
    document = "\n".join(clean_data)
    return document


def text_splitter(file):
    text = read_database(file)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=30
    )
    splitted_docs = splitter.split_text(text)
    return splitted_docs

# data = text_splitter('data/news.json')
# print(data[0])

