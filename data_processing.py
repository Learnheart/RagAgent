import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# def read_json(file):
#     with open(file, 'r') as f:
#         articles = json.load(f)
#     data = [article['content'] for article in articles]
#     return data

def read_files_from_data(folder):
    json_list = []
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            file_path = os.path.join(folder, filename)

            with open(file_path, 'r') as f:
                data = json.load(f)
                json_list.append(data)
    return json_list

def clean_text(file):
    if not isinstance(file, str):
        return ""
    text = file.replace('\\n', ' ').replace('\r', '').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def extract_text_from_json(json):
    if isinstance(json, dict):
        return clean_text(" ").join(str(value) for value in json.values())
    elif isinstance(json, list):
        return clean_text(" ").join(str(item) for item in json)
    return ""

def read_database(file) -> str:
    articles = read_files_from_data(file)
    clean_data = (extract_text_from_json(article) for article in articles)
    document = "\n".join(clean_data)
    return document

def text_splitter(file):
    text = read_database(file)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splitted_docs = splitter.split_text(text)
    return splitted_docs

# folder = 'data'
# data = text_splitter(folder)
# print(data)
