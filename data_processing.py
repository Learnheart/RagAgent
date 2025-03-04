import json
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# read json files
def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_all_json_files(folder):
    json_data = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            file_path = os.path.join(folder, filename)
            json_data.append(read_json_file(file_path))
    return json_data

# read txt files
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def read_all_txt_files(folder):
    txt_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            txt_files.append(read_txt_file(file_path))
    return txt_files

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "").replace("\n", " ") 
    text = re.sub(r"\s+", " ", text).strip()  
    return text

# process text for read all json files in folder
# def extract_text(data):
#     text_list = []
    
#     # Extract the title and subtitle
#     if "title" in data:
#         text_list.append(f"LUẬT: {data['title']}")
#     if "subtitle" in data:
#         text_list.append(f"CHỦ ĐỀ: {data['subtitle']}")
    
#     if "chapters" in data:
#         for chapter in data["chapters"]:
#             if "chapter_title" in chapter:
#                 text_list.append(f"{chapter['chapter_title']}: {chapter.get('chapter_name', '')}")
            
#             if "sections" in chapter:
#                 for section in chapter["sections"]:
#                     section_title = section.get("section_title", "")
#                     section_content = section.get("section_content", "")
#                     text_list.append(f"{section_title}\n{section_content}")

#     return clean_text("\n\n".join(text_list))

# def process_and_chunk_data(folder):
#     json_data = read_all_json_files(folder)
#     # full text cleaned
#     full_text = "\n\n".join(extract_text(data) for data in json_data)
    
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500, 
#         chunk_overlap=100
#     )
#     return splitter.split_text(full_text)

# # process text for chunking a given file in folder
# def extract_text_from_a_file(article):
#     text_list = []
    
#     # Extract the title and content
#     if "title" in article:
#         text_list.append(f"Tiêu đề: {article['title']}")
#     if "content" in article:
#         text_list.append(f"Nội dung: {article['content']}")
        
#     if "chapters" in article:
#         for chapter in article["chapters"]:
#             if "chapter_title" in chapter:
#                 text_list.append(f"{chapter['chapter_title']}: {chapter.get('chapter_name', '')}")
            
#             if "sections" in chapter:
#                 for section in chapter["sections"]:
#                     section_title = section.get("section_title", "")
#                     section_content = section.get("section_content", "")
#                     text_list.append(f"{section_title}\n{section_content}")

#     return clean_text("\n\n".join(text_list))

# def chunk_single_file(file_path):
#     try:
#         data = read_json_file(file_path)
        
#         if isinstance(data, dict):
#             data = [data]
            
#         if not isinstance(data, list):
#             raise ValueError("Invalid JSON structure: Expected a list of articles.")
        
#         full_text = "\n\n".join(extract_text_from_a_file(article) for article in data)
        
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500, 
#             chunk_overlap=100
#         )
#         return splitter.split_text(full_text)
    
#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")
#         return []

# recursive chunking
def chunking_txt_file(file_path):
    data = read_txt_file(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(clean_text(data))

# chunking theo từng điều luật
def chunk_text(text, max_length=1000):
    text = read_txt_file(clean_text(text))
    pattern = r"((?:Chương\s+[IVXLCDM]+.*?|Mục\s+\d+.*?)(?=\s*Điều\s+\d+)|(?:Điều\s+\d+\..*?))(?=(?:Chương\s+[IVXLCDM]+|Mục\s+\d+|Điều\s+\d+\.|$))"
    matches = re.findall(pattern, text, re.S)

    chunks = []
    current_chunk = ""
    for match in matches:
        provision = clean_text(match)

        if "Chương" in provision or "Mục" in provision:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = provision
        else:
            current_chunk += "\n\n" + provision

        if len(current_chunk) > max_length:
            text_splitter = RecursiveCharacterTextSplitter(separators=[". ", "; ", "\n"],
                chunk_size=max_length,
                chunk_overlap=100
            )
            split_chunks = text_splitter.split_text(current_chunk)
            chunks.extend(split_chunks)
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# test
# folder = "data/luat_dat_dai.json"
# data_chunks = chunk_single_file(folder)
# if data_chunks:
#     print(data_chunks[-1])

# file = 'data/luat_dat_dai.txt'
# chunks = chunk_text(file)
# print(chunks[0])