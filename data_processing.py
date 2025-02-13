import json
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "").replace("\n", " ") 
    text = re.sub(r"\s+", " ", text).strip()  
    return text

def extract_text(data):
    text_list = []
    
    # Extract the title and subtitle
    if "title" in data:
        text_list.append(f"LUẬT: {data['title']}")
    if "subtitle" in data:
        text_list.append(f"CHỦ ĐỀ: {data['subtitle']}")
    
    # Extract chapters and sections
    if "chapters" in data:
        for chapter in data["chapters"]:
            if "chapter_title" in chapter:
                text_list.append(f"{chapter['chapter_title']}: {chapter.get('chapter_name', '')}")
            
            if "sections" in chapter:
                for section in chapter["sections"]:
                    section_title = section.get("section_title", "")
                    section_content = section.get("section_content", "")
                    text_list.append(f"{section_title}\n{section_content}")

    return clean_text("\n\n".join(text_list))

def process_and_chunk_data(folder):
    json_data = read_all_json_files(folder)
    # full text cleaned
    full_text = "\n\n".join(extract_text(data) for data in json_data)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    return splitter.split_text(full_text)

# test
folder = "data"
data_chunks = process_and_chunk_data(folder)
print(data_chunks[-1])
