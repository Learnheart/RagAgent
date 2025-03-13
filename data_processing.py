import json
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Extract chapter, title, and date from text
def extract_metadata(text):
    chapter_pattern = r"(Chương\s+[IVXLCDM]+.*?|Mục\s+\d+.*?)(?=\s*Điều\s+\d+)"
    title_date_pattern = r"(VĂN BẢN .+?)\s+(\d{2}/\d{2}/\d{4})"

    chapter_match = re.search(chapter_pattern, text)
    title_date_match = re.search(title_date_pattern, text)

    chapter = chapter_match.group(1).strip() if chapter_match else "Unknown Chapter"
    title = title_date_match.group(1).strip() if title_date_match else "Unknown Title"
    date = title_date_match.group(2) if title_date_match else "Unknown Date"

    return chapter, title, date

# Normalize text -> handle case Điều overlapping
def normalize_text(text):
    # Pattern to find all "Điều" mentions
    article_pattern = re.compile(r"(Điều\s+(\d+))(?=[\s.,])", re.IGNORECASE)

    # Store the first occurrence of each article number
    first_occurrence = {}
    matches = list(article_pattern.finditer(text))

    # Identify the correct "Điều" (with format "Điều + number + .")
    for match in matches:
        article_number = match.group(2)

        # Look ahead to check if it matches the correct format "Điều + number + ."
        if text[match.end():match.end() + 1] == '.':
            if article_number not in first_occurrence:
                first_occurrence[article_number] = match.start()

    # Normalize duplicates: lowercase all but the first correct occurrence
    normalized_text = text
    offset = 0

    for match in matches:
        article_number = match.group(2)
        start, end = match.span()

        if article_number in first_occurrence and first_occurrence[article_number] == start:
            continue  # Keep the first valid occurrence unchanged

        # Replace "Điều" with "điều" for duplicates
        normalized_text = (
            normalized_text[:start + offset] + "đ" + normalized_text[start + offset + 1:end + offset] + normalized_text[end + offset:]
        )
        offset += 0  # Adjust offset to account for modified text

    return normalized_text


# Extract all "Điều" 
def extract_articles(text):
    article_pattern = re.compile(r"(Điều\s+\d+.*?)((?=Điều\s+\d+)|$)", re.S)
    return article_pattern.findall(text)

# Read txt file 
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Chunk each article while maintaining metadata
def chunk_text_with_metadata(text, max_length=1000):
    text = normalize_text(text)
    chapter, title, date = extract_metadata(text)
    articles = extract_articles(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[". ", "; ", "\n"],
        chunk_size=max_length,
        chunk_overlap=100
    )

    chunks = []

    for article_full, _ in articles:
        article_match = re.search(r"(Điều\s+\d+)", article_full)
        article_number = article_match.group(0) if article_match else "Unknown Article"

        article_chunks = text_splitter.split_text(article_full)

        for chunk in article_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "chapter": chapter,
                    "title": title,
                    "date": date,
                    "article": article_number
                }
            })

    return chunks

# Process all text files and save chunks to JSON
def process_folder_with_metadata(folder):
    all_chunks = []

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            print(f"Processing: {filename}")

            text = read_txt_file(file_path)
            chunks = chunk_text_with_metadata(text)
            all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # Save to JSON
    output_file = "data/vectorstore.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"Chunks with metadata saved to {output_file}")

folder_path = "data/"
process_folder_with_metadata(folder_path)
