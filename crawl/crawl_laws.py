import requests
from bs4 import BeautifulSoup

def scrape_law_content(url, output_file):
    """
    Scrape legal document content from a given URL and save it to a text file.

    Args:
        url (str): The URL of the webpage to scrape.
        output_file (str): The path of the text file to save the results.
    """
    # Fetch HTML content from the webpage
    response = requests.get(url)
    response.encoding = response.apparent_encoding  # Ensure correct encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract structured legal content
    text_content = []
    seen_sections = set()  # Set to track unique sections

    for chapter in soup.find_all('a', {'name': lambda x: x and x.startswith('chuong_')}):
        chapter_title = chapter.find_next('b').get_text(strip=True) if chapter.find_next('b') else ""
        if chapter_title and chapter_title not in seen_sections:
            text_content.append(chapter_title)
            seen_sections.add(chapter_title)

        chapter_name_tag = chapter.find_next('span')
        chapter_name = chapter_name_tag.get_text(strip=True) if chapter_name_tag else ""
        if chapter_name and chapter_name not in seen_sections:
            text_content.append(chapter_name)
            seen_sections.add(chapter_name)

        section = chapter.find_next('a', {'name': lambda x: x and x.startswith('dieu_')})
        while section:
            section_title = section.find_next('b').get_text(strip=True) if section.find_next('b') else ""
            if section_title and section_title not in seen_sections:
                text_content.append(section_title)
                seen_sections.add(section_title)
            
            section_content = []
            current_tag = section.find_next('p')
            while current_tag and not (current_tag.find('a') and current_tag.find('a').get('name', '').startswith('dieu_')):
                paragraph = " ".join(current_tag.stripped_strings)
                if paragraph and paragraph not in seen_sections:
                    section_content.append(paragraph)
                    seen_sections.add(paragraph)
                current_tag = current_tag.find_next_sibling('p')

            if section_content:
                text_content.append(" ".join(section_content))  # Ensure continuous flow of text

            section = section.find_next('a', {'name': lambda x: x and x.startswith('dieu_')})
    
    # Write data to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(text_content))
    
    print(f"Data has been saved to {output_file}")

# Example usage
# url = "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Nha-o-27-2023-QH15-528669.aspx"
# output_file = "luat_nha_o.txt"

# url="https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Dat-dai-2024-31-2024-QH15-523642.aspx"
# output_file = "luat_dat_dai.txt"

url = "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-94-2024-ND-CP-huong-dan-Luat-Kinh-doanh-bat-dong-san-xay-dung-co-so-du-lieu-ve-nha-o-619415.aspx"
output_file = "luat_kinh_doanh_bat_dong_san.txt"
scrape_law_content(url, output_file)

