import requests
from bs4 import BeautifulSoup
import re

def extract_structured_text_from_html_content(html_content):
    """Trích xuất văn bản từ nội dung HTML giữ nguyên cấu trúc"""
    soup = BeautifulSoup(html_content, 'html.parser')
    structured_text = ""
    
    chapter_title = soup.find('a', {'name': 'chuong_1_name'})
    if chapter_title:
        structured_text += "Chương I\n\n"
    
    title = soup.find('strong', string=lambda text: text and "NHỮNG QUY ĐỊNH CHUNG" in text)
    if title:
        structured_text += "NHỮNG QUY ĐỊNH CHUNG\n\n"
    
    articles = soup.find_all('a', {'name': re.compile(r'dieu_\d+')})
    
    for article in articles:
        parent_p = article.find_parent('p')
        if parent_p:
            article_title = parent_p.get_text().strip()
            structured_text += article_title + "\n\n"
            next_p = parent_p.find_next_sibling('p')
            while next_p and not next_p.find('a', {'name': re.compile(r'dieu_\d+')}):
                paragraph_text = next_p.get_text().strip()
                if re.match(r'^\d+\.', paragraph_text):
                    structured_text += paragraph_text + "\n\n"
                else:
                    structured_text += paragraph_text + "\n\n"
                
                next_p = next_p.find_next_sibling('p')
    
    if not structured_text:
        # Xử lý tất cả các đoạn văn
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text().strip()
            if text:
                # Tìm các tiêu đề chính
                if "Điều" in text and re.search(r'Điều \d+\.', text):
                    structured_text += "\n" + text + "\n\n"
                # Các điểm con
                elif re.match(r'^\d+\.', text):
                    structured_text += text + "\n\n"
                # Văn bản bình thường
                else:
                    structured_text += text + "\n\n"
    
    return structured_text

def extract_text_from_url(url, output_txt_file):
    """Trích xuất văn bản từ URL và lưu vào file TXT"""
    try:
        # Gửi yêu cầu HTTP để lấy nội dung HTML
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        
        # Đảm bảo encoding chính xác
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding
            
        html_content = response.text
        
        # Trích xuất văn bản có cấu trúc
        text_content = extract_structured_text_from_html_content(html_content)
        
        # Ghi nội dung vào file TXT
        with open(output_txt_file, 'w', encoding='utf-8') as file:
            file.write(text_content)
        
        print(f"Đã trích xuất nội dung từ URL thành công và lưu vào file {output_txt_file}")
        return text_content
    
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải nội dung từ URL: {e}")
        return None

# URL cần trích xuất - bạn có thể thay đổi URL ở đây
url = "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-94-2024-ND-CP-huong-dan-Luat-Kinh-doanh-bat-dong-san-xay-dung-co-so-du-lieu-ve-nha-o-619415.aspx"
# url="https://thuvienphapluat.vn/van-ban/Thuong-mai/Nghi-dinh-96-2024-ND-CP-huong-dan-Luat-Kinh-doanh-bat-dong-san-600395.aspx"
# url="https://thuvienphapluat.vn/van-ban/Bat-dong-san/Thong-tu-04-2024-TT-BXD-chuong-trinh-khung-dao-tao-kien-thuc-hanh-nghe-moi-gioi-bat-dong-san-619409.aspx"
# Đường dẫn file đầu ra
output_file = "output_document.txt"

# Thực hiện trích xuất
text_content = extract_text_from_url(url, output_file)

