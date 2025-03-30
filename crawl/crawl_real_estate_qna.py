import requests
from bs4 import BeautifulSoup
import json


def crawl_legal_questions(url):
    """
    Crawl câu hỏi và câu trả lời từ trang web Thư Viện Pháp Luật và lưu vào file JSON.
    
    :param url: URL của trang web cần crawl
    :param output_file: Tên file JSON để lưu kết quả (mặc định: data.json)
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra nếu request bị lỗi
    except requests.RequestException as e:
        print(f"Lỗi khi gửi request: {e}")
        return
    
    soup = BeautifulSoup(response.text, "html.parser")
    questions = soup.find_all("h2", id=True)
    data = {}

    for question in questions:
        question_text = question.get_text(strip=True)
        answer_parts = []
        sibling = question.find_next_sibling()

        while sibling and sibling.name in ["p", "blockquote"]:
            for em in sibling.find_all("em"):
                em.decompose()
            answer_parts.append(sibling.get_text(separator=" ", strip=True))  # Đảm bảo khoảng cách giữa các phần
            sibling = sibling.find_next_sibling()

        answer_text = " ".join(answer_parts)  # Ghép nội dung với khoảng trắng rõ ràng
        data[question_text] = answer_text

    return data 

def get_article_links(base_url, max_pages=10):
    """
    Lấy tất cả các liên kết bài viết từ trang chính và các trang tiếp theo 
    :param base_url: URL của trang web cần crawl
    :param max_pages: Số trang tối đa để crawl (mặc định: 10)
    :return: Danh sách các URL bài viết
    """
    all_links = []
    page = 1

    while page <= max_pages:
        url = f"{base_url}?page={page}" if page > 1 else base_url
        print(f"Đang lấy link từ trang {page}: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Lỗi khi gửi request đến {url}: {e}")
            break 

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article", class_="news-card")

        new_links = []
        for article in articles:
            link_tag = article.find("a", href=True)
            if link_tag:
                full_link = link_tag["href"]
                if not full_link.startswith("http"):  # Chuyển đường dẫn tương đối thành tuyệt đối
                    full_link = base_url.rstrip("/") + "/" + full_link.lstrip("/")
                new_links.append(full_link)

        if not new_links:
            print("Không tìm thấy link mới, dừng crawl.")
            break 

        all_links.extend(new_links)
        page += 1

    print(f"Đã thu thập tổng cộng {len(all_links)} link.")
    return all_links

def process(base_url, output_file,pages):
    """
    Thu thập dữ liệu từ các bài viết trên trang web và lưu vào file JSON.
    
    :param base_url: URL của trang web cần crawl
    :param output_file: Tên file JSON để lưu kết quả
    """
    article_links = get_article_links(base_url,pages)
    content = {}
    
    for link in article_links:
        data = crawl_legal_questions(link)
        content.update(data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được lưu vào {output_file}")

#Execute
output_file ="crawl_realestate_qna.json"
url=f"https://thuvienphapluat.vn/hoi-dap-phap-luat/bat-dong-san"
process(url,output_file,20)


