from wtpsplit import SaT
from dotenv import load_dotenv

from google import genai
load_dotenv()
import os

value = os.getenv('MY_ENV_VAR')


class Preprocessor:
    def __init__(self):
        self.Sat = SaT
    
    def tokenize(self,paragraph):
        sat = SaT("sat-3l")
        value = os.getenv('API_KEY')

        client = genai.Client(api_key=value)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents= f"""Hãy sửa lại đoạn văn để đúng chính tả, ngữ pháp, dấu câu và dùng từ đúng và từ nội dung của văn bản đó hãy viết lại và dài hơn tập trung vào lý thuyết chính của văn bản đó.Chỉ trả về văn bản đã được tạo và không xuống dòng  

                                                    Văn bản: 
                                                    {paragraph}
                                                    """
        )
        print(response.text)
        sentences=sat.split(response.text)
        return sentences