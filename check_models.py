# check_models.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일에서 API 키를 불러옵니다.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Gemini API 키를 찾을 수 없습니다! .env 파일을 확인해주세요.")
else:
    genai.configure(api_key=api_key)
    print("--- 🔬 당신의 API 키로 사용 가능한 모델 목록 ---")
    
    # 사용 가능한 모든 모델을 순회하며 이름과 설명을 출력합니다.
    for model in genai.list_models():
        # generateContent 메서드를 지원하는, 즉 텍스트 생성이 가능한 모델만 필터링합니다.
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")
            
    print("\n--- [사용 방법] ---")
    print("위 목록에서 'gemini-pro' 또는 '-pro'가 포함된 모델 이름을 복사하여")
    print("ai_analyzer.py 파일의 ai_model = genai.GenerativeModel('모델이름') 부분에 붙여넣으세요.")