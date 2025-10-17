# ai_analyzer.py (타임아웃 기능 추가 최종 완성본)
import os
from dotenv import load_dotenv
import google.generativeai as genai
from collections import Counter

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('models/gemini-2.5-flash')
    print("✅ Gemini API 키가 성공적으로 로드되었습니다.")
else:
    ai_model = None
    print("❌ Gemini API 키를 찾을 수 없습니다! .env 파일을 확인해주세요.")

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def analyze_article_with_ai(content):
    """Gemini AI를 사용하여 뉴스 기사를 분석합니다."""
    if not ai_model or not content or len(content) < 50:
        return {"summary": "분석 불가", "sentiment": 0.0, "keywords": []}
    
    prompt = f"""
    Analyze the following financial news article and provide the response strictly in the following format:
    SENTIMENT: [A single number between -1.0 and 1.0]
    SUMMARY: [A 3-sentence summary in Korean]
    KEYWORDS: [keyword1], [keyword2], [keyword3]

    News Content:\n{content[:1500]}
    """
    try:
        # ✨ 중요: 60초 타임아웃 설정 추가
        response = ai_model.generate_content(
            prompt,
            safety_settings=SAFETY_SETTINGS,
            generation_config={"temperature": 0.3},
            request_options={"timeout": 60}
        )
        
        text = response.text.strip()
        sentiment, summary, keywords = 0.0, "분석 실패", []
        for line in text.split('\n'):
            if 'SENTIMENT:' in line: sentiment = float(line.split(':', 1)[1].strip())
            elif 'SUMMARY:' in line: summary = line.split(':', 1)[1].strip()
            elif 'KEYWORDS:' in line: keywords = [k.strip() for k in line.split(':', 1)[1].split(',')]
        return {"summary": summary, "sentiment": max(-1.0, min(1.0, sentiment)), "keywords": keywords[:3]}
    except Exception as e:
        print(f"❌ AI 기사 분석 중 타임아웃 또는 오류 발생: {e}")
        return {"summary": "AI 응답 지연", "sentiment": 0.0, "keywords": ["오류"]}

def generate_trend_summary_with_ai(keywords, sentiment_score):
    """AI를 사용하여 시장 트렌드 요약을 생성합니다."""
    if not ai_model or not keywords:
        return {"title": "분석 데이터 부족", "summary": "AI 트렌드 요약을 생성하기 위한 데이터가 부족합니다.", "keywords": []}
    
    common_keywords = [item[0] for item in Counter(keywords).most_common(5)]
    prompt = f"""
    As a professional financial analyst, write a concise market trend report in Korean based on the following data.
    - Key Keywords: {', '.join(common_keywords)}
    - Overall Market Sentiment Score: {sentiment_score:.3f}

    Strictly adhere to this format:
    TITLE: [Engaging one-sentence title]
    SUMMARY: [Expert analysis in 2-3 sentences with a gentle, advisory tone.]
    """
    try:
        # ✨ 중요: 여기에도 60초 타임아웃 설정 추가
        response = ai_model.generate_content(
            prompt,
            safety_settings=SAFETY_SETTINGS,
            generation_config={"temperature": 0.5},
            request_options={"timeout": 60}
        )
        text = response.text.strip()
        title, summary = "AI 동향 분석", "리포트 생성 실패"
        for line in text.split('\n'):
            if 'TITLE:' in line: title = line.split(':', 1)[1].strip()
            elif 'SUMMARY:' in line: summary = line.split(':', 1)[1].strip()
        return {"title": title, "summary": summary, "keywords": common_keywords}
    except Exception as e:
        print(f"❌ AI 트렌드 요약 중 타임아웃 또는 오류 발생: {e}")
        return {"title": "AI 응답 지연", "summary": "트렌드 요약 생성에 실패했습니다.", "keywords": common_keywords}