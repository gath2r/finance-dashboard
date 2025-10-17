# ai_analyzer.py
import os
import google.generativeai as genai
from collections import Counter

# --- AI 설정 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    ai_model = None
    print("⚠️ Gemini API 키가 없어 AI 분석 기능이 비활성화됩니다.")

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
        safety_settings = [
            {"category": c, "threshold": "BLOCK_NONE"} for c in 
            ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        response = ai_model.generate_content(prompt, safety_settings=safety_settings, generation_config={"temperature": 0.3})
        
        text = response.text.strip()
        sentiment, summary, keywords = 0.0, "분석 실패", []
        
        for line in text.split('\n'):
            if line.startswith('SENTIMENT:'):
                sentiment = float(line.split(':', 1)[1].strip())
            elif line.startswith('SUMMARY:'):
                summary = line.split(':', 1)[1].strip()
            elif line.startswith('KEYWORDS:'):
                keywords = [k.strip() for k in line.split(':', 1)[1].split(',')]
                
        return {"summary": summary, "sentiment": max(-1.0, min(1.0, sentiment)), "keywords": keywords[:3]}
    except Exception as e:
        print(f"❌ AI 기사 분석 오류: {e}")
        return {"summary": "기사 분석 중 오류 발생", "sentiment": 0.0, "keywords": ["오류"]}

def generate_trend_summary_with_ai(keywords, sentiment_score):
    """AI를 사용하여 시장 트렌드 요약을 생성합니다."""
    if not ai_model or not keywords:
        return {"title": "분석 데이터 부족", "summary": "AI 트렌드 요약을 생성하기 위한 데이터가 부족합니다.", "keywords": []}
    
    common_keywords = [item[0] for item in Counter(keywords).most_common(5)]
    prompt = f"""
    As a professional financial analyst, analyze the following data and write a concise market trend report for investors.
    - Key Keywords: {', '.join(common_keywords)}
    - Overall Market Sentiment Score: {sentiment_score:.3f}

    Please strictly adhere to the following format in Korean:
    TITLE: [An engaging one-sentence title summarizing today's market trend]
    SUMMARY: [Analyze the market situation in 2-3 sentences from an expert's perspective, combining the data above. Use a gentle, advisory tone.]
    """
    try:
        response = ai_model.generate_content(prompt, generation_config={"temperature": 0.5})
        text = response.text.strip()
        title, summary = "AI 동향 분석", "리포트 생성 실패"
        
        for line in text.split('\n'):
            if line.startswith('TITLE:'):
                title = line.split(':', 1)[1].strip()
            elif line.startswith('SUMMARY:'):
                summary = line.split(':', 1)[1].strip()
                
        return {"title": title, "summary": summary, "keywords": common_keywords}
    except Exception as e:
        print(f"❌ AI 트렌드 요약 오류: {e}")
        return {"title": "AI 동향 분석 실패", "summary": "리포트 생성 중 오류가 발생했습니다.", "keywords": common_keywords}