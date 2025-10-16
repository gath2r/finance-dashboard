# app.py (가벼워진 최종 버전)

import os
import sqlite3
from datetime import date
import requests
import joblib
import pandas as pd
import json # JSON 파일을 읽기 위해 추가
import google.generativeai as genai
from flask import Flask, render_template
from dotenv import load_dotenv
from collections import Counter

# --- 초기 설정 ---
load_dotenv()
app = Flask(__name__)

# --- API 및 모델 설정 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-2.0-flash-lite')
else:
    ai_model = None
    print("⚠️  Gemini API 키가 없어 AI 분석 기능이 비활성화됩니다.")

try:
    market_model = joblib.load('market_predictor.pkl')
    print("✅ 학습된 AI 모델(market_predictor.pkl)을 불러왔습니다.")
except FileNotFoundError:
    market_model = None
    print("⚠️  저장된 AI 모델이 없습니다. 기본 규칙으로 예측합니다.")


# --- 데이터 수집 및 개별 분석 함수들 ---

def get_marketaux_news(api_key):
    if not api_key: return []
    url = f"https://api.marketaux.com/v1/news/all?countries=us&filter_entities=true&language=en&limit=10&api_token={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Marketaux API 오류: {e}")
        return []

def analyze_article_with_ai(content):
    if not ai_model or not content or len(content) < 50: return {"summary": "분석 불가", "sentiment": 0.0, "keywords": []}
    prompt = f"""
다음 금융 뉴스를 분석하여 정확히 이 형식으로 답변하세요:
SENTIMENT: [–1.0과 1.0 사이의 숫자만]
SUMMARY: [한국어로 3문장 요약]
KEYWORDS: [키워드1], [키워드2], [키워드3]
뉴스 내용:\n{content[:1000]}"""
    try:
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = ai_model.generate_content(prompt, safety_settings=safety_settings, generation_config={"temperature": 0.3})
        text = response.text.strip()
        sentiment, summary, keywords = 0.0, "분석 실패", []
        for line in text.split('\n'):
            if line.startswith('SENTIMENT:'): sentiment = float(line.split(':', 1)[1].strip())
            elif line.startswith('SUMMARY:'): summary = line.split(':', 1)[1].strip()
            elif line.startswith('KEYWORDS:'): keywords = [k.strip() for k in line.split(':', 1)[1].split(',')]
        return {"summary": summary, "sentiment": max(-1.0, min(1.0, sentiment)), "keywords": keywords[:3]}
    except Exception as e:
        print(f"❌ AI 분석 오류: {e}")
        return {"summary": "기사 분석 중 오류 발생", "sentiment": 0.0, "keywords": ["오류"]}

def save_prediction_to_db(sentiment_score, trend):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    today_str = date.today().strftime('%Y-%m-%d')
    try:
        cursor.execute("INSERT OR IGNORE INTO predictions (prediction_date, market_sentiment_score, predicted_trend) VALUES (?, ?, ?)", (today_str, sentiment_score, trend))
        conn.commit()
    finally:
        conn.close()

# --- AI 애널리스트 리포트 생성 함수 ---
def generate_trend_summary_with_ai(keywords, sentiment_score):
    if not ai_model or not keywords:
        return {"title": "분석 불가", "summary": "분석할 데이터가 부족합니다.", "keywords": []}

    common_keywords = [item[0] for item in Counter(keywords).most_common(5)]
    
    prompt = f"""
당신은 전문 금융 애널리스트입니다. 아래 데이터를 바탕으로 오늘 시장의 핵심 동향을 분석하고, 투자자를 위한 간결한 요약 리포트를 작성해주세요.

- 주요 키워드: {', '.join(common_keywords)}
- 종합 시장 감성 지수: {sentiment_score:.3f} (1.0에 가까울수록 긍정적, -1.0에 가까울수록 부정적)

아래 형식을 반드시 지켜서 한국어로 작성해주세요.
TITLE: [오늘의 시장 동향을 요약하는 매력적인 한 문장 제목]
SUMMARY: [위 데이터를 종합하여, 전문가의 시각으로 시장 상황을 2~3문장으로 분석. 딱딱한 말투가 아닌, 부드러운 조언의 톤으로.]
"""

    try:
        response = ai_model.generate_content(prompt)
        text = response.text.strip()
        
        title, summary = "AI 동향 분석", "분석 리포트 생성에 실패했습니다."
        for line in text.split('\n'):
            if line.startswith('TITLE:'): title = line.split(':', 1)[1].strip()
            elif line.startswith('SUMMARY:'): summary = line.split(':', 1)[1].strip()
                
        return {"title": title, "summary": summary, "keywords": common_keywords}
        
    except Exception as e:
        print(f"❌ AI 트렌드 요약 오류: {e}")
        return {"title": "AI 동향 분석 실패", "summary": "리포트 생성 중 오류가 발생했습니다.", "keywords": common_keywords}

# --- 데이터 읽기 함수 (새로 추가) ---
def read_json_data(file_path):
    """JSON 파일을 안전하게 읽어오는 함수"""
    try:
        # data 폴더의 상대 경로를 올바르게 설정
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 파일이 없거나 깨졌을 경우 None 반환
        print(f"⚠️ {file_path} 파일을 찾을 수 없거나 손상되었습니다.")
        return None

# --- 메인 대시보드 함수 (최종 수정) ---
@app.route('/')
def dashboard():
    # 1. GitHub Actions가 미리 만들어둔 예측 결과 파일 읽기
    nasdaq_data = read_json_data('data/nasdaq_data.json')
    kospi_data = read_json_data('data/kospi_data.json')
    fx_data = read_json_data('data/fx_data.json')
    
    # 2. 뉴스 수집 및 개별 분석
    marketaux_key = os.getenv("MARKETAUX_API_KEY", "").strip('"').strip("'")
    articles = get_marketaux_news(marketaux_key)
    
    processed_articles, all_keywords, total_sentiment = [], [], 0
    if articles:
        for article in articles:
            desc = article.get('description', '')
            if desc and len(desc) > 100:
                ai_result = analyze_article_with_ai(desc)
                article.update(ai_result)
                processed_articles.append(article)
                total_sentiment += ai_result.get('sentiment', 0.0)
                if ai_result.get('keywords'):
                    all_keywords.extend(ai_result['keywords'])

    market_sentiment_score = total_sentiment / len(processed_articles) if processed_articles else 0.0
    
    # 3. AI 애널리스트의 종합 분석 리포트 생성
    trend_summary = generate_trend_summary_with_ai(all_keywords, market_sentiment_score)
    
    # 4. 예측 및 DB 저장
    if market_model:
        trend_prediction = market_model.predict(pd.DataFrame({'market_sentiment_score': [market_sentiment_score]}))[0]
        market_status = f"{trend_prediction} 예측 (AI 모델)"
    else:
        trend_prediction = "상승" if market_sentiment_score > 0.1 else "하락"
        market_status = f"{trend_prediction} 예측 (규칙 기반)"
    save_prediction_to_db(market_sentiment_score, trend_prediction)

    return render_template(
        'index.html',
        articles=processed_articles,
        market_sentiment=market_status,
        sentiment_score=round(market_sentiment_score, 3),
        nasdaq_data=nasdaq_data,
        kospi_data=kospi_data,
        fx_data=fx_data,
        trend_summary=trend_summary
    )

if __name__ == '__main__':
    # Render가 포트를 자동으로 할당할 수 있도록 host='0.0.0.0' 추가
    # 로컬 테스트 시에는 debug=True, 배포 시에는 debug=False
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)