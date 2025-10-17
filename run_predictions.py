# run_predictions.py (사용자 요청 맞춤 최종본)

import os
import json
from datetime import date, timedelta
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from ai_analyzer import analyze_article_with_ai, generate_trend_summary_with_ai

load_dotenv()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_marketaux_news(api_key):
    if not api_key: return []
    url = f"https://api.marketaux.com/v1/news/all?countries=us&filter_entities=true&language=en&limit=10&api_token={api_key}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        print("✅ Marketaux 뉴스 수집 성공")
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Marketaux API 호출 중 오류 발생: {e}")
        raise e

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_yfinance_chart_data(ticker, days=30):
    """오직 지난 30일간의 과거 데이터만 가져옵니다."""
    print(f"--- 📈 '{ticker}' 과거 데이터 수집 시작 ---")
    try:
        today = date.today()
        start_date = today - timedelta(days=days)
        
        data = yf.download(ticker, start=start_date, end=today, progress=False, timeout=30)
        if data is None or data.empty:
            print(f"❌ '{ticker}' 데이터 수집 실패.")
            return None
        
        hist_data = data['Close'].dropna()
        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        
        print(f"✅ '{ticker}' 데이터 수집 완료.")
        return {
            "labels": hist_labels,
            "historical": np.round(hist_data.values, 2).tolist(),
        }
    except Exception as e:
        print(f"❌ '{ticker}' 차트 데이터 생성 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')

    print("--- 📰 뉴스 수집 및 AI 분석 시작 ---")
    marketaux_key = os.getenv("MARKETAUX_API_KEY")
    articles = []
    try:
        articles = get_marketaux_news(marketaux_key)
    except Exception:
        print("최종적으로 Marketaux 뉴스 수집에 실패했습니다.")

    print(f"➡️ 총 {len(articles)}개의 최신 뉴스를 수집했습니다.")
    if not articles: print("⚠️  수집된 뉴스가 없어 AI 분석을 건너뜁니다.")
    
    processed_articles, all_keywords, total_sentiment = [], [], 0.0
    if articles:
        for i, article in enumerate(articles):
            print(f"➡️  기사 {i+1}/{len(articles)} 분석 중...")
            content = article.get('description') or article.get('snippet', '')
            if content and len(content) > 100:
                ai_result = analyze_article_with_ai(content)
                article.update(ai_result)
                processed_articles.append(article)
                total_sentiment += ai_result.get('sentiment', 0.0)
                if ai_result.get('keywords'): all_keywords.extend(ai_result['keywords'])
    
    market_sentiment_score = total_sentiment / len(processed_articles) if processed_articles else 0.0
    trend_summary = generate_trend_summary_with_ai(all_keywords, market_sentiment_score)
    print("✅ 뉴스 분석 완료.")
    
    nasdaq_data = get_yfinance_chart_data('^IXIC')
    kospi_data = get_yfinance_chart_data('^KS11')
    fx_data = get_yfinance_chart_data('USDKRW=X')
    
    final_data = {"articles": processed_articles, "trend_summary": trend_summary, "market_sentiment_score": round(market_sentiment_score, 3), "nasdaq_data": nasdaq_data, "kospi_data": kospi_data, "fx_data": fx_data, "last_updated": date.today().strftime("%Y-%m-%d %H:%M:%S")}

    with open('data/daily_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("\n🚀 모든 데이터가 'data/daily_data.json'에 성공적으로 저장되었습니다.")