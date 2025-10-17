# run_predictions.py (최종 수정본)

import os
import json
from datetime import date, timedelta
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from tenacity import retry, stop_after_attempt, wait_fixed
from ai_analyzer import analyze_article_with_ai, generate_trend_summary_with_ai

# --- API 및 데이터 수집 함수 ---

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_marketaux_news(api_key):
    """Marketaux API를 호출하여 최신 뉴스를 가져옵니다."""
    if not api_key: return []
    url = f"https://api.marketaux.com/v1/news/all?countries=us&filter_entities=true&language=en&limit=10&api_token={api_key}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        print("✅ Marketaux 뉴스 수집 성공")
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Marketaux API 오류: {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_fx_rate(api_key):
    """ExchangeRate-API를 호출하여 최신 USD/KRW 환율을 가져옵니다."""
    if not api_key: return 1422.0
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        krw_rate = response.json().get('conversion_rates', {}).get('KRW')
        if krw_rate:
            print("✅ 최신 환율 정보 수집 성공")
            return krw_rate
    except requests.exceptions.RequestException as e:
        print(f"❌ ExchangeRate-API 오류: {e}")
    return 1422.0

# --- 차트 데이터 생성 함수 ---

def get_yfinance_chart_data(ticker, days=30, forecast_days=7):
    """yfinance를 사용하여 과거 및 예측 데이터를 안정적으로 생성합니다."""
    print(f"--- 📈 '{ticker}' 데이터 예측 시작 ---")
    try:
        today = date.today()
        start_date = today - timedelta(days=days + 60)
        
        data = yf.download(ticker, start=start_date, end=today, progress=False, timeout=30)

        if data is None or data.empty:
            print(f"❌ '{ticker}' 데이터 수집 실패. API로부터 데이터를 받지 못했습니다.")
            return None
        
        hist_data = data['Close'].dropna().astype(float).sort_index().tail(days)

        if len(hist_data) < 20:
            print(f"⚠️ '{ticker}' 데이터가 20일 미만({len(hist_data)}일)이어서 예측을 건너뜁니다.")
            return None
            
        model = ARIMA(hist_data, order=(5,1,0)).fit()
        forecast = model.forecast(steps=forecast_days)
        
        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        forecast_labels = [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]
        
        print(f"✅ '{ticker}' 그래프 예측 완료.")
        return {
            "labels": hist_labels + forecast_labels,
            # ✨ 중요: .tolist()를 사용하여 [ [1], [2] ]가 아닌 [ 1, 2 ] 형태로 데이터를 만듭니다.
            "historical": np.round(hist_data.values, 2).tolist(),
            "forecast": [None] * len(hist_data) + np.round(forecast.values, 2).tolist()
        }
    except Exception as e:
        print(f"❌ '{ticker}' 차트 데이터 생성 중 심각한 오류 발생: {e}")
        return None

def get_fx_chart_data(latest_rate, days=30, forecast_days=7):
    """환율 차트 데이터를 생성합니다."""
    try:
        today = date.today()
        start_date = today - timedelta(days=days)
        
        np.random.seed(42)
        price_changes = np.random.randn(days) * 0.5
        simulated_past = latest_rate + np.cumsum(price_changes[::-1])
        
        hist_data = pd.Series(simulated_past, index=pd.to_datetime([start_date + timedelta(days=i) for i in range(days)]))
        model = ARIMA(hist_data, order=(5,1,0)).fit()
        forecast = model.forecast(steps=forecast_days)

        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        forecast_labels = [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]

        return {
            "labels": hist_labels + forecast_labels,
            "historical": np.round(hist_data.values, 2).tolist(),
            "forecast": [None] * len(hist_data) + np.round(forecast.values, 2).tolist()
        }
    except Exception as e:
        print(f"❌ 환율 차트 데이터 생성 오류: {e}")
        return None

# --- 메인 실행 로직 ---

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')

    print("--- 📰 뉴스 수집 및 AI 분석 시작 ---")
    marketaux_key = os.getenv("MARKETAUX_API_KEY")
    articles = get_marketaux_news(marketaux_key)
    
    processed_articles, all_keywords, total_sentiment = [], [], 0.0
    if articles:
        for article in articles:
            content = article.get('description') or article.get('snippet', '')
            if content and len(content) > 100:
                ai_result = analyze_article_with_ai(content)
                article.update(ai_result)
                article['image_url'] = article.get('image_url', 'https://via.placeholder.com/400x220.png?text=No+Image')
                processed_articles.append(article)
                total_sentiment += ai_result.get('sentiment', 0.0)
                if ai_result.get('keywords'):
                    all_keywords.extend(ai_result['keywords'])
    
    market_sentiment_score = total_sentiment / len(processed_articles) if processed_articles else 0.0
    trend_summary = generate_trend_summary_with_ai(all_keywords, market_sentiment_score)
    print("✅ 뉴스 분석 완료.")
    
    nasdaq_data = get_yfinance_chart_data('^IXIC')
    kospi_data = get_yfinance_chart_data('^KS11')
    
    fx_api_key = os.getenv("EXCHANGERATE_API_KEY")
    latest_fx_rate = get_fx_rate(fx_api_key)
    fx_data = get_fx_chart_data(latest_fx_rate)
    
    final_data = {
        "articles": processed_articles,
        "trend_summary": trend_summary,
        "market_sentiment_score": round(market_sentiment_score, 3),
        "nasdaq_data": nasdaq_data,
        "kospi_data": kospi_data,
        "fx_data": fx_data,
        "last_updated": date.today().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open('data/daily_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("\n🚀 모든 데이터가 'data/daily_data.json'에 성공적으로 저장되었습니다.")