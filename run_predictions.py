# run_predictions.py (Alpha Vantage + 3일 예측 최종 완성본)

import os
import json
from datetime import date, timedelta
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.timeseries import TimeSeries
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

def get_chart_data(api_key, ticker, source='alpha_vantage', days=30, forecast_days=3):
    """금융 데이터를 가져와 분석 및 예측합니다."""
    print(f"--- 📈 '{ticker}' 데이터 예측 시작 (Source: {source}) ---")
    try:
        today = date.today()
        hist_data = None

        if source == 'alpha_vantage':
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
            if data is None or data.empty:
                print(f"❌ Alpha Vantage에서 '{ticker}' 데이터 수집 실패.")
                return None
            hist_data = data['4. close'].dropna().astype(float).sort_index().tail(days)
        else: # yfinance
            start_date = today - timedelta(days=days + 60)
            data = yf.download(ticker, start=start_date, end=today, progress=False, timeout=30)
            if data is None or data.empty:
                print(f"❌ yfinance에서 '{ticker}' 데이터 수집 실패.")
                return None
            hist_data = data['Close'].dropna().astype(float).sort_index().tail(days)

        if len(hist_data) < 20:
            print(f"⚠️ '{ticker}' 데이터가 부족하여 예측을 건너뜁니다.")
            return None
            
        model = ARIMA(hist_data, order=(5,1,0)).fit()
        forecast = model.forecast(steps=forecast_days)
        
        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        forecast_labels = [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]
        
        print(f"✅ '{ticker}' 그래프 예측 완료.")
        return {
            "labels": hist_labels + forecast_labels,
            "historical": np.round(hist_data.values, 2).tolist(),
            "forecast": [None] * len(hist_data) + np.round(forecast.values, 2).tolist()
        }
    except Exception as e:
        print(f"❌ '{ticker}' 차트 데이터 생성 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')

    print("--- 📰 뉴스 수집 및 AI 분석 시작 ---")
    marketaux_key = os.getenv("MARKETAUX_API_KEY")
    articles = []
    try: articles = get_marketaux_news(marketaux_key)
    except Exception: print("최종적으로 Marketaux 뉴스 수집에 실패했습니다.")

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
    
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    nasdaq_data = get_chart_data(av_key, ticker='QQQ', source='alpha_vantage')
    kospi_data = get_chart_data(av_key, ticker='^KS11', source='yfinance') # 코스피는 yfinance 사용
    fx_data = get_chart_data(av_key, ticker='USDKRW', source='alpha_vantage')
    
    final_data = {"articles": processed_articles, "trend_summary": trend_summary, "market_sentiment_score": round(market_sentiment_score, 3), "nasdaq_data": nasdaq_data, "kospi_data": kospi_data, "fx_data": fx_data, "last_updated": date.today().strftime("%Y-%m-%d %H:%M:%S")}

    with open('data/daily_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("\n🚀 모든 데이터가 'data/daily_data.json'에 성공적으로 저장되었습니다.")