# run_predictions.py (Alpha Vantage API 적용 최종본)

import os
import json
from datetime import date, timedelta
import requests
import pandas as pd
import numpy as np
import yfinance as yf
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

def get_alpha_vantage_chart_data(ticker, api_key, days=30):
    """Alpha Vantage API를 사용하여 과거 데이터를 안정적으로 가져옵니다."""
    print(f"--- 📈 '{ticker}' 과거 데이터 수집 시작 ---")
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        # outputsize='compact'는 최근 100일치 데이터를 가져옵니다.
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        if data is None or data.empty:
            print(f"❌ '{ticker}' 데이터 수집 실패.")
            return None
        
        # Alpha Vantage는 컬럼 이름이 다르므로 맞게 수정
        hist_data = data['4. close'].dropna().astype(float).sort_index().tail(days)
        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        
        print(f"✅ '{ticker}' 데이터 수집 완료.")
        return {
            "labels": hist_labels,
            "historical": np.round(hist_data.values, 2).tolist(),
        }
    except Exception as e:
        print(f"❌ '{ticker}' 차트 데이터 생성 중 오류 발생: {e}")
        # 무료 API는 분당 5회 호출 제한이 있으므로, yfinance로 재시도
        print(f"⚠️ Alpha Vantage 실패, yfinance로 재시도합니다...")
        return get_yfinance_fallback(ticker, days)

def get_yfinance_fallback(ticker, days=30):
    """Alpha Vantage 실패 시 대체 작동하는 yfinance 함수"""
    try:
        today = date.today()
        start_date = today - timedelta(days=days)
        data = yf.download(ticker, start=start_date, end=today, progress=False, timeout=30)
        if data is None or data.empty: return None
        hist_data = data['Close'].dropna()
        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        return {"labels": hist_labels, "historical": np.round(hist_data.values, 2).tolist()}
    except Exception as e:
        print(f"❌ yfinance 대체 시도도 실패했습니다: {e}")
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
    
    # ... (이하 AI 분석 로직은 동일) ...
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
    # Alpha Vantage는 ^IXIC, ^KS11 같은 지수 티커를 지원하지 않으므로, 대표 ETF로 대체합니다.
    nasdaq_data = get_alpha_vantage_chart_data('QQQ', av_key)  # 나스닥 100 추종 ETF
    kospi_data = get_yfinance_fallback('^KS11') # 코스피는 yfinance 유지
    fx_data = get_alpha_vantage_chart_data('KRW', av_key) # 환율은 KRW로 조회
    
    final_data = {"articles": processed_articles, "trend_summary": trend_summary, "market_sentiment_score": round(market_sentiment_score, 3), "nasdaq_data": nasdaq_data, "kospi_data": kospi_data, "fx_data": fx_data, "last_updated": date.today().strftime("%Y-%m-%d %H:%M:%S")}

    with open('data/daily_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("\n🚀 모든 데이터가 'data/daily_data.json'에 성공적으로 저장되었습니다.")