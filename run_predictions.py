# run_predictions.py (AI 예측 DB 저장 기능 추가 최종본)

import os
import json
from datetime import date, timedelta, datetime
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.foreignexchange import ForeignExchange
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from ai_analyzer import analyze_article_with_ai, generate_trend_summary_with_ai
import yfinance as yf
import sqlite3 # ▼▼▼ 1. sqlite3 임포트 ▼▼▼

load_dotenv()

def process_chart_data(hist_data, forecast_days=3):
    """데이터프레임을 받아 예측을 수행하고 차트 형식으로 반환합니다."""
    if hist_data is None or len(hist_data) < 20:
        raise ValueError(f"예측을 위한 데이터가 부족합니다 (현재: {len(hist_data) if hist_data is not None else 0}개)")
    
    # hist_data가 DataFrame인 경우 Series로 변환
    if isinstance(hist_data, pd.DataFrame):
        hist_data = hist_data.iloc[:, 0]
    
    today = date.today()
    
    try:
        # ARIMA 모델 학습 및 예측
        model = ARIMA(hist_data, order=(5,1,0)).fit()
        forecast = model.forecast(steps=forecast_days)
        
        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        forecast_labels = [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]
        
        # 1차원 배열로 통일하여 반환
        historical_values = hist_data.values
        if len(historical_values.shape) > 1:
            historical_values = historical_values.flatten()
        
        forecast_values = forecast.values
        if len(forecast_values.shape) > 1:
            forecast_values = forecast_values.flatten()
        
        return {
            "labels": hist_labels + forecast_labels,
            "historical": np.round(historical_values, 2).tolist(),
            "forecast": [None] * len(hist_data) + np.round(forecast_values, 2).tolist()
        }
    except Exception as e:
        print(f"❌ 예측 모델 생성 중 오류: {e}")
        import traceback
        print(traceback.format_exc())
        raise e


# --- ▼▼▼ 2. AI 예측을 DB에 저장하는 함수 추가 ▼▼▼ ---
def save_prediction_to_db(prediction_date, sentiment_score, predicted_trend):
    """AI의 예측 결과를 DB에 저장합니다."""
    conn = None
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # database_setup.py의 predictions 테이블 구조에 맞게 삽입
        cursor.execute("""
        INSERT OR REPLACE INTO predictions (prediction_date, market_sentiment_score, predicted_trend)
        VALUES (?, ?, ?)
        """, (prediction_date, sentiment_score, predicted_trend))
        
        conn.commit()
        print(f"✅ DB에 {prediction_date}의 예측 결과 '{predicted_trend}' (점수: {sentiment_score}) 저장 완료")
    except Exception as e:
        print(f"❌ DB 예측 저장 실패: {e}")
    finally:
        if conn:
            conn.close()
# --- ▲▲▲ 함수 추가 완료 ▲▲▲ ---


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_marketaux_news(api_key):
    """Marketaux API로 최신 금융 뉴스를 가져옵니다."""
    if not api_key:
        return []
    url = f"https://api.marketaux.com/v1/news/all?countries=us&filter_entities=true&language=en&limit=10&api_token={api_key}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        print("✅ Marketaux 뉴스 수집 성공")
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Marketaux API 호출 중 오류 발생: {e}")
        raise e


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')

    # 1. 뉴스 수집 및 AI 분석
    print("=" * 60)
    print("📰 뉴스 수집 및 AI 분석 시작")
    print("=" * 60)
    
    marketaux_key = os.getenv("MARKETAUX_API_KEY")
    articles = []
    try:
        articles = get_marketaux_news(marketaux_key)
    except Exception:
        print("⚠️  최종적으로 Marketaux 뉴스 수집에 실패했습니다.")

    print(f"➡️  총 {len(articles)}개의 최신 뉴스를 수집했습니다.")
    
    processed_articles, all_keywords, total_sentiment = [], [], 0.0
    if articles:
        for i, article in enumerate(articles):
            print(f"➡️  기사 {i+1}/{len(articles)} 분석 중...", end=" ")
            content = article.get('description') or article.get('snippet', '')
            if content and len(content) > 100:
                ai_result = analyze_article_with_ai(content)
                article.update(ai_result)
                processed_articles.append(article)
                total_sentiment += ai_result.get('sentiment', 0.0)
                if ai_result.get('keywords'):
                    all_keywords.extend(ai_result['keywords'])
                print("✅")
            else:
                print("⏭️  (내용 부족)")
    
    market_sentiment_score = total_sentiment / len(processed_articles) if processed_articles else 0.0
    trend_summary = generate_trend_summary_with_ai(all_keywords, market_sentiment_score)
    print("✅ 뉴스 분석 완료\n")
    
    # 2. 주식 데이터 수집 (yfinance)
    nasdaq_data, kospi_data, fx_data = None, None, None
    
    print("=" * 60)
    print("📈 주식 데이터 수집 시작 (yfinance)")
    print("=" * 60)
    
    # 나스닥
    try:
        print("➡️  나스닥 종합지수 데이터 수집 중...", end=" ")
        nasdaq = yf.Ticker("^IXIC")
        nasdaq_hist = nasdaq.history(period="3mo", auto_adjust=False, actions=False)
        
        if len(nasdaq_hist) > 0:
            nasdaq_df = nasdaq_hist['Close'].tail(30)
            print(f"({len(nasdaq_df)}일 수집)")
            nasdaq_data = process_chart_data(nasdaq_df)
            print("✅ 나스닥 예측 완료")
        else:
            print("⚠️  데이터 없음")
    except Exception as e:
        print(f"❌ 실패: {e}")
    
    # 코스피
    try:
        print("➡️  코스피 지수 데이터 수집 중...", end=" ")
        kospi = yf.Ticker("^KS11")
        kospi_hist = kospi.history(period="3mo", auto_adjust=False, actions=False)
        
        if len(kospi_hist) > 0:
            kospi_df = kospi_hist['Close'].tail(30)
            print(f"({len(kospi_df)}일 수집)")
            kospi_data = process_chart_data(kospi_df)
            print("✅ 코스피 예측 완료")
        else:
            print("⚠️  데이터 없음")
    except Exception as e:
        print(f"❌ 실패: {e}")

    # 3. 환율 데이터 수집 (Alpha Vantage)
    print("\n" + "=" * 60)
    print("💱 환율 데이터 수집 시작 (Alpha Vantage)")
    print("=" * 60)
    
    try:
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if av_key:
            print("➡️  USD/KRW 환율 데이터 수집 중...", end=" ")
            cc = ForeignExchange(key=av_key, output_format='pandas')
            fx_raw, _ = cc.get_currency_exchange_daily(from_symbol='USD', to_symbol='KRW', outputsize='compact')
            fx_hist = fx_raw['4. close'].dropna().astype(float).sort_index().tail(30)
            print(f"({len(fx_hist)}일 수집)")
            fx_data = process_chart_data(fx_hist)
            print("✅ 환율 예측 완료")
        else:
            print("⚠️  Alpha Vantage API 키가 설정되지 않았습니다.")
    except Exception as e:
        print(f"❌ 실패: {e}")

    # 4. 최종 데이터 저장
    print("\n" + "=" * 60)
    print("💾 데이터 저장 중...")
    print("=" * 60)
    
    # --- ▼▼▼ 3. DB 저장 로직 추가 (수정된 부분) ▼▼▼ ---
    today_str = date.today().strftime('%Y-%m-%d')
    
    # 시장 심리 점수에 따라 AI의 예측 결정 (0.1 이상 '상승', -0.1 이하 '하락')
    ai_predicted_trend = '상승'
    if market_sentiment_score < -0.1:
        ai_predicted_trend = '하락'
    
    # DB에 오늘의 예측 저장
    save_prediction_to_db(today_str, round(market_sentiment_score, 3), ai_predicted_trend)
    # --- ▲▲▲ DB 저장 로직 추가 완료 ▲▲▲ ---

    final_data = {
        "articles": processed_articles,
        "trend_summary": trend_summary,
        "market_sentiment_score": round(market_sentiment_score, 3),
        "nasdaq_data": nasdaq_data,
        "kospi_data": kospi_data,
        "fx_data": fx_data,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open('data/daily_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("✅ 'data/daily_data.json'에 저장 완료")
    
    # 저장된 데이터 검증
    print("\n" + "=" * 60)
    print("🔍 저장된 데이터 검증")
    print("=" * 60)
    
    if nasdaq_data:
        print(f"✅ 나스닥: {len(nasdaq_data['historical'])}개 데이터포인트")
        print(f"   샘플: {nasdaq_data['historical'][:3]}")
    else:
        print("⚠️  나스닥 데이터 없음")
    
    if kospi_data:
        print(f"✅ 코스피: {len(kospi_data['historical'])}개 데이터포인트")
        print(f"   샘플: {kospi_data['historical'][:3]}")
    else:
        print("⚠️  코스피 데이터 없음")
    
    if fx_data:
        print(f"✅ 환율: {len(fx_data['historical'])}개 데이터포인트")
        print(f"   샘플: {fx_data['historical'][:3]}")
    else:
        print("⚠️  환율 데이터 없음")
    
    print("\n" + "=" * 60)
    print("🚀 모든 작업 완료!")
    print("=" * 60)