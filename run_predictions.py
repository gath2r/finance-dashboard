# run_predictions.py (KIS API 통합 최종 완성본)

import os
import json
from datetime import date, timedelta, datetime
import time
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.foreignexchange import ForeignExchange
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from ai_analyzer import analyze_article_with_ai, generate_trend_summary_with_ai

load_dotenv()

# --- KIS API 관련 함수 ---
ACCESS_TOKEN = ""
TOKEN_EXPIRATION = None

def get_kis_token(app_key, app_secret):
    """한국투자증권 API 인증 토큰을 발급받습니다."""
    global ACCESS_TOKEN, TOKEN_EXPIRATION
    
    # 토큰이 유효하면 재사용
    if ACCESS_TOKEN and TOKEN_EXPIRATION and datetime.now() < TOKEN_EXPIRATION:
        print("✅ KIS 토큰 재사용")
        return ACCESS_TOKEN

    print("--- 🔑 KIS 신규 토큰 발급 시도 ---")
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP" # 모의투자용 URL
    # url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP" # 실전투자용 URL
    headers = {"content-type": "application/json"}
    data = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
    
    res = requests.post(url, headers=headers, data=json.dumps(data))
    if res.status_code != 200:
        print(f"❌ KIS 토큰 발급 실패: {res.text}")
        return None

    token_data = res.json()
    ACCESS_TOKEN = f"Bearer {token_data['access_token']}"
    # 토큰 만료 시간보다 10분 일찍 만료된 것으로 처리하여 안정성 확보
    TOKEN_EXPIRATION = datetime.now() + timedelta(seconds=token_data['expires_in'] - 600)
    print("✅ KIS 신규 토큰 발급 성공")
    return ACCESS_TOKEN

def get_kis_daily_chart(token, market, code, days=30):
    """KIS API를 사용하여 일봉 데이터를 가져옵니다."""
    url = "https://openapivts.koreainvestment.com:29443/uapi/overseas-price/v1/quotations/dailyprice" # 해외(모의)
    if market == "KSP":
        url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-price" # 국내(모의)
        
    headers = {
        "content-type": "application/json",
        "authorization": token,
        "appkey": os.getenv("KIS_APP_KEY"),
        "appsecret": os.getenv("KIS_APP_SECRET"),
        "tr_id": "HHDFS76240000" if market != "KSP" else "FHKST01010400"
    }
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days + 60) # 넉넉하게 데이터 조회
    
    params = {
        "PBLS": code,
        "GUBN": "D", # 일봉
        "STD_DT": start_date.strftime('%Y%m%d'),
        "MODP": "0",
        "KEY_DATA": ""
    }
    if market != "KSP": # 해외 API 파라미터
        params = {"AUTH": "", "EXCD": market, "SYMB": code, "GUBN": "D", "BYMD": "", "MODP": "0"}

    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        raise Exception(f"KIS API 데이터 요청 실패: {res.text}")
    
    data = res.json()['output']
    df = pd.DataFrame(data)
    
    # API별 컬럼 이름 통일
    date_col = 'stck_bsop_date' if market == "KSP" else 'ymd'
    close_col = 'stck_clpr' if market == "KSP" else 'clos'
    
    df = df[[date_col, close_col]]
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['close'] = pd.to_numeric(df['close'])
    df = df.set_index('date').sort_index()
    return df['close'].tail(days)


# --- 기존 함수들 ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_marketaux_news(api_key):
    # ... (내용 동일) ...
    if not api_key: return []
    url = f"https://api.marketaux.com/v1/news/all?countries=us&filter_entities=true&language=en&limit=10&api_token={api_key}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status(); print("✅ Marketaux 뉴스 수집 성공")
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Marketaux API 호출 중 오류 발생: {e}"); raise e

def process_chart_data(hist_data, forecast_days=3):
    """데이터프레임을 받아 예측을 수행하고 차트 형식으로 반환합니다."""
    if hist_data is None or len(hist_data) < 20:
        raise ValueError("예측을 위한 데이터가 부족합니다.")
    
    today = date.today()
    model = ARIMA(hist_data, order=(5,1,0)).fit()
    forecast = model.forecast(steps=forecast_days)
    
    hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
    forecast_labels = [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]
    
    return {
        "labels": hist_labels + forecast_labels,
        "historical": np.round(hist_data.values, 2).tolist(),
        "forecast": [None] * len(hist_data) + np.round(forecast.values, 2).tolist()
    }

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')

    # ... (뉴스 분석 로직은 동일) ...
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
                article.update(ai_result); processed_articles.append(article)
                total_sentiment += ai_result.get('sentiment', 0.0)
                if ai_result.get('keywords'): all_keywords.extend(ai_result['keywords'])
    
    market_sentiment_score = total_sentiment / len(processed_articles) if processed_articles else 0.0
    trend_summary = generate_trend_summary_with_ai(all_keywords, market_sentiment_score)
    print("✅ 뉴스 분석 완료.")
    
    # --- 차트 데이터 생성 ---
    nasdaq_data, kospi_data, fx_data = None, None, None
    try:
        # 1. KIS API로 나스닥/코스피 데이터 가져오기
        kis_token = get_kis_token(os.getenv("KIS_APP_KEY"), os.getenv("KIS_APP_SECRET"))
        if kis_token:
            print("--- 📈 KIS API로 데이터 예측 시작 ---")
            nasdaq_hist = get_kis_daily_chart(kis_token, market="NAS", code="COMP") # 나스닥 종합지수
            nasdaq_data = process_chart_data(nasdaq_hist)
            print("✅ 나스닥 예측 완료.")
            
            time.sleep(1) # API 과호출 방지
            
            kospi_hist = get_kis_daily_chart(kis_token, market="KSP", code="0001") # 코스피 지수
            kospi_data = process_chart_data(kospi_hist)
            print("✅ 코스피 예측 완료.")
    except Exception as e:
        print(f"❌ KIS API 처리 중 오류: {e}")

    try:
        # 2. Alpha Vantage로 환율 데이터 가져오기 (사용자가 만족했던 부분)
        print("--- 📈 Alpha Vantage API로 환율 예측 시작 ---")
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        cc = ForeignExchange(key=av_key, output_format='pandas')
        data, _ = cc.get_currency_exchange_daily(from_symbol='USD', to_symbol='KRW', outputsize='compact')
        fx_hist = data['4. close'].dropna().astype(float).sort_index().tail(30)
        fx_data = process_chart_data(fx_hist)
        print("✅ 환율 예측 완료.")
    except Exception as e:
        print(f"❌ Alpha Vantage 처리 중 오류: {e}")

    final_data = {"articles": processed_articles, "trend_summary": trend_summary, "market_sentiment_score": round(market_sentiment_score, 3), "nasdaq_data": nasdaq_data, "kospi_data": kospi_data, "fx_data": fx_data, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    with open('data/daily_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("\n🚀 모든 데이터가 'data/daily_data.json'에 성공적으로 저장되었습니다.")