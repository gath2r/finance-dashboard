# run_predictions.py (데이터 형식 수정 및 3일 예측 기능 추가 최종본)

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
    if ACCESS_TOKEN and TOKEN_EXPIRATION and datetime.now() < TOKEN_EXPIRATION:
        print("✅ KIS 토큰 재사용")
        return ACCESS_TOKEN
    print("--- 🔑 KIS 신규 토큰 발급 시도 ---")
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP" # 모의투자용 URL
    headers = {"content-type": "application/json"}
    data = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
    try:
        res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
        res.raise_for_status(); token_data = res.json()
        ACCESS_TOKEN = f"Bearer {token_data['access_token']}"
        TOKEN_EXPIRATION = datetime.now() + timedelta(seconds=token_data['expires_in'] - 600)
        print("✅ KIS 신규 토큰 발급 성공")
        return ACCESS_TOKEN
    except Exception as e:
        print(f"❌ KIS 토큰 발급 실패: {e}")
        return None

def get_kis_daily_chart(token, market, code, days=30):
    """KIS API를 사용하여 일봉 데이터를 가져옵니다."""
    if market == "KSP":
        url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        tr_id = "FHKST03010100"
    else:
        url = "https://openapivts.koreainvestment.com:29443/uapi/overseas-price/v1/quotations/inquire-daily-chartprice"
        tr_id = "HHDFS76950200"
    headers = {"content-type":"application/json; charset=utf-8", "authorization":token, "appkey":os.getenv("KIS_APP_KEY"), "appsecret":os.getenv("KIS_APP_SECRET"), "tr_id":tr_id}
    end_date = date.today(); start_date = end_date - timedelta(days=100)
    if market == "KSP":
        params = {"FID_COND_MRKT_DIV_CODE":"J", "FID_INPUT_ISCD":code, "FID_INPUT_DATE_1":start_date.strftime('%Y%m%d'), "FID_INPUT_DATE_2":end_date.strftime('%Y%m%d'), "FID_PERIOD_DIV_CODE":"D", "FID_ORG_ADJ_PRC":"0"}
    else:
        params = {"FID_COND_MRKT_DIV_CODE":"N", "FID_INPUT_ISCD":code, "FID_INPUT_DATE_1":start_date.strftime('%Y%m%d'), "FID_INPUT_DATE_2":end_date.strftime('%Y%m%d'), "FID_PERIOD_DIV_CODE":"D"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=15); res.raise_for_status(); data = res.json()
        if data.get('rt_cd') != '0': raise Exception(f"KIS API 에러: {data.get('msg1')}")
        output = data.get('output2', data.get('output', []));
        if not output: raise Exception("API 응답에 데이터가 없습니다")
        df = pd.DataFrame(output)
        date_col = 'stck_bsop_date' if market == "KSP" else ('xymd' if 'xymd' in df.columns else 'stck_bsop_date')
        close_col = 'stck_clpr' if market == "KSP" else ('clos' if 'clos' in df.columns else 'stck_clpr')
        if date_col not in df.columns or close_col not in df.columns: raise Exception(f"컬럼 불일치: {df.columns.tolist()}")
        df = df[[date_col, close_col]].copy(); df.columns = ['date', 'close']
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d'); df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna().set_index('date').sort_index(); result = df['close'].tail(days)
        if len(result) < 20: print(f"⚠️ 데이터 부족 ({len(result)}개).")
        return result
    except Exception as e: print(f"❌ {market} {code} 데이터 수집 중 오류: {e}"); raise e

def process_chart_data(hist_data, forecast_days=3):
    """데이터프레임을 받아 예측을 수행하고 차트 형식으로 반환합니다."""
    if hist_data is None or len(hist_data) < 20:
        raise ValueError(f"예측 데이터 부족 (현재: {len(hist_data) if hist_data is not None else 0}개)")
    today = date.today()
    try:
        model = ARIMA(hist_data, order=(5,1,0)).fit()
        forecast = model.forecast(steps=forecast_days)
        hist_labels = [d.strftime('%m-%d') for d in hist_data.index]
        forecast_labels = [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]
        return {
            "labels": hist_labels + forecast_labels,
            "historical": np.round(hist_data.values, 2).tolist(), # 올바른 1차원 배열
            "forecast": [None] * len(hist_data) + np.round(forecast.values, 2).tolist()
        }
    except Exception as e: print(f"❌ 예측 모델 생성 중 오류: {e}"); raise e

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_marketaux_news(api_key):
    if not api_key: return []
    url = f"https://api.marketaux.com/v1/news/all?countries=us&filter_entities=true&language=en&limit=10&api_token={api_key}"
    try:
        response = requests.get(url, timeout=20); response.raise_for_status()
        print("✅ Marketaux 뉴스 수집 성공"); return response.json().get('data', [])
    except requests.exceptions.RequestException as e: print(f"❌ Marketaux API 호출 중 오류: {e}"); raise e

if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')
    print("--- 📰 뉴스 수집 및 AI 분석 시작 ---")
    marketaux_key = os.getenv("MARKETAUX_API_KEY"); articles = []
    try: articles = get_marketaux_news(marketaux_key)
    except Exception: print("최종적으로 Marketaux 뉴스 수집에 실패했습니다.")
    print(f"➡️ 총 {len(articles)}개의 최신 뉴스를 수집했습니다.")
    processed_articles, all_keywords, total_sentiment = [], [], 0.0
    if articles:
        for i, article in enumerate(articles):
            print(f"➡️  기사 {i+1}/{len(articles)} 분석 중...")
            content = article.get('description') or article.get('snippet', '')
            if content and len(content) > 100:
                ai_result = analyze_article_with_ai(content); article.update(ai_result); processed_articles.append(article)
                total_sentiment += ai_result.get('sentiment', 0.0)
                if ai_result.get('keywords'): all_keywords.extend(ai_result['keywords'])
    else: print("⚠️  수집된 뉴스가 없어 AI 분석을 건너뜁니다.")
    market_sentiment_score = total_sentiment / len(processed_articles) if processed_articles else 0.0
    trend_summary = generate_trend_summary_with_ai(all_keywords, market_sentiment_score); print("✅ 뉴스 분석 완료.")
    nasdaq_data, kospi_data, fx_data = None, None, None
    kis_app_key = os.getenv("KIS_APP_KEY"); kis_app_secret = os.getenv("KIS_APP_SECRET")
    if kis_app_key and kis_app_secret:
        try:
            kis_token = get_kis_token(kis_app_key, kis_app_secret)
            if kis_token:
                print("\n--- 📈 KIS API로 데이터 예측 시작 ---")
                try: nasdaq_hist = get_kis_daily_chart(kis_token, market="NAS", code="COMP"); nasdaq_data = process_chart_data(nasdaq_hist); print("✅ 나스닥 예측 완료.")
                except Exception as e: print(f"❌ 나스닥 KIS API 실패: {e}")
                time.sleep(1)
                try: kospi_hist = get_kis_daily_chart(kis_token, market="KSP", code="0001"); kospi_data = process_chart_data(kospi_hist); print("✅ 코스피 예측 완료.")
                except Exception as e: print(f"❌ 코스피 KIS API 실패: {e}")
        except Exception as e: print(f"❌ KIS API 전체 처리 중 오류: {e}")
    else: print("⚠️  KIS API 키가 설정되지 않았습니다.")
    try:
        print("\n--- 📈 Alpha Vantage API로 환율 예측 시작 ---")
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if av_key:
            cc = ForeignExchange(key=av_key, output_format='pandas')
            data, _ = cc.get_currency_exchange_daily(from_symbol='USD', to_symbol='KRW', outputsize='compact')
            fx_hist = data['4. close'].dropna().astype(float).sort_index().tail(30)
            fx_data = process_chart_data(fx_hist); print("✅ 환율 예측 완료.")
        else: print("⚠️  Alpha Vantage API 키가 설정되지 않았습니다.")
    except Exception as e: print(f"❌ Alpha Vantage 처리 중 오류: {e}")
    final_data = {"articles": processed_articles, "trend_summary": trend_summary, "market_sentiment_score": round(market_sentiment_score, 3), "nasdaq_data": nasdaq_data, "kospi_data": kospi_data, "fx_data": fx_data, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open('data/daily_data.json', 'w', encoding='utf-8') as f: json.dump(final_data, f, ensure_ascii=False, indent=4)
    print("\n🚀 모든 데이터가 'data/daily_data.json'에 성공적으로 저장되었습니다.")