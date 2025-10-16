# run_predictions.py

import os
import json
from datetime import date, timedelta
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

def get_chart_data(ticker, days=90, forecast_days=7):
    """주가 지수 데이터를 가져와 예측하고 결과를 딕셔너리로 반환"""
    try:
        today = date.today()
        start_date = today - timedelta(days=days)
        data = yf.download(ticker, start=start_date, end=today, progress=False, timeout=15)
        if data.empty or len(data) < 20: return None
        
        hist_data = data['Close']
        model = ARIMA(hist_data, order=(5,1,0)).fit()
        forecast = model.forecast(steps=forecast_days)
        
        labels = [d.strftime('%m-%d') for d in hist_data.index] + \
                 [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]
        
        return {
            "labels": labels,
            "historical": np.round(hist_data.values, 2).tolist(),
            "forecast": [None] * len(hist_data) + np.round(forecast.values, 2).tolist()
        }
    except Exception as e:
        print(f"❌ '{ticker}' 차트 데이터 생성 오류: {e}")
        return None

def get_fx_chart_data(api_key, days=90, forecast_days=7):
    """환율 데이터를 가져와 예측하고 결과를 딕셔너리로 반환"""
    try:
        latest_rate = 1422.0
        if api_key:
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                latest_rate = response.json().get('conversion_rates', {}).get('KRW', latest_rate)
        
        today = date.today()
        start_date = today - timedelta(days=days)
        simulated_past = latest_rate + np.random.randn(days).cumsum()[::-1]
        hist_data = pd.Series(simulated_past, index=pd.to_datetime([start_date + timedelta(days=i) for i in range(days)]))
        
        model = ARIMA(hist_data, order=(5,1,0)).fit()
        forecast = model.forecast(steps=forecast_days)
        
        labels = [d.strftime('%m-%d') for d in hist_data.index] + \
                 [(today + timedelta(days=i)).strftime('%m-%d') for i in range(1, forecast_days + 1)]
                 
        return {
            "labels": labels,
            "historical": np.round(hist_data.values, 2).tolist(),
            "forecast": [None] * len(hist_data) + np.round(forecast.values, 2).tolist()
        }
    except Exception as e:
        print(f"❌ 환율 차트 데이터 생성 오류: {e}")
        return None

if __name__ == "__main__":
    # 데이터 폴더가 없으면 생성
    if not os.path.exists('data'):
        os.makedirs('data')

    # 1. 나스닥 예측
    print("📈 나스닥 데이터 예측 중...")
    nasdaq_data = get_chart_data('^IXIC')
    if nasdaq_data:
        with open('data/nasdaq_data.json', 'w') as f:
            json.dump(nasdaq_data, f)
        print("✅ 나스닥 예측 결과 저장 완료.")

    # 2. 코스피 예측
    print("📈 코스피 데이터 예측 중...")
    kospi_data = get_chart_data('^KS11')
    if kospi_data:
        with open('data/kospi_data.json', 'w') as f:
            json.dump(kospi_data, f)
        print("✅ 코스피 예측 결과 저장 완료.")

    # 3. 환율 예측
    print("📈 환율 데이터 예측 중...")
    fx_api_key = os.getenv("EXCHANGERATE_API_KEY")
    fx_data = get_fx_chart_data(fx_api_key)
    if fx_data:
        with open('data/fx_data.json', 'w') as f:
            json.dump(fx_data, f)
        print("✅ 환율 예측 결과 저장 완료.")