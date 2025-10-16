# collector.py
import yfinance as yf
import sqlite3
from datetime import date, timedelta

def get_yesterday_market_trend():
    """어제의 S&P 500 지수 등락을 가져옵니다."""
    yesterday = date.today() - timedelta(days=1)
    day_before = yesterday - timedelta(days=4) # 주말을 고려하여 넉넉하게

    try:
        # S&P 500 티커인 ^GSPC 데이터를 가져옴
        data = yf.download('^GSPC', start=day_before, end=yesterday + timedelta(days=1))
        
        # 어제 날짜의 종가와 그 전날 종가 비교
        yesterday_close = data.loc[yesterday.strftime('%Y-%m-%d')]['Close']
        day_before_close = data.iloc[-2]['Close']

        if yesterday_close > day_before_close:
            return yesterday, '상승'
        else:
            return yesterday, '하락'
    except Exception as e:
        print(f"시장 데이터 수집 실패: {e}")
        return None, None

def save_actual_trend(actual_date, trend):
    """실제 결과를 데이터베이스에 저장합니다."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    try:
        # 데이터가 이미 있는지 확인 후 삽입 (중복 방지)
        cursor.execute("INSERT OR IGNORE INTO actuals (actual_date, actual_trend) VALUES (?, ?)", 
                       (actual_date.strftime('%Y-%m-%d'), trend))
        conn.commit()
        print(f"{actual_date}의 실제 시장 결과 '{trend}'를 저장했습니다.")
    except Exception as e:
        print(f"DB 저장 실패: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    yesterday, actual_trend = get_yesterday_market_trend()
    if actual_trend:
        save_actual_trend(yesterday, actual_trend)