# database_setup.py
import sqlite3

# 데이터베이스 연결 (파일이 없으면 새로 생성됨)
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# 1. AI의 예측을 저장할 테이블
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date DATE UNIQUE NOT NULL,
    market_sentiment_score REAL NOT NULL,
    predicted_trend TEXT NOT NULL -- '상승' 또는 '하락'
)
''')

# 2. 실제 시장 결과를 저장할 테이블
cursor.execute('''
CREATE TABLE IF NOT EXISTS actuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actual_date DATE UNIQUE NOT NULL,
    actual_trend TEXT NOT NULL -- '상승' 또는 '하락'
)
''')

# 변경사항 저장 및 연결 종료
conn.commit()
conn.close()

print("데이터베이스 테이블이 성공적으로 생성되었습니다.")