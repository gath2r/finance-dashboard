# trainer.py
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    """DB에서 데이터를 가져와 AI 모델을 학습시키고 파일로 저장합니다."""
    conn = sqlite3.connect('database.db')
    
    # DB에서 예측과 실제 결과를 날짜 기준으로 합쳐서 가져옴
    query = """
    SELECT
        p.market_sentiment_score,
        a.actual_trend
    FROM predictions p
    JOIN actuals a ON p.prediction_date = a.actual_date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) < 20: # 최소 20개의 데이터가 있어야 학습 의미가 있음
        print(f"학습 데이터가 부족합니다. 현재 데이터 {len(df)}개. 학습을 건너뜁니다.")
        return

    # 1. 특성(X)과 라벨(y) 분리
    X = df[['market_sentiment_score']]
    y = df['actual_trend']

    # 2. 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 모델 선택 및 학습 (로지스틱 회귀)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. 모델 평가
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"모델 재학습 완료! 새 모델의 예측 정확도: {accuracy:.2f}")

    # 5. 똑똑해진 모델을 파일로 저장
    joblib.dump(model, 'market_predictor.pkl')
    print("새로운 AI 모델 'market_predictor.pkl'을 저장했습니다.")

if __name__ == "__main__":
    train_and_save_model()