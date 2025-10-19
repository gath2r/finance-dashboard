# app.py (서빙 로봇 최종 버전)

import os
import json
# ▼▼▼ sitemap.xml을 서빙하기 위해 send_from_directory를 import 합니다. ▼▼▼
from flask import Flask, render_template, send_from_directory
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

load_dotenv()
app = Flask(__name__)

# ▼▼▼ robots.txt와 sitemap.xml을 위한 라우트 추가 ▼▼▼
# 1. /robots.txt 경로 요청이 오면
@app.route('/robots.txt')
def robots_txt():
    # 'static' 폴더에 있는 robots.txt 파일을 반환합니다.
    return send_from_directory(os.path.join(app.root_path, 'static'), 'robots.txt')

# 2. /sitemap.xml 경로 요청이 오면
@app.route('/sitemap.xml')
def sitemap_xml():
    # 'static' 폴더에 있는 sitemap.xml 파일을 반환합니다.
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')

def read_json_data(file_path):
    """JSON 파일을 안전하게 읽어오는 함수"""
    try:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"⚠️ {file_path} 파일을 찾을 수 없어 기본 데이터를 사용합니다.")
        return None

@app.route('/')
def dashboard():
    # GitHub Actions가 미리 만들어둔 결과 파일 하나만 읽기
    all_data = read_json_data('data/daily_data.json')

    # 데이터가 없을 경우를 대비한 기본값 설정
    if all_data:
        return render_template(
            'index.html',
            articles=all_data.get("articles", []),
            trend_summary=all_data.get("trend_summary", {"title": "분석 중", "summary": "데이터를 준비하고 있습니다...", "keywords": []}),
            market_sentiment="", # 이 부분은 이제 큰 의미가 없음
            sentiment_score=all_data.get("market_sentiment_score", 0.0),
            nasdaq_data=all_data.get("nasdaq_data"),
            kospi_data=all_data.get("kospi_data"),
            fx_data=all_data.get("fx_data")
        )
    else:
        # 파일이 아예 없을 때 보여줄 최소한의 정보
        return render_template(
            'index.html', articles=[], 
            trend_summary={"title": "분석 데이터 없음", "summary": "데이터를 준비 중입니다. 잠시 후 새로고침해주세요.", "keywords": []},
            nasdaq_data=None, kospi_data=None, fx_data=None,
            sentiment_score=0.0, market_sentiment=""
        )

if __name__ == '__main__':
    # Render가 포트를 자동으로 할당할 수 있도록 host='0.0.0.0' 추가
    # debug=False로 설정해야 배포 환경에서 안정적으로 작동합니다.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)