# ============================================================
# 주식 차트 이미지 생성 스크립트
# ============================================================
# 목적: Alpha Vantage API로 주식 데이터를 받아와서
#       슬라이딩 윈도우 방식으로 캔들 차트 이미지를 생성하고,
#       머신러닝 학습용으로 train/test 폴더에 분할 저장
# ============================================================

import os
import time
import requests
import pandas as pd
import mplfinance as mpf  # 캔들 차트 생성 라이브러리
from dotenv import load_dotenv

# 1. 환경 설정 로드
# .env 파일에서 Alpha Vantage API 키를 불러옴
load_dotenv()
api_key = os.getenv("ALPHAVANTAGE_API_KEY")

# 2. 분석할 주식 심볼 리스트
symbols = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "NFLX", "NVDA", "META"]

# 3. 차트 이미지 저장 경로 생성
# train: 모델 학습용, test: 모델 평가용
os.makedirs("charts/train", exist_ok=True)
os.makedirs("charts/test", exist_ok=True)

# 4. 슬라이딩 윈도우 파라미터 설정
window_size = 100  # 한 차트에 표시할 일수 (100일 단위)
step = 20          # 슬라이딩 간격 (20일씩 이동하면서 차트 생성)
train_ratio = 0.8  # 데이터 분할 비율: 전체 데이터의 앞쪽 80%는 train, 뒤쪽 20%는 test

# ============================================================
# 함수 1: Alpha Vantage API로 일간 주식 데이터 가져오기
# ============================================================
def fetch_daily(symbol: str) -> pd.DataFrame | None:
    """
    Alpha Vantage API를 통해 특정 심볼의 일간 주식 데이터를 조회
    
    매개변수:
        symbol: 주식 심볼 (예: "AAPL", "MSFT")
    
    반환값:
        정제된 OHLCV 데이터프레임 또는 실패 시 None
    """
    # API 요청 URL 구성
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}"
        f"&apikey={api_key}&outputsize=full"  # outputsize=full: 전체 데이터 (20년+)
    )
    resp = requests.get(url)
    data = resp.json()

    # API 응답 검증
    if "Time Series (Daily)" not in data:
        print(f"⚠️ {symbol} 데이터 실패: {data}")
        return None

    # JSON에서 시계열 데이터 추출
    ts = data["Time Series (Daily)"]
    df = pd.DataFrame(ts).T  # 전치(Transpose)하여 날짜를 행으로 변환
    
    # 컬럼명을 표준 OHLCV 형식으로 변경
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })
    
    # 문자열 → 숫자형 변환 및 날짜 인덱스 설정
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)

    # ⚠️ 중요: 날짜를 오름차순으로 정렬 (과거 → 현재 순서)
    # 슬라이딩 윈도우가 시간 순서대로 진행되도록 하기 위함
    df = df.sort_index()

    # mplfinance 라이브러리가 요구하는 컬럼 순서로 재정렬
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df

# ============================================================
# 함수 2: 슬라이딩 윈도우 방식으로 차트 이미지 생성 및 저장
# ============================================================
def save_windows(df: pd.DataFrame, symbol: str):
    """
    데이터프레임을 슬라이딩 윈도우로 분할하여 캔들 차트 이미지 생성
    
    매개변수:
        df: OHLCV 데이터프레임
        symbol: 주식 심볼 (파일명에 사용)
    
    동작 흐름:
        1. 데이터를 window_size(100일) 단위로 분할
        2. step(20일) 간격으로 윈도우를 이동시키며 반복
        3. 각 윈도우별로 캔들 차트 이미지 생성
        4. 시간 순서에 따라 train/test 폴더에 분할 저장
    """
    n = len(df)
    
    # 데이터 충분성 검증
    if n < window_size:
        print(f"ℹ️ {symbol}: 데이터가 {n}행으로 {window_size} 미만이라 스킵")
        return

    # train/test 분할 기준 인덱스 계산
    # 예: 1000개 데이터 → split_idx = 800 (앞쪽 80%까지가 train)
    split_idx = int(n * train_ratio)

    # 슬라이딩 윈도우 순회
    # range(0, n - window_size, step): 0부터 시작해서 step(20)씩 증가
    for i in range(0, n - window_size, step):
        # 현재 윈도우 추출 (100일치 데이터)
        win = df.iloc[i:i+window_size]
        
        # 파일명에 사용할 날짜 범위 추출
        start_date = win.index[0].strftime("%Y%m%d")   # 시작일
        end_date   = win.index[-1].strftime("%Y%m%d")  # 종료일

        # train/test 분할 결정
        # 윈도우의 끝 인덱스가 split_idx 이하면 train, 초과면 test
        split = "train" if (i + window_size) <= split_idx else "test"

        # 저장 경로 생성 (예: charts/train/AAPL_20200101_20200515.png)
        save_path = f"charts/{split}/{symbol}_{start_date}_{end_date}.png"
        
        # mplfinance로 캔들 차트 생성 및 저장
        mpf.plot(
            win,
            type="candle",        # 캔들스틱 차트
            style="yahoo",        # 야후 파이낸스 스타일
            volume=True,          # 거래량 포함
            mav=(5, 20),         # 이동평균선: 5일, 20일
            savefig=save_path    # 파일로 저장
        )
    
    print(f"✅ {symbol}: 차트 저장 완료 (train/test 폴더)")

# ============================================================
# 메인 실행 루프: 각 심볼별 차트 생성 프로세스
# ============================================================
for symbol in symbols:
    # 1단계: API로 주식 데이터 가져오기
    df = fetch_daily(symbol)
    
    # 2단계: 데이터가 정상적으로 조회된 경우에만 차트 생성
    if df is not None:
        save_windows(df, symbol)
        
        # 3단계: API 호출 제한 대응을 위한 대기
        # Alpha Vantage 무료 플랜: 1분당 5회 호출 제한
        # → 12초 간격으로 호출하면 안전 (60초 / 5회 = 12초)
        time.sleep(12)

# ============================================================
# 실행 완료 후 결과물
# ============================================================
# - charts/train/ : 학습용 차트 이미지들 (전체 데이터의 80%)
# - charts/test/  : 평가용 차트 이미지들 (전체 데이터의 20%)
# - 파일명 형식: {심볼}_{시작일}_{종료일}.png
#   예) AAPL_20200101_20200515.png
# ============================================================