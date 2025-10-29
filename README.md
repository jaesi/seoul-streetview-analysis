# 🏙️ Seoul Street View Analysis

> 서울 거리뷰 이미지로 도시 활력도를 예측하는 머신러닝 프로젝트
> Predicting Urban Vitality Index from Seoul street view images using AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📚 목차 (Table of Contents)

- [프로젝트 소개](#-프로젝트-소개)
- [주요 기능](#-주요-기능)
- [빠른 시작](#-빠른-시작)
- [설치 가이드](#-설치-가이드)
- [사용 방법](#-사용-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [결과](#-결과)
- [FAQ](#-자주-묻는-질문-faq)
- [문제 해결](#-문제-해결)

---

## 🎯 프로젝트 소개

이 프로젝트는 **Google Street View 이미지**를 분석하여 서울의 **도시 활력도 지수(UVI)**를 예측합니다.

### 연구 대상 지역

- 🎨 **홍대 (Hongdae)**: 활기찬 젊음의 거리
- ☕ **샤로수길 (Syarosu-gil)**: 트렌디한 카페거리
- 🏘️ **쑥고개길 (Sookgogae-gil)**: 주거 상업 혼재 지역

### 작동 원리

```
📸 거리뷰 이미지 수집
    ↓
🤖 AI 이미지 분석 (건물, 도로, 녹지, 사람 등 인식)
    ↓
📊 머신러닝 모델 학습
    ↓
🎯 도시 활력도 예측
```

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🗺️ **자동 이미지 수집** | Google Maps API로 원하는 경로의 거리뷰 자동 다운로드 |
| 🧠 **AI 이미지 분석** | ResNet50 기반 딥러닝으로 도시 요소 자동 추출 |
| 📈 **다중 모델 비교** | 5가지 머신러닝 모델로 최적 성능 찾기 |
| 🎨 **시각화** | 분석 결과를 그래프와 차트로 한눈에 확인 |
| 🚀 **쉬운 실행** | 명령어 한 줄로 전체 파이프라인 실행 |

---

## ⚡ 빠른 시작

### 🎬 3분 안에 시작하기!

#### 1️⃣ 프로젝트 다운로드
```bash
git clone <repository-url>
cd seoul-streetview-analysis
```

#### 2️⃣ 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

#### 3️⃣ 바로 실행해보기
```bash
# 기존 데이터로 모델 학습 실행
python main.py --model
```

🎉 **완료!** 몇 분 후 결과를 확인하실 수 있습니다!

---

## 📦 설치 가이드

### 필수 요구사항

- 🐍 **Python 3.8 이상**
- 💾 **최소 2GB 디스크 공간**
- 🌐 **인터넷 연결** (패키지 설치 시)

### 상세 설치 단계

#### Step 1: Python 확인
```bash
python --version
# Python 3.8.0 이상이어야 합니다
```

Python이 없다면? 👉 [Python 공식 사이트](https://www.python.org/downloads/)에서 다운로드

#### Step 2: 프로젝트 받기
```bash
git clone <repository-url>
cd seoul-streetview-analysis
```

#### Step 3: 가상환경 생성 (권장)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

#### Step 4: 패키지 설치
```bash
pip install -r requirements.txt
```

설치 중 문제가 생기면? 👉 [문제 해결](#-문제-해결) 섹션 참고

#### Step 5: (선택사항) API 키 설정

새로운 거리뷰 이미지를 받으려면 Google Maps API 키가 필요합니다:

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
# GOOGLE_API_KEY=your_actual_api_key_here
```

💡 **팁**: API 키 없이도 기존 데이터로 프로젝트를 실행할 수 있습니다!

---

## 🚀 사용 방법

### 방법 1: 기존 데이터 사용 (가장 쉬움!) ⭐

이미 수집된 데이터가 있어서 바로 분석할 수 있어요!

```bash
# ML 모델 학습 및 평가
python main.py --model
```

**실행 결과:**
```
================================================================================
MODEL PERFORMANCE SUMMARY
================================================================================
Model                     Train R2     Test R2      Test RMSE    Test MAE
--------------------------------------------------------------------------------
Gradient Boosting         0.2064       0.0115       1.0787       0.8244
Random Forest             0.8359       -0.0340      1.1033       0.8419
...
```

### 방법 2: 새로운 이미지 수집

Google Maps API 키가 있다면 새로운 지역의 이미지를 받을 수 있어요:

```bash
python main.py --collect-images
```

### 방법 3: 샘플 데이터 생성

API 키가 없어도 테스트용 데이터를 만들 수 있어요:

```bash
python main.py --generate-data
```

### 방법 4: 전체 파이프라인 실행

모든 단계를 한번에 실행:

```bash
python main.py --full
```

### 💻 Python 코드로 직접 사용하기

```python
from src.modeling import UVIPredictor

# 예측 모델 생성
predictor = UVIPredictor()

# 데이터 준비
X_train, X_test, y_train, y_test = predictor.prepare_data(
    segmentation_csv="class_percentages.csv",
    uvi_excel="Urban_vitality_index.xlsx"
)

# 모델 학습
predictor.train_models(X_train, y_train)

# 평가
predictor.evaluate_models(X_train, X_test, y_train, y_test)
predictor.print_results()

# 새로운 데이터로 예측
predictions = predictor.predict(X_test, model_name='Gradient Boosting')
print(f"예측된 활력도: {predictions}")
```

---

## 📂 프로젝트 구조

```
seoul-streetview-analysis/
│
├── 📁 src/                          # 소스 코드
│   ├── __init__.py
│   ├── image_getter.py              # 📸 이미지 수집
│   ├── segmenter.py                 # 🤖 AI 이미지 분석
│   ├── modeling.py                  # 📊 ML 모델링
│   ├── generate_sample_data.py      # 🎲 샘플 데이터 생성
│   └── utils/                       # 🛠️ 유틸리티
│       ├── __init__.py
│       └── exceptions.py
│
├── 📁 config/                       # ⚙️ 설정 파일
│   └── settings.py                  # 프로젝트 설정
│
├── 📁 data/                         # 💾 데이터
│   ├── raw/                         # 원본 이미지
│   │   ├── hongdae/
│   │   ├── syarosu/
│   │   └── ssook/
│   ├── processed/                   # 처리된 데이터
│   └── models/                      # 학습된 모델
│
├── 📁 notebooks/                    # 📓 Jupyter 노트북 (아카이브)
│   ├── image_getter.ipynb
│   ├── segmenter.ipynb
│   └── modeling_prediction.ipynb
│
├── 📁 tests/                        # 🧪 테스트 코드
├── 📁 docs/                         # 📚 문서
├── 📁 examples/                     # 💡 예제 코드
├── 📁 scripts/                      # 🔧 실행 스크립트
│
├── 📄 main.py                       # 🚀 메인 실행 파일
├── 📄 requirements.txt              # 📦 필요한 패키지 목록
├── 📄 .env.example                  # 🔑 환경변수 템플릿
└── 📄 README.md                     # 📖 이 파일!
```

---

## 🎨 파이프라인 상세 설명

### 1️⃣ 이미지 수집 (image_getter.py)

Google Maps API를 사용하여 거리뷰 이미지를 자동으로 수집합니다.

**특징:**
- ✅ 원하는 경로를 따라 자동 수집
- ✅ 4방향(동서남북) 촬영
- ✅ 좌표 기반 정확한 위치 지정

**예제 코드:**
```python
from src.image_getter import StreetViewImageGetter

getter = StreetViewImageGetter(api_key="YOUR_API_KEY")
getter.fetch_images_along_path(
    start_lat=37.554197,  # 시작 위도
    start_lon=126.922500,  # 시작 경도
    end_lat=37.550833,     # 끝 위도
    end_lon=126.921323,    # 끝 경도
    num_points=40,         # 샘플링 포인트 수
    output_folder="data/raw/hongdae"
)
```

### 2️⃣ 이미지 분석 (segmenter.py)

AI가 이미지에서 도시 구성 요소를 자동으로 인식합니다.

**인식 가능한 요소:**
- 🏢 건물
- 🛣️ 도로
- 🌳 녹지 공간
- ☁️ 하늘
- 🚶 보행자
- 🪧 기타 도시 시설물

**예제 코드:**
```python
from src.segmenter import StreetViewSegmenter

segmenter = StreetViewSegmenter()

# 단일 이미지 분석
percentages = segmenter.segment_image("path/to/image.jpg")
print(f"건물 비율: {percentages['building']:.1f}%")

# 폴더 전체 분석
df = segmenter.process_folder(
    input_folder="data/raw/hongdae",
    output_csv="data/processed/hongdae_segmentation.csv"
)
```

### 3️⃣ 머신러닝 모델링 (modeling.py)

5가지 머신러닝 모델로 도시 활력도를 예측합니다.

**사용 모델:**
- 🌲 Decision Tree (결정 트리)
- 🌳 Random Forest (랜덤 포레스트)
- 📈 Gradient Boosting (그래디언트 부스팅) - **Best!**
- 🎯 Support Vector Machine (SVM)
- 👥 K-Nearest Neighbors (KNN)

---

## 📊 결과

### 모델 성능 비교

| 모델 | Train R² | Test R² | Test RMSE | Test MAE | 평가 |
|------|----------|---------|-----------|----------|------|
| Gradient Boosting | 0.21 | 0.01 | 1.08 | 0.82 | ⭐⭐⭐ |
| Random Forest | 0.84 | -0.03 | 1.10 | 0.84 | ⭐⭐ |
| Decision Tree | 0.03 | -0.10 | 1.40 | 1.20 | ⭐ |

**🔍 분석:**
- Gradient Boosting이 가장 안정적인 성능을 보입니다
- Random Forest는 과적합(overfitting) 경향이 있습니다
- 전반적으로 R² 값이 낮은 것은 시각적 특징만으로는 도시 활력도를 완전히 설명하기 어렵다는 것을 의미합니다

**💡 개선 방안:**
- 유동인구 데이터 추가
- 상권 정보 통합
- 시간대별 데이터 수집
- 더 많은 지역 데이터 확보

### 연구 지역별 특성

#### 🎨 홍대 (Hongdae)
- **좌표**: 37.554197, 126.922500 → 37.550833, 126.921323
- **이미지**: 160장 (40개 지점 × 4방향)
- **특징**: 높은 활력도, 많은 보행자, 다양한 상가

#### ☕ 샤로수길 (Syarosu-gil)
- **좌표**: 37.479241, 126.952545 → 37.479476, 126.944457
- **이미지**: 80장 (20개 지점 × 4방향)
- **특징**: 중간 활력도, 카페·부티크 중심

#### 🏘️ 쑥고개길 (Sookgogae-gil)
- **좌표**: 37.478701, 126.952144 → 37.479476, 126.944457
- **이미지**: 80장 (20개 지점 × 4방향)
- **특징**: 낮은 활력도, 주거 지역 특성

---

## 🛠️ 기술 스택

### 핵심 기술

| 분야 | 기술 | 용도 |
|------|------|------|
| 🐍 **언어** | Python 3.8+ | 전체 프로젝트 |
| 🤖 **딥러닝** | PyTorch, torchvision | 이미지 세그멘테이션 |
| 📊 **머신러닝** | scikit-learn | 예측 모델 |
| 📈 **데이터** | pandas, numpy | 데이터 처리 및 분석 |
| 📉 **시각화** | matplotlib | 결과 시각화 |
| 🗺️ **API** | Google Maps API | 거리뷰 이미지 수집 |
| 🖼️ **이미지** | Pillow | 이미지 처리 |

---

## ❓ 자주 묻는 질문 (FAQ)

### Q1: Google Maps API 키가 꼭 필요한가요?

**A:** 아니요! 프로젝트에 이미 수집된 데이터가 포함되어 있어서 API 키 없이도 모델 학습과 분석을 할 수 있습니다. 새로운 지역의 이미지를 수집하고 싶을 때만 API 키가 필요합니다.

### Q2: 얼마나 시간이 걸리나요?

**A:**
- 패키지 설치: 5-10분
- 모델 학습: 2-5분
- 이미지 수집 (API 사용 시): 지역당 5-10분

### Q3: 다른 지역도 분석할 수 있나요?

**A:** 네! `config/settings.py` 파일에서 새로운 지역의 좌표를 추가하면 됩니다:

```python
LOCATIONS["gangnam"] = {
    "start_coords": (37.498, 127.027),
    "end_coords": (37.502, 127.030),
    "num_points": 30
}
```

### Q4: 왜 모델 성능(R²)이 낮나요?

**A:** 거리뷰 이미지의 시각적 특징만으로는 도시 활력도를 완전히 예측하기 어렵습니다. 실제 도시 활력도는 유동인구, 시간대, 상권 밀도 등 다양한 요소의 영향을 받기 때문입니다.

### Q5: GPU가 필요한가요?

**A:** 필수는 아니지만 권장됩니다. CPU만으로도 실행 가능하며, 다만 이미지 세그멘테이션이 조금 더 오래 걸립니다.

### Q6: 에러가 발생하면 어떻게 하나요?

**A:** [문제 해결](#-문제-해결) 섹션을 참고하시거나, GitHub Issues에 질문을 남겨주세요!

---

## 🔧 문제 해결

### 문제 1: `ModuleNotFoundError: No module named 'torch'`

**해결방법:**
```bash
pip install torch torchvision
```

### 문제 2: `Permission denied` 에러

**해결방법:**
```bash
# 가상환경을 사용하거나
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 또는
pip install --user -r requirements.txt
```

### 문제 3: 메모리 부족 에러

**해결방법:**
- 한 번에 처리하는 이미지 수를 줄이세요
- `config/settings.py`에서 `batch_size` 값을 줄이세요

### 문제 4: Google API 할당량 초과

**해결방법:**
- 샘플 데이터 생성 기능을 사용하세요: `python main.py --generate-data`
- 또는 Google Cloud Console에서 할당량을 늘리세요

### 문제 5: 한글이 깨져요

**해결방법:**
```python
# matplotlib 한글 폰트 설정
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'  # Mac/Linux
# 또는
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
```

---

## 🚀 향후 개선 계획

- [ ] 🕐 시간대별 데이터 수집 및 분석
- [ ] 📍 POI(관심지점) 데이터 통합
- [ ] 👥 실제 유동인구 데이터 연계
- [ ] 🗺️ 웹 인터페이스 개발
- [ ] 📱 모바일 앱 버전
- [ ] 🌏 다른 도시로 확장

---

## 👥 기여하기

이 프로젝트는 교육 목적의 프로젝트입니다. 개선 제안이나 버그 리포트는 언제나 환영합니다!

**기여 방법:**
1. 이 저장소를 Fork 하세요
2. 새로운 브랜치를 만드세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push 하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 열어주세요

---

## 📜 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. Google Maps API 사용 시 [서비스 약관](https://cloud.google.com/maps-platform/terms)을 준수해주세요.

---

## 🙏 감사의 말

- **Google Maps Platform**: Street View API 제공
- **PyTorch Team**: torchvision 모델 제공
- **서울특별시**: 도시 활력도 데이터

---

## 📞 연락처

궁금한 점이나 문의사항이 있으시면 GitHub Issues에 남겨주세요!

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐**

Made with ❤️ for Urban Analytics

**프로젝트 타입**: 학술 연구
**분야**: 도시 분석, 컴퓨터 비전, 머신러닝
**상태**: ✅ 완료

</div>
