# Seoul Street View Analysis

> 서울 거리뷰 이미지로 도시 활력도를 예측하는 머신러닝 프로젝트
> Predicting the Urban Vitality Index of Seoul through Google Street View imagery

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Project Overview
이 프로젝트는 **Google Street View 이미지**를 분석하여 서울의 **도시 활력도 지수(UVI)**를 예측합니다. 이미 수집된 데이터를 활용하거나 Google Maps API로 새로운 이미지를 수집해 도시 환경을 정량적으로 평가할 수 있습니다.

### Study Areas
- **홍대 (Hongdae)** — 활력도와 유동 인구가 높은 상업 지역
- **샤로수길 (Syarosu-gil)** — 카페와 소규모 상권이 밀집한 지역
- **쑥고개길 (Sookgogae-gil)** — 주거와 상업이 혼재된 생활권

### Workflow
1. 거리뷰 이미지 수집
2. 이미지 세그멘테이션 및 요소 비율 산출
3. 데이터 전처리 및 피처 엔지니어링
4. 머신러닝 모델 학습과 평가

---

## Features
| Feature | Description |
|---------|-------------|
| Automated image collection | Google Maps API를 통해 지정한 경로의 Street View 이미지를 일괄 수집합니다. |
| Semantic segmentation | 도시 구성 요소(건물, 도로, 녹지 등)를 픽셀 단위로 추정합니다. |
| Multiple ML models | Gradient Boosting, Random Forest 등 다양한 회귀 모델을 비교합니다. |
| Visualization utilities | 모델 결과와 도시 특성을 그래프로 확인할 수 있습니다. |
| Scriptable pipeline | `main.py`로 전체 파이프라인을 순차 실행할 수 있습니다. |

---

## Quick Start
```bash
git clone <repository-url>
cd seoul-streetview-analysis
pip install -r requirements.txt
python main.py --model
```

---

## Installation
- Python 3.8 이상 환경을 권장합니다.
- 가상환경 사용 시 다음 예시를 참고하세요.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

새로운 이미지를 수집하려면 Google Maps API 키를 `.env` 파일에 설정하세요.

```bash
cp .env.example .env
# .env 파일을 열어 GOOGLE_API_KEY 값을 입력합니다.
```

---

## Usage
### 기존 데이터로 모델 학습
```bash
python main.py --model
```

### 새로운 이미지 수집
```bash
python main.py --collect-images
```

### 샘플 데이터 생성
```bash
python main.py --generate-data
```

### 전체 파이프라인 실행
```bash
python main.py --full
```

### Python API 예시
```python
from src.seoul_streetview.modeling import UVIPredictor

predictor = UVIPredictor()
X_train, X_test, y_train, y_test = predictor.prepare_data(
    segmentation_csv="class_percentages.csv",
    uvi_excel="Urban_vitality_index.xlsx"
)

predictor.train_models(X_train, y_train)
predictor.evaluate_models(X_train, X_test, y_train, y_test)
```

---

## Project Structure
```
seoul-streetview-analysis/
├── config/
├── data/
├── examples/
├── notebooks/
├── src/
├── streetview_data/
│   ├── streetview/
│   ├── streetview_assignment/
│   ├── streetview_assignment_is/
│   ├── streetview_assignment_od/
│   ├── streetview_images_total/
│   └── streetview_processed/
├── test_images/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

> Street View 이미지와 세그멘테이션 결과 폴더는 모두 `streetview_data/` 아래로 이동했습니다.

---

## Results
### Model Performance
| Model | Train R² | Test R² | Test RMSE | Test MAE |
|-------|----------|---------|-----------|----------|
| Gradient Boosting | 0.21 | 0.01 | 1.08 | 0.82 |
| Random Forest | 0.84 | -0.03 | 1.10 | 0.84 |
| Decision Tree | 0.03 | -0.10 | 1.40 | 1.20 |

Gradient Boosting이 가장 안정적인 테스트 성능을 보이며, 시각적 피처만으로는 활력도를 완전히 설명하기 어렵다는 점을 시사합니다.

### Area Highlights
- **홍대**: 40개 지점 × 4방향, 활력도와 보행자 밀도가 높음
- **샤로수길**: 20개 지점 × 4방향, 카페 중심의 중간 활력도
- **쑥고개길**: 20개 지점 × 4방향, 주거 지역 특성이 강함

---

## FAQ
**Google Maps API 키가 꼭 필요한가요?**
: 기본 데이터가 포함되어 있으므로 필수는 아닙니다. 새로운 지역을 수집할 때만 필요합니다.

**모델 학습에는 얼마나 걸리나요?**
: 패키지 설치 약 5분, 모델 학습 2~5분 정도가 소요됩니다.

**다른 지역도 분석할 수 있나요?**
: `config/settings.py`에 좌표를 추가하고 이미지를 수집하면 새로운 지역을 평가할 수 있습니다.

**GPU가 필요한가요?**
: CPU만으로도 실행 가능하지만, 세그멘테이션 속도는 GPU가 더 빠릅니다.

---

## Troubleshooting
- `ModuleNotFoundError: No module named 'torch'`
  - `pip install torch torchvision`
- 권한 오류가 발생할 때
  - 가상환경을 사용하거나 `pip install --user -r requirements.txt`
- Google API 할당량 초과
  - `python main.py --generate-data`로 샘플 데이터를 사용하거나 할당량을 조정하세요.
- matplotlib에서 한글이 깨질 때
  - 적절한 한글 폰트를 설정하세요 (`NanumGothic`, `Malgun Gothic` 등).

---

## Roadmap
- 시간대별 데이터 수집 및 분석
- POI(관심지점) 데이터 통합
- 실제 유동인구 데이터 연계
- 웹 인터페이스 또는 경량 대시보드 개발
- 다른 도시로의 확장 적용

---

## Contributing
1. 저장소를 포크합니다.
2. 새로운 브랜치를 만듭니다 (`git checkout -b feature/name`).
3. 변경 사항을 커밋합니다 (`git commit -m "Add feature"`).
4. 브랜치에 푸시하고 Pull Request를 생성합니다.

---

## License
이 프로젝트는 MIT License를 따릅니다. Google Maps API를 사용할 때는 [서비스 약관](https://cloud.google.com/maps-platform/terms)을 준수하세요.

---

## Acknowledgements
- Google Maps Platform — Street View API 제공
- PyTorch Team — torchvision 모델 제공
- 서울특별시 — 도시 활력도 데이터 제공

---

## Contact
질문이나 제안은 GitHub Issues에 남겨주세요. 프로젝트가 도움이 되었다면 ⭐️ Star로 응원 부탁드립니다.

---

**프로젝트 상태:** 완료
