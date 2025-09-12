# 예측 모델링 시스템 환경 설정 가이드

K-Pop Dashboard 예측 모델링 시스템을 위한 환경 설정 안내서입니다.

## 1. 가상 환경 설정 (권장)

시스템 환경 보호를 위해 가상 환경 사용을 권장합니다.

### macOS/Linux
```bash
# 가상 환경 생성
python3 -m venv venv_kpop

# 가상 환경 활성화
source venv_kpop/bin/activate

# 종속성 설치
pip install -r requirements_postgresql.txt

# 또는 기본 requirements
pip install -r requirements.txt
```

### Windows
```cmd
# 가상 환경 생성
python -m venv venv_kpop

# 가상 환경 활성화
venv_kpop\Scripts\activate

# 종속성 설치
pip install -r requirements_postgresql.txt
```

## 2. 필수 라이브러리

예측 모델링 시스템에 필요한 핵심 라이브러리들:

### 머신러닝 & 통계
- `scikit-learn>=1.3.0` - 머신러닝 모델 (Random Forest, Linear Regression 등)
- `statsmodels>=0.14.0` - 시계열 분석 (ARIMA, Exponential Smoothing)
- `numpy>=1.24.0` - 수치 연산
- `scipy>=1.11.0` - 과학적 연산

### 시각화 & 데이터 처리
- `matplotlib>=3.7.0` - 기본 플롯팅
- `seaborn>=0.12.0` - 통계적 시각화
- `pandas` - 데이터 처리 (이미 포함)

### 고급 시계열 예측 (선택사항)
- `prophet>=1.1.0` - Facebook의 시계열 예측 라이브러리
- `pmdarima>=2.0.0` - 자동 ARIMA 모델 선택

## 3. 환경 검증

설치 후 다음 명령으로 환경을 검증하세요:

```python
# 기본 라이브러리 테스트
python3 -c "
import sklearn
import statsmodels
import numpy
import scipy
import matplotlib
import seaborn
print('✅ 모든 필수 라이브러리가 설치되었습니다!')
"

# 예측 모델링 모듈 테스트
python3 -c "
from kpop_dashboard.analytics.prediction_models import PredictiveModelingEngine
print('✅ 예측 모델링 엔진이 성공적으로 로드되었습니다!')
"
```

## 4. 메모리 및 성능 최적화

### 시스템 요구사항
- **최소 RAM**: 4GB (8GB 권장)
- **디스크 공간**: 1GB (모델 캐시 포함)
- **Python 버전**: 3.8 이상

### 성능 최적화 설정
```python
# 환경 변수 설정 (선택사항)
import os
os.environ['OMP_NUM_THREADS'] = '4'  # CPU 코어 수에 맞게 조정
os.environ['MKL_NUM_THREADS'] = '4'
```

## 5. 트러블슈팅

### 일반적인 문제 해결

#### 1. scikit-learn 설치 오류
```bash
# macOS에서 Homebrew 사용
brew install python-tk
pip install --upgrade setuptools wheel
pip install scikit-learn
```

#### 2. statsmodels 종속성 오류
```bash
pip install --upgrade cython
pip install statsmodels
```

#### 3. matplotlib 백엔드 오류
```python
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
```

#### 4. 메모리 부족 오류
```python
# 모델 설정에서 메모리 사용량 제한
config = ModelConfig()
config.random_forest_params['n_estimators'] = 50  # 기본값 100에서 감소
config.random_forest_params['max_depth'] = 5      # 기본값 10에서 감소
```

## 6. 개발 환경 설정

### IDE 설정 (VS Code 예시)
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv_kpop/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

### Jupyter Notebook 설정 (선택사항)
```bash
pip install jupyter
pip install ipykernel

# 가상 환경을 Jupyter 커널로 등록
python -m ipykernel install --user --name=venv_kpop
```

## 7. 예측 모델링 기능 확인

환경 설정 완료 후, 다음과 같이 기능을 확인할 수 있습니다:

```python
from kpop_dashboard.analytics.prediction_models import (
    PredictiveModelingEngine, 
    ModelType, 
    PredictionHorizon
)

# 엔진 초기화
engine = PredictiveModelingEngine()

# 사용 가능한 모델 타입 확인
print("사용 가능한 모델:")
for model in ModelType:
    print(f"  - {model.value}")

# 예측 기간 옵션 확인
print("\\n예측 기간 옵션:")
for horizon in PredictionHorizon:
    print(f"  - {horizon.value}")
```

## 8. 다음 단계

환경 설정 완료 후:
1. 🔮 **예측 모델 구현** - 시계열 예측 알고리즘 개발
2. 📈 **성장 예측 모델** - 아티스트별 성장률 예측
3. 🎯 **마일스톤 예측** - 구독자/재생수 목표 달성 시기 예측
4. 📊 **계절성 분석** - 주간/월간 패턴 분석
5. 🚨 **예측 기반 알럿** - 이상치 예측 및 조기 경고

## 문의 및 지원

환경 설정 중 문제가 발생하면:
1. 가상 환경 재생성
2. 최신 pip 업그레이드: `pip install --upgrade pip`
3. 종속성 개별 설치
4. 시스템별 특정 해결책 적용

---
**최종 업데이트**: 2025-09-08  
**작성자**: Backend Development Team