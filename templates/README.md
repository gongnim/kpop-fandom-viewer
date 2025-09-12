# K-POP Analytics Template System

K-POP 엔터테인먼트 분석 대시보드를 위한 종합적인 템플릿 시스템입니다.

## 🎯 주요 기능

### 📄 템플릿 시스템
- **HTML 리포트**: 주간/월간/분기별 리포트 템플릿
- **PDF 생성**: 전문적인 PDF 리포트 자동 생성
- **반응형 디자인**: 모든 디바이스에서 최적화된 표시
- **다국어 지원**: 한국어/영어 혼합 콘텐츠 최적화

### 🎨 K-POP 브랜딩
- **테마 시스템**: 4가지 K-POP 테마 (Vibrant, Neon, Pastel, Luxury)
- **색상 팔레트**: 브랜드 일관성을 위한 체계적 색상 관리
- **타이포그래피**: 한글/영문 최적화 폰트 시스템
- **아이콘/그래픽**: K-POP 특화 시각적 요소

### 📊 차트 생성
- **자동 차트**: Plotly 기반 인터랙티브 차트
- **다양한 타입**: 선형/막대/파이/버블/히트맵 차트
- **실시간 업데이트**: 데이터 변경시 자동 차트 갱신
- **내보내기**: PNG/SVG/PDF 형식 지원

### ⚡ 성능 최적화
- **템플릿 캐싱**: LRU 캐시로 렌더링 성능 향상
- **배치 렌더링**: 여러 템플릿 병렬 처리
- **비동기 처리**: 대용량 데이터 비동기 렌더링

### 🔍 미리보기 시스템
- **실시간 미리보기**: 템플릿 변경사항 즉시 확인
- **테마 전환**: 실시간 테마 변경 및 비교
- **인터랙티브 편집**: 데이터 실시간 편집 및 적용

## 🏗️ 시스템 아키텍처

```
templates/
├── reports/           # 리포트 템플릿
│   ├── html/         # HTML 템플릿
│   ├── pdf/          # PDF 생성 시스템
│   └── chart_integration.py  # 차트 통합
├── branding/         # 브랜딩 시스템
│   ├── brand_system.py      # 브랜드 정의
│   └── style_generator.py   # 스타일 생성기
├── rendering/        # 렌더링 엔진
│   ├── template_engine.py   # 메인 엔진
│   └── preview_dashboard.py # 미리보기 대시보드
└── demo/            # 데모 및 예제
    └── template_demo.py     # 종합 데모
```

## 🚀 빠른 시작

### 1. 기본 템플릿 렌더링

```python
from templates.rendering import create_template_engine, RenderConfig, TemplateContext

# 템플릿 엔진 생성
engine = create_template_engine('templates/')

# 설정
config = RenderConfig(
    template_name='reports/html/weekly_report.html',
    output_format='html',
    theme='kpop_vibrant',
    include_charts=True
)

# 데이터 컨텍스트
context = TemplateContext(
    data={
        'period_start': '2024-01-01',
        'period_end': '2024-01-07',
        'summary_cards': [
            {'title': '총 팔로워', 'value': '2.8M', 'change': 5.2}
        ]
    },
    metadata={'report_type': 'weekly'}
)

# 렌더링
result = engine.render_template(config, context)
if result.success:
    print(f"렌더링 성공: {len(result.content):,} bytes")
else:
    print(f"오류: {result.errors}")
```

### 2. 미리보기 서버 시작

```python
from templates.rendering import start_preview_server

# 미리보기 서버 시작 (포트 8080)
server = start_preview_server('templates/', port=8080)

# 브라우저에서 http://localhost:8080 접속
print("미리보기 서버 실행 중...")
input("Enter를 눌러 서버 중지...")
server.stop()
```

### 3. Streamlit 대시보드

```bash
# 필요한 패키지 설치
pip install streamlit plotly

# 대시보드 실행
streamlit run templates/rendering/preview_dashboard.py
```

### 4. 테마 시스템 사용

```python
from templates.branding import KPOPBrandSystem, generate_theme_css

brand_system = KPOPBrandSystem()

# 사용 가능한 테마
themes = brand_system.themes.keys()
print(f"Available themes: {list(themes)}")

# 테마 정보 가져오기
theme_data = brand_system.get_theme('kpop_vibrant')
colors = theme_data['colors']

# CSS 생성
css = generate_theme_css('kpop_vibrant')
print(f"Generated CSS: {len(css)} characters")
```

### 5. 차트 생성

```python
from templates.reports.chart_integration import generate_weekly_charts

# 샘플 데이터
data = {
    'daily_scores': [85, 92, 88, 95, 102, 98, 105],
    'platform_data': {
        'platforms': ['YouTube', 'Instagram', 'TikTok', 'Spotify'],
        'followers': [2800000, 1900000, 3200000, 1100000]
    }
}

# 차트 생성
charts = generate_weekly_charts(data)
print(f"Generated {len(charts)} charts")

# 각 차트는 HTML, JSON, Base64 형식 지원
for name, chart_data in charts.items():
    print(f"Chart '{name}': {list(chart_data.keys())}")
```

## 📊 템플릿 구조

### HTML 템플릿 예제

```html
{% extends "base.html" %}

{% block title %}K-POP Analytics 주간 리포트{% endblock %}

{% block content %}
<section class="section">
    <h2 class="section-title">경영진 요약</h2>
    
    {% if summary_cards %}
    <div class="metric-cards">
        {% for card in summary_cards %}
        <div class="metric-card {{ card.status }}">
            <div class="metric-value">{{ card.value }}</div>
            <div class="metric-label">{{ card.title }}</div>
            <div class="metric-change">
                {{ card.change|percentage }}
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</section>
{% endblock %}
```

### 차트 통합

```html
<!-- 자동 생성된 차트 삽입 -->
<div class="chart-container">
    {{ charts.performance_trend.html|safe }}
</div>

<!-- 인터랙티브 컨트롤 -->
<div class="chart-controls">
    <button onclick="exportChart('performance_trend')">내보내기</button>
    <button onclick="toggleChartType('performance_trend')">차트 타입 변경</button>
</div>
```

## 🎨 테마 커스터마이제이션

### 1. 새 테마 추가

```python
from templates.branding import ColorScheme

# 새 컬러 스키마 정의
custom_scheme = ColorScheme(
    primary='#ff6b9d',      # 커스텀 핑크
    secondary='#4ecdc4',    # 터콰이즈
    accent='#45b7d1',       # 하늘색
    background='#ffffff',   # 화이트
    surface='#f8f9fa',      # 라이트 그레이
    text_primary='#2d3748', # 다크 그레이
    text_secondary='#718096', # 미디엄 그레이
    success='#48bb78',      # 그린
    warning='#ed8936',      # 오렌지
    error='#e53e3e',        # 레드
    info='#3182ce'          # 블루
)

# 브랜드 시스템에 추가
brand_system = KPOPBrandSystem()
brand_system.color_schemes['my_custom'] = custom_scheme
```

### 2. CSS 변수 생성

```python
from templates.branding import KPOPStyleGenerator

generator = KPOPStyleGenerator()

# CSS 변수 생성
css_vars = generator.generate_css_variables('kpop_vibrant')

# 컴포넌트 스타일 생성
button_styles = generator.create_component_styles('button', 'kpop_vibrant')

# 전체 스타일시트 내보내기
generator.export_styles('./output', 'kpop_vibrant')
```

## 📈 성능 최적화

### 캐시 설정

```python
from templates.rendering import TemplateCache

# 캐시 설정 (최대 100개 아이템, 1시간 TTL)
cache = TemplateCache(max_size=100, ttl=3600)

# 캐시 통계 확인
stats = engine.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

### 배치 렌더링

```python
# 여러 템플릿을 병렬로 렌더링
batch_requests = []
for theme in ['kpop_vibrant', 'kpop_neon', 'kpop_pastel']:
    config = RenderConfig(template_name='weekly_report.html', theme=theme)
    context = TemplateContext(data=report_data)
    batch_requests.append((config, context))

# 병렬 실행
results = engine.batch_render(batch_requests)
```

### 비동기 렌더링

```python
import asyncio

async def render_async():
    result = await engine.render_async(config, context)
    return result

# 비동기 실행
result = asyncio.run(render_async())
```

## 🔧 개발 도구

### 템플릿 검증

```python
# 템플릿 구문 검사
is_valid, errors = engine.validate_template('weekly_report.html')
if not is_valid:
    for error in errors:
        print(f"오류: {error}")
```

### 디버그 모드

```python
config = RenderConfig(
    template_name='weekly_report.html',
    preview_mode=True,  # 디버그 정보 포함
    cache_enabled=False  # 캐시 비활성화
)
```

### 개발 서버

```bash
# 개발용 미리보기 서버 (자동 새로고침)
python templates/demo/template_demo.py

# 선택사항:
# 1. Full System Demo - 모든 기능 테스트
# 2. Interactive Preview Dashboard - Streamlit 대시보드
# 3. Quick Template Test - 빠른 템플릿 테스트
```

## 📚 API 레퍼런스

### KPOPTemplateEngine

주요 메서드:
- `render_template(config, context)` - 템플릿 렌더링
- `render_async(config, context)` - 비동기 렌더링
- `batch_render(requests)` - 배치 렌더링
- `validate_template(name)` - 템플릿 검증
- `get_available_templates()` - 사용 가능한 템플릿 목록
- `clear_cache()` - 캐시 초기화

### RenderConfig

설정 옵션:
- `template_name` - 템플릿 파일명
- `output_format` - 출력 형식 (html/pdf/json)
- `theme` - 테마명
- `include_charts` - 차트 포함 여부
- `cache_enabled` - 캐시 사용 여부
- `preview_mode` - 미리보기 모드
- `responsive` - 반응형 디자인 여부

### TemplateContext

데이터 구조:
- `data` - 템플릿 데이터
- `metadata` - 메타데이터 (report_type 등)
- `user_preferences` - 사용자 설정
- `timestamp` - 생성 시간

## 🤝 기여하기

### 개발 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd kpop-dashboard

# 의존성 설치
pip install -r requirements.txt

# 개발용 패키지 설치
pip install streamlit plotly reportlab jinja2

# 템플릿 시스템 테스트
python templates/demo/template_demo.py
```

### 새 템플릿 추가

1. `templates/reports/html/` 에 새 템플릿 파일 생성
2. `base.html` 을 상속하여 구조 유지
3. 필요한 블록 정의 (`title`, `content`, `extra_css`, `extra_js`)
4. 템플릿 검증 및 테스트

### 새 테마 추가

1. `templates/branding/brand_system.py` 에서 색상 스키마 정의
2. `style_generator.py` 에서 스타일 생성 로직 구현
3. 테마 호환성 테스트

## 📄 라이선스

이 프로젝트는 K-POP Analytics Dashboard의 일부로, 해당 라이선스를 따릅니다.

## 🎵 K-POP Analytics Template System v1.0

**Built with ❤️ for K-POP Industry Analytics**

---

더 자세한 정보나 지원이 필요하시면 [프로젝트 이슈 트래커](../../issues)를 확인해주세요.