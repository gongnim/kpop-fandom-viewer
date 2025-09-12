"""
반응형 디자인 스타일 모듈
======================

K-Pop Dashboard의 모든 페이지에서 사용할 수 있는 반응형 CSS 스타일을 제공합니다.
- 모바일, 태블릿, 데스크톱 대응
- 일관된 디자인 시스템
- 접근성 고려
- 성능 최적화

Author: Frontend Development Team  
Date: 2025-09-09
"""

def get_responsive_css():
    """반응형 CSS 스타일 반환"""
    return """
    <style>
        /* ==========================================================================
           글로벌 스타일 및 CSS 변수
           ========================================================================== */
        
        :root {
            /* 컬러 시스템 */
            --primary-color: #1e3c72;
            --primary-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            --secondary-color: #667eea;
            --accent-color: #764ba2;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            
            /* 간격 시스템 */
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            --spacing-xxl: 3rem;
            
            /* 폰트 시스템 */
            --font-size-xs: 0.75rem;
            --font-size-sm: 0.875rem;
            --font-size-base: 1rem;
            --font-size-lg: 1.125rem;
            --font-size-xl: 1.25rem;
            --font-size-2xl: 1.5rem;
            --font-size-3xl: 2rem;
            --font-size-4xl: 2.5rem;
            
            /* 그림자 시스템 */
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --shadow-base: 0 2px 4px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 8px 15px rgba(0, 0, 0, 0.15);
            --shadow-xl: 0 16px 30px rgba(0, 0, 0, 0.2);
            
            /* 둥근 모서리 */
            --radius-sm: 4px;
            --radius-base: 8px;
            --radius-md: 10px;
            --radius-lg: 12px;
            --radius-xl: 16px;
            --radius-full: 50%;
        }
        
        /* ==========================================================================
           기본 레이아웃 및 컨테이너
           ========================================================================== */
        
        .responsive-container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 var(--spacing-md);
        }
        
        .section {
            margin-bottom: var(--spacing-xl);
        }
        
        .section-header {
            margin-bottom: var(--spacing-lg);
            padding-bottom: var(--spacing-sm);
            border-bottom: 2px solid var(--light-color);
        }
        
        .section-title {
            font-size: var(--font-size-2xl);
            font-weight: 700;
            color: var(--primary-color);
            margin: 0;
        }
        
        .section-subtitle {
            font-size: var(--font-size-base);
            color: #666;
            margin: var(--spacing-xs) 0 0 0;
        }
        
        /* ==========================================================================
           카드 시스템
           ========================================================================== */
        
        .card {
            background: white;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-base);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-md);
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }
        
        .card-header {
            display: flex;
            justify-content: between;
            align-items: center;
            padding-bottom: var(--spacing-md);
            margin-bottom: var(--spacing-md);
            border-bottom: 1px solid #e9ecef;
        }
        
        .card-title {
            font-size: var(--font-size-lg);
            font-weight: 600;
            color: var(--primary-color);
            margin: 0;
        }
        
        .card-body {
            padding: 0;
        }
        
        .card-footer {
            margin-top: var(--spacing-md);
            padding-top: var(--spacing-md);
            border-top: 1px solid #e9ecef;
            font-size: var(--font-size-sm);
            color: #666;
        }
        
        /* 메트릭 카드 */
        .metric-card {
            background: var(--primary-gradient);
            color: white;
            text-align: center;
            border: none;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }
        
        .metric-value {
            font-size: var(--font-size-3xl);
            font-weight: 700;
            margin: var(--spacing-sm) 0;
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: var(--font-size-sm);
            opacity: 0.9;
            margin-bottom: var(--spacing-xs);
        }
        
        .metric-change {
            font-size: var(--font-size-sm);
            opacity: 0.9;
            margin-top: var(--spacing-xs);
        }
        
        /* ==========================================================================
           상태 및 알림 시스템
           ========================================================================== */
        
        .status-positive, .status-success {
            color: var(--success-color) !important;
        }
        
        .status-negative, .status-danger {
            color: var(--danger-color) !important;
        }
        
        .status-warning {
            color: var(--warning-color) !important;
        }
        
        .status-neutral, .status-info {
            color: var(--info-color) !important;
        }
        
        .alert {
            padding: var(--spacing-md);
            border-radius: var(--radius-base);
            margin-bottom: var(--spacing-md);
            border: 1px solid transparent;
        }
        
        .alert-success {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffeaa7;
            color: #856404;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        
        .alert-info {
            background-color: #d1ecf1;
            border-color: #bee5eb;
            color: #0c5460;
        }
        
        /* ==========================================================================
           버튼 시스템
           ========================================================================== */
        
        .btn {
            display: inline-block;
            padding: var(--spacing-sm) var(--spacing-lg);
            border-radius: var(--radius-base);
            text-decoration: none;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: var(--font-size-base);
            font-weight: 500;
            border: none;
            outline: none;
        }
        
        .btn-primary {
            background: var(--primary-gradient);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-outline {
            background: transparent;
            border: 2px solid var(--primary-color);
            color: var(--primary-color);
        }
        
        .btn-outline:hover {
            background: var(--primary-color);
            color: white;
        }
        
        /* ==========================================================================
           차트 및 시각화 컨테이너
           ========================================================================== */
        
        .chart-container {
            background: white;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-base);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-md);
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-md);
        }
        
        .chart-title {
            font-size: var(--font-size-lg);
            font-weight: 600;
            color: var(--primary-color);
            margin: 0;
        }
        
        .chart-controls {
            display: flex;
            gap: var(--spacing-sm);
            align-items: center;
        }
        
        /* ==========================================================================
           테이블 스타일
           ========================================================================== */
        
        .responsive-table {
            width: 100%;
            background: white;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-base);
            overflow: hidden;
        }
        
        .table-header {
            background: var(--light-color);
            font-weight: 600;
            color: var(--primary-color);
        }
        
        .table-row:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .table-row:hover {
            background-color: #e9ecef;
        }
        
        /* ==========================================================================
           반응형 그리드 시스템
           ========================================================================== */
        
        .grid {
            display: grid;
            gap: var(--spacing-md);
        }
        
        .grid-1 { grid-template-columns: 1fr; }
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .grid-4 { grid-template-columns: repeat(4, 1fr); }
        .grid-6 { grid-template-columns: repeat(6, 1fr); }
        
        /* ==========================================================================
           모바일 반응형 (768px 이하)
           ========================================================================== */
        
        @media (max-width: 768px) {
            :root {
                --spacing-md: 0.75rem;
                --spacing-lg: 1rem;
                --spacing-xl: 1.5rem;
                --font-size-2xl: 1.25rem;
                --font-size-3xl: 1.75rem;
                --font-size-4xl: 2rem;
            }
            
            .responsive-container {
                padding: 0 var(--spacing-sm);
            }
            
            .card {
                padding: var(--spacing-md);
                border-radius: var(--radius-base);
            }
            
            .metric-value {
                font-size: var(--font-size-2xl);
            }
            
            .section-title {
                font-size: var(--font-size-xl);
            }
            
            .chart-container {
                padding: var(--spacing-md);
            }
            
            .chart-header {
                flex-direction: column;
                align-items: flex-start;
                gap: var(--spacing-sm);
            }
            
            .chart-controls {
                width: 100%;
                justify-content: space-between;
            }
            
            /* 그리드를 모바일에서 단일 컬럼으로 */
            .grid-2, .grid-3, .grid-4, .grid-6 {
                grid-template-columns: 1fr;
            }
            
            /* 버튼을 전체 너비로 */
            .btn {
                width: 100%;
                margin-bottom: var(--spacing-sm);
            }
            
            /* 테이블 가로 스크롤 */
            .responsive-table {
                overflow-x: auto;
                font-size: var(--font-size-sm);
            }
            
            /* 메트릭 카드 간격 조정 */
            .metric-card {
                margin-bottom: var(--spacing-sm);
            }
            
            /* 알림 패널 간격 조정 */
            .alert {
                padding: var(--spacing-sm);
                font-size: var(--font-size-sm);
            }
        }
        
        /* ==========================================================================
           태블릿 반응형 (769px - 1024px)
           ========================================================================== */
        
        @media (min-width: 769px) and (max-width: 1024px) {
            .grid-4, .grid-6 {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .responsive-container {
                padding: 0 var(--spacing-lg);
            }
            
            .chart-container {
                padding: var(--spacing-lg);
            }
        }
        
        /* ==========================================================================
           데스크톱 대형 화면 (1200px 이상)
           ========================================================================== */
        
        @media (min-width: 1200px) {
            .responsive-container {
                max-width: 1400px;
                padding: 0 var(--spacing-xl);
            }
            
            .section {
                margin-bottom: var(--spacing-xxl);
            }
        }
        
        /* ==========================================================================
           접근성 개선
           ========================================================================== */
        
        /* 포커스 표시 개선 */
        .btn:focus,
        input:focus,
        select:focus,
        textarea:focus {
            outline: 2px solid var(--primary-color);
            outline-offset: 2px;
        }
        
        /* 고대비 모드 지원 */
        @media (prefers-contrast: high) {
            :root {
                --primary-color: #000080;
                --success-color: #008000;
                --danger-color: #800000;
                --warning-color: #808000;
            }
            
            .card {
                border: 2px solid #333;
            }
        }
        
        /* 모션 감소 선호 사용자를 위한 설정 */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* ==========================================================================
           인쇄 스타일
           ========================================================================== */
        
        @media print {
            .card {
                box-shadow: none;
                border: 1px solid #ccc;
            }
            
            .btn {
                display: none;
            }
            
            .chart-container {
                break-inside: avoid;
            }
            
            .metric-card {
                background: white !important;
                color: black !important;
                border: 1px solid #ccc;
            }
        }
        
        /* ==========================================================================
           다크 모드 지원 (선택사항)
           ========================================================================== */
        
        @media (prefers-color-scheme: dark) {
            :root {
                --light-color: #343a40;
                --dark-color: #f8f9fa;
            }
            
            .card {
                background: #2d3748;
                color: white;
                border-color: #4a5568;
            }
            
            .section-title {
                color: var(--secondary-color);
            }
            
            .chart-container {
                background: #2d3748;
                color: white;
            }
        }
        
        /* ==========================================================================
           유틸리티 클래스
           ========================================================================== */
        
        /* 간격 */
        .m-0 { margin: 0 !important; }
        .m-1 { margin: var(--spacing-xs) !important; }
        .m-2 { margin: var(--spacing-sm) !important; }
        .m-3 { margin: var(--spacing-md) !important; }
        .m-4 { margin: var(--spacing-lg) !important; }
        .m-5 { margin: var(--spacing-xl) !important; }
        
        .p-0 { padding: 0 !important; }
        .p-1 { padding: var(--spacing-xs) !important; }
        .p-2 { padding: var(--spacing-sm) !important; }
        .p-3 { padding: var(--spacing-md) !important; }
        .p-4 { padding: var(--spacing-lg) !important; }
        .p-5 { padding: var(--spacing-xl) !important; }
        
        /* 텍스트 정렬 */
        .text-left { text-align: left !important; }
        .text-center { text-align: center !important; }
        .text-right { text-align: right !important; }
        
        /* 디스플레이 */
        .d-none { display: none !important; }
        .d-block { display: block !important; }
        .d-flex { display: flex !important; }
        .d-grid { display: grid !important; }
        
        /* 플렉스박스 */
        .justify-content-start { justify-content: flex-start !important; }
        .justify-content-center { justify-content: center !important; }
        .justify-content-end { justify-content: flex-end !important; }
        .justify-content-between { justify-content: space-between !important; }
        
        .align-items-start { align-items: flex-start !important; }
        .align-items-center { align-items: center !important; }
        .align-items-end { align-items: flex-end !important; }
        
        /* 너비 */
        .w-25 { width: 25% !important; }
        .w-50 { width: 50% !important; }
        .w-75 { width: 75% !important; }
        .w-100 { width: 100% !important; }
        
        /* 높이 */
        .h-25 { height: 25% !important; }
        .h-50 { height: 50% !important; }
        .h-75 { height: 75% !important; }
        .h-100 { height: 100% !important; }
        
        /* 색상 */
        .text-primary { color: var(--primary-color) !important; }
        .text-success { color: var(--success-color) !important; }
        .text-warning { color: var(--warning-color) !important; }
        .text-danger { color: var(--danger-color) !important; }
        .text-info { color: var(--info-color) !important; }
        .text-muted { color: #6c757d !important; }
        
        /* 배경 색상 */
        .bg-primary { background-color: var(--primary-color) !important; }
        .bg-light { background-color: var(--light-color) !important; }
        .bg-white { background-color: white !important; }
        
        /* 그림자 */
        .shadow-none { box-shadow: none !important; }
        .shadow-sm { box-shadow: var(--shadow-sm) !important; }
        .shadow { box-shadow: var(--shadow-base) !important; }
        .shadow-lg { box-shadow: var(--shadow-lg) !important; }
        
        /* 둥근 모서리 */
        .rounded-none { border-radius: 0 !important; }
        .rounded-sm { border-radius: var(--radius-sm) !important; }
        .rounded { border-radius: var(--radius-base) !important; }
        .rounded-lg { border-radius: var(--radius-lg) !important; }
        .rounded-full { border-radius: var(--radius-full) !important; }
        
    </style>
    """

def get_streamlit_theme_config():
    """Streamlit 테마 설정 반환"""
    return {
        "primaryColor": "#1e3c72",
        "backgroundColor": "#ffffff", 
        "secondaryBackgroundColor": "#f8f9fa",
        "textColor": "#262730",
        "font": "sans serif"
    }

def get_plotly_theme():
    """Plotly 차트 테마 설정 반환"""
    return {
        "layout": {
            "colorway": [
                "#1e3c72", "#667eea", "#764ba2", "#2a5298",
                "#28a745", "#ffc107", "#dc3545", "#17a2b8"
            ],
            "template": "plotly_white",
            "font": {
                "family": "Arial, sans-serif",
                "size": 12,
                "color": "#262730"
            },
            "title": {
                "font": {
                    "size": 16,
                    "color": "#1e3c72"
                }
            },
            "xaxis": {
                "gridcolor": "#e9ecef",
                "linecolor": "#dee2e6"
            },
            "yaxis": {
                "gridcolor": "#e9ecef", 
                "linecolor": "#dee2e6"
            }
        }
    }

# 페이지별 스타일 클래스
class PageStyles:
    """페이지별 특화 스타일"""
    
    @staticmethod
    def growth_analysis():
        """성장률 분석 페이지 전용 스타일"""
        return """
        .growth-trend-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .growth-positive .metric-value {
            color: #4CAF50 !important;
        }
        
        .growth-negative .metric-value {
            color: #f44336 !important;
        }
        
        .growth-chart-container {
            border: 2px solid #667eea;
            border-radius: 12px;
        }
        """
    
    @staticmethod
    def executive_kpi():
        """경영진 KPI 페이지 전용 스타일"""
        return """
        .executive-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .kpi-executive-card {
            border-left: 4px solid #2a5298;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .strategic-insight-panel {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
        }
        """
    
    @staticmethod
    def artist_analysis():
        """아티스트 분석 페이지 전용 스타일"""
        return """
        .artist-profile-card {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #333;
        }
        
        .platform-metric-card {
            border-left: 4px solid;
            border-left-color: var(--platform-color, #667eea);
        }
        """