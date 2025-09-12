"""
성장률 분석 페이지 (기본 구현)
==========================

K-Pop 아티스트 및 그룹의 성장률을 분석하고 시각화하는 페이지입니다.
- 실제 데이터베이스 기반 성장률 계산
- 플랫폼별 성장률 비교
- 아티스트별 성장률 순위
- 기본적인 트렌드 분석

Author: Backend Development Team  
Date: 2025-09-11
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Import internal modules
try:
    from database_postgresql import (
        get_group_growth_analysis, get_platform_growth_comparison_groups, 
        format_number, get_all_artists_with_details
    )
except ImportError as e:
    st.error(f"모듈 import 오류: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="성장률 분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .growth-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .growth-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    
    .growth-positive { 
        color: #28a745; 
        font-weight: bold;
        font-size: 1.1rem;
    }
    .growth-negative { 
        color: #dc3545; 
        font-weight: bold;
        font-size: 1.1rem;
    }
    .growth-neutral { 
        color: #ffc107; 
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .artist-growth-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .platform-comparison {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="growth-header">
    <h1>📈 성장률 분석</h1>
    <p>K-Pop 아티스트의 플랫폼별 성장률을 종합적으로 분석합니다</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(ttl=300)  # 5분 캐싱
def load_growth_data():
    """그룹 기준 성장률 분석 데이터 로드"""
    group_growth = get_group_growth_analysis(days_back=30)
    platform_comparison = get_platform_growth_comparison_groups()
    return group_growth, platform_comparison

try:
    with st.spinner("📊 성장률 데이터를 분석하고 있습니다..."):
        growth_data, platform_data = load_growth_data()
    
    if not growth_data:
        st.warning("📊 분석할 성장률 데이터가 충분하지 않습니다. 데이터 수집 기간이 더 필요할 수 있습니다.")
        st.info("💡 며칠 후에 다시 확인해보시거나, 데이터 수집 스케줄러가 정상 작동하는지 확인해주세요.")
        st.stop()
    
    # 분석 설정 섹션
    st.markdown("### ⚙️ 분석 설정")
    col_settings1, col_settings2, col_settings3 = st.columns(3)
    
    with col_settings1:
        analysis_period = st.selectbox(
            "분석 기간",
            ["최근 30일", "최근 7일", "최근 60일"],
            index=0
        )
    
    with col_settings2:
        min_growth_filter = st.slider(
            "최소 성장률 필터 (%)",
            min_value=-50.0,
            max_value=50.0,
            value=-10.0,
            step=1.0
        )
        
    with col_settings3:
        show_platform = st.multiselect(
            "표시할 플랫폼",
            ["youtube", "spotify", "twitter"],
            default=["youtube", "spotify", "twitter"]
        )
    
    st.markdown("---")
    
    # 데이터 필터링
    df_growth = pd.DataFrame(growth_data)
    if not df_growth.empty:
        # Ensure avg_growth_rate is numeric before filtering
        df_growth['avg_growth_rate'] = pd.to_numeric(df_growth['avg_growth_rate'], errors='coerce')
        df_growth.dropna(subset=['avg_growth_rate'], inplace=True)

        df_growth = df_growth[
            (df_growth['avg_growth_rate'] >= min_growth_filter) &
            (df_growth['platform'].isin(show_platform))
        ]
    
    # 상위 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    
    if not df_growth.empty:
        avg_growth = df_growth['avg_growth_rate'].mean()
        top_growth = df_growth['avg_growth_rate'].max()
        total_groups = df_growth['group_name'].nunique()
        positive_growth_count = len(df_growth[df_growth['avg_growth_rate'] > 0])
        
        with col1:
            st.metric(
                label="평균 성장률",
                value=f"{avg_growth:.1f}%",
                delta="지난 30일 기준"
            )
        
        with col2:
            st.metric(
                label="최고 성장률",
                value=f"{top_growth:.1f}%",
                delta="개별 플랫폼 기준"
            )
            
        with col3:
            st.metric(
                label="분석 대상",
                value=f"{total_groups}개",
                delta="그룹/솔로"
            )
            
        with col4:
            positive_rate = (positive_growth_count / len(df_growth)) * 100
            st.metric(
                label="양의 성장률",
                value=f"{positive_growth_count}개",
                delta=f"{positive_rate:.1f}% 비율"
            )
    
    st.markdown("---")
    
    # 차트 섹션
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📊 플랫폼별 성장률 비교")
        
        if platform_data:
            df_platform = pd.DataFrame(platform_data)
            
            # 플랫폼별 성장률 막대 차트
            fig_platform = px.bar(
                df_platform,
                x='platform',
                y='growth_rate',
                title="플랫폼별 평균 성장률",
                color='growth_rate',
                color_continuous_scale='RdYlGn',
                text='growth_rate'
            )
            
            fig_platform.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='auto'
            )
            
            fig_platform.update_layout(
                height=400,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_platform, use_container_width=True)
        else:
            st.info("📊 플랫폼 비교 데이터를 준비 중입니다...")
            
        # 그룹별 성장률 분포
        if not df_growth.empty:
            st.markdown("### 📈 그룹별 성장률 분포")
            
            fig_dist = px.histogram(
                df_growth,
                x='avg_growth_rate',
                nbins=20,
                title="그룹 성장률 분포",
                color_discrete_sequence=['#667eea']
            )
            
            fig_dist.update_layout(
                height=350,
                template='plotly_white',
                xaxis_title="성장률 (%)",
                yaxis_title="그룹 수"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with col_right:
        st.markdown("### 🏆 Top 10 성장 그룹")
        
        if not df_growth.empty:
            top_groups = df_growth.nlargest(10, 'avg_growth_rate')
            
            for i, (_, group) in enumerate(top_groups.iterrows(), 1):
                growth_rate = group['avg_growth_rate']
                growth_class = "growth-positive" if growth_rate > 10 else "growth-warning" if growth_rate > 0 else "growth-negative"
                
                st.markdown(f"""
                <div class="artist-growth-item">
                    <div>
                        <strong>#{i} {group['group_name']}</strong><br>
                        <small>{group['company_name']}</small><br>
                        <small>{group['platform'].title()} · {format_number(group['total_followers'])} 팔로워</small>
                    </div>
                    <div class="{growth_class}">
                        {growth_rate:+.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🔍 성장률 데이터를 분석 중입니다...")
    
    st.markdown("---")
    
    # 상세 데이터 테이블
    if not df_growth.empty:
        st.markdown("### 📋 상세 그룹 성장률 데이터")
        
        # 테이블 표시용 데이터 준비
        display_df = df_growth.copy()
        display_df['total_followers'] = display_df['total_followers'].apply(format_number)
        display_df['avg_growth_rate'] = display_df['avg_growth_rate'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(
            display_df[[
                'group_name', 'company_name', 
                'platform', 'total_followers', 'avg_growth_rate'
            ]],
            column_config={
                "group_name": "그룹/솔로", 
                "company_name": "소속사",
                "platform": "플랫폼",
                "total_followers": "팔로워 수",
                "avg_growth_rate": "성장률"
            },
            hide_index=True,
            use_container_width=True
        )
    
    # 분석 정보
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
        <h4>📊 분석 정보 (그룹 기준)</h4>
        <ul>
            <li><strong>분석 기준:</strong> 그룹 단위 통합 분석 (멤버들의 지표 합산)</li>
            <li><strong>분석 기간:</strong> 최근 30일간 수집된 데이터 기준</li>
            <li><strong>성장률 계산:</strong> 그룹 전체 팔로워 수 변화율 기준</li>
            <li><strong>데이터 갱신:</strong> 5분마다 자동 업데이트</li>
            <li><strong>플랫폼:</strong> YouTube 구독자, Spotify 팔로워, Twitter 팔로워 기준</li>
            <li><strong>필터 조건:</strong> 최소 2개 이상의 데이터 포인트가 있는 그룹만 포함</li>
            <li><strong>솔로 아티스트:</strong> 개별 아티스트는 단일 그룹으로 취급</li>
        </ul>
        <p style="margin-bottom: 0; font-size: 0.9rem; color: #6c757d;">
            💡 그룹 단위 분석으로 전체적인 트렌드를 더 명확히 파악할 수 있습니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ 성장률 분석 중 오류가 발생했습니다: {e}")
    st.info("🔧 시스템 관리자에게 문의하거나 잠시 후 다시 시도해주세요.")
    
    # 디버그 정보 (개발 중에만)
    if st.checkbox("🔧 디버그 정보 표시"):
        st.exception(e)
