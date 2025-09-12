"""
K-Pop 이벤트 API 테스트 페이지
백엔드 API와 Streamlit 프론트엔드 연동 테스트
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import json

# 상위 디렉토리에서 모듈 import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_client import get_api_client, safe_api_call, check_api_connection, display_api_error

# 페이지 설정
st.set_page_config(
    page_title="API 테스트",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 K-Pop 이벤트 API 테스트")
st.markdown("---")

# API 클라이언트 초기화
api_client = get_api_client()

# API 연결 상태 확인
with st.sidebar:
    st.header("🌐 API 서버 상태")
    
    if st.button("연결 상태 확인", type="primary"):
        if check_api_connection(api_client):
            st.success("✅ API 서버 연결됨")
            
            # API 정보 표시
            api_info = safe_api_call(api_client.api_info)
            if api_info:
                st.json(api_info)
        else:
            st.error("❌ API 서버 연결 실패")
            st.info("💡 API 서버를 먼저 실행하세요:")
            st.code("python -m kpop_dashboard.api_server")

# 메인 콘텐츠
tab1, tab2, tab3, tab4 = st.tabs(["📅 이벤트 조회", "➕ 이벤트 생성", "📊 영향도 분석", "📈 통계 정보"])

# 탭 1: 이벤트 조회
with tab1:
    st.header("📅 이벤트 조회 테스트")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 이벤트 검색")
        
        # 검색 필터
        start_date = st.date_input("시작 날짜", value=date.today() - timedelta(days=30))
        end_date = st.date_input("종료 날짜", value=date.today() + timedelta(days=30))
        event_type = st.selectbox("이벤트 타입", ["", "award_ceremony", "comeback", "concert"])
        category = st.selectbox("카테고리", ["", "AWARD_CEREMONY", "COMEBACK", "CONCERT", "COLLABORATION"])
        
        if st.button("이벤트 조회", type="primary"):
            with st.spinner("이벤트를 조회하는 중..."):
                # API 호출
                events_data = safe_api_call(
                    api_client.get_events,
                    start_date=start_date,
                    end_date=end_date,
                    event_type=event_type if event_type else None,
                    category=category if category else None
                )
                
                if events_data and events_data.get("success"):
                    events = events_data.get("data", {}).get("events", [])
                    
                    if events:
                        st.success(f"✅ {len(events)}개의 이벤트를 찾았습니다!")
                        
                        # 데이터프레임으로 표시
                        df = pd.DataFrame(events)
                        st.dataframe(df, use_container_width=True)
                        
                        # 다운로드 버튼
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 CSV 다운로드",
                            data=csv,
                            file_name=f"events_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("검색 결과가 없습니다.")
                else:
                    display_api_error("이벤트 조회에 실패했습니다.")
    
    with col2:
        st.subheader("📈 다가오는 이벤트")
        
        days_ahead = st.slider("조회 기간 (일)", 1, 90, 30)
        
        if st.button("다가오는 이벤트 조회"):
            with st.spinner("다가오는 이벤트를 조회하는 중..."):
                upcoming_data = safe_api_call(api_client.get_upcoming_events, days_ahead=days_ahead)
                
                if upcoming_data and upcoming_data.get("success"):
                    events = upcoming_data.get("data", {}).get("events", [])
                    
                    if events:
                        st.success(f"✅ {len(events)}개의 다가오는 이벤트")
                        
                        for event in events[:5]:  # 최대 5개만 표시
                            with st.container():
                                st.write(f"**{event.get('name', 'N/A')}**")
                                st.write(f"📅 {event.get('date', 'N/A')}")
                                st.write(f"🏷️ {event.get('category_name', 'N/A')}")
                                st.markdown("---")
                    else:
                        st.info("다가오는 이벤트가 없습니다.")
                else:
                    display_api_error("다가오는 이벤트 조회에 실패했습니다.")

# 탭 2: 이벤트 생성
with tab2:
    st.header("➕ 새 이벤트 생성 테스트")
    
    with st.form("create_event_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            event_name = st.text_input("이벤트 이름*", placeholder="예: 2024 MAMA Awards")
            event_type = st.selectbox("이벤트 타입*", ["award_ceremony", "comeback", "concert", "collaboration"])
            event_category = st.selectbox("카테고리*", ["AWARD_CEREMONY", "COMEBACK", "CONCERT", "COLLABORATION"])
            event_date = st.date_input("이벤트 날짜*")
            venue = st.text_input("개최 장소", placeholder="예: Seoul Olympic Stadium")
        
        with col2:
            end_date = st.date_input("종료 날짜 (선택사항)")
            importance_level = st.slider("중요도", 1, 5, 3)
            global_impact_score = st.number_input("글로벌 영향도 점수", 0.0, 100.0, 0.0)
            is_annual = st.checkbox("연례 이벤트")
            description = st.text_area("이벤트 설명", placeholder="이벤트에 대한 상세 설명...")
        
        submitted = st.form_submit_button("이벤트 생성", type="primary")
        
        if submitted:
            if event_name and event_type and event_category and event_date:
                # API 호출용 데이터 준비
                event_data = {
                    "name": event_name,
                    "event_type": event_type,
                    "category": event_category,
                    "date": event_date.isoformat(),
                    "end_date": end_date.isoformat() if end_date else None,
                    "venue": venue,
                    "description": description,
                    "importance_level": importance_level,
                    "global_impact_score": global_impact_score,
                    "is_annual": is_annual,
                    "metadata": {}
                }
                
                with st.spinner("이벤트를 생성하는 중..."):
                    result = safe_api_call(api_client.create_event, event_data)
                    
                    if result and result.get("success"):
                        st.success("✅ 이벤트가 성공적으로 생성되었습니다!")
                        st.json(result)
                    else:
                        display_api_error("이벤트 생성에 실패했습니다.")
            else:
                st.error("❌ 필수 항목을 모두 입력해주세요.")

# 탭 3: 영향도 분석
with tab3:
    st.header("📊 이벤트 영향도 분석 테스트")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 영향도 기록")
        
        with st.form("impact_form"):
            event_id = st.number_input("이벤트 ID", min_value=1, value=1)
            artist_id = st.number_input("아티스트 ID", min_value=1, value=1)
            metric_type = st.selectbox("지표 타입", ["subscribers", "followers", "views", "plays", "likes"])
            platform = st.selectbox("플랫폼", ["youtube", "spotify", "instagram", "twitter", "tiktok"])
            
            before_value = st.number_input("이벤트 전 값", min_value=0, value=1000000)
            after_value = st.number_input("이벤트 후 값", min_value=0, value=1200000)
            measurement_period = st.number_input("측정 기간 (일)", min_value=1, max_value=30, value=7)
            
            impact_submitted = st.form_submit_button("영향도 기록", type="primary")
            
            if impact_submitted:
                impact_data = {
                    "event_id": event_id,
                    "artist_id": artist_id,
                    "metric_type": metric_type,
                    "platform": platform,
                    "before_value": before_value,
                    "after_value": after_value,
                    "measurement_period": measurement_period,
                    "statistical_significance": 0.05,
                    "confidence_level": 0.95
                }
                
                with st.spinner("영향도를 기록하는 중..."):
                    result = safe_api_call(api_client.record_event_impact, impact_data)
                    
                    if result and result.get("success"):
                        st.success("✅ 영향도가 성공적으로 기록되었습니다!")
                        
                        # 계산된 영향도 표시
                        impact_percentage = ((after_value - before_value) / before_value) * 100
                        st.metric("계산된 영향도", f"{impact_percentage:.2f}%")
                    else:
                        display_api_error("영향도 기록에 실패했습니다.")
    
    with col2:
        st.subheader("🔍 영향도 조회")
        
        query_event_id = st.number_input("조회할 이벤트 ID", min_value=1, value=1, key="query_event")
        
        if st.button("영향도 분석 조회"):
            with st.spinner("영향도 분석을 조회하는 중..."):
                impact_analysis = safe_api_call(api_client.get_event_impact_analysis, query_event_id)
                
                if impact_analysis and impact_analysis.get("success"):
                    impact_data = impact_analysis.get("data", {}).get("impact_measurements", [])
                    
                    if impact_data:
                        st.success(f"✅ {len(impact_data)}개의 영향도 측정 기록")
                        
                        # 데이터프레임으로 표시
                        df = pd.DataFrame(impact_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # 평균 영향도 계산
                        if 'impact_percentage' in df.columns:
                            avg_impact = df['impact_percentage'].mean()
                            st.metric("평균 영향도", f"{avg_impact:.2f}%")
                    else:
                        st.info("해당 이벤트의 영향도 기록이 없습니다.")
                else:
                    display_api_error("영향도 분석 조회에 실패했습니다.")

# 탭 4: 통계 정보
with tab4:
    st.header("📈 이벤트 통계 정보")
    
    if st.button("통계 정보 조회", type="primary"):
        with st.spinner("통계 정보를 조회하는 중..."):
            stats_data = safe_api_call(api_client.get_event_statistics)
            
            if stats_data and stats_data.get("success"):
                stats = stats_data.get("data", {})
                
                # 주요 통계 메트릭
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총 이벤트 수", stats.get("total_events", 0))
                
                with col2:
                    st.metric("이번 달 이벤트", stats.get("monthly_events", 0))
                
                with col3:
                    st.metric("다가오는 이벤트", stats.get("upcoming_events", 0))
                
                with col4:
                    st.metric("활성 카테고리", len(stats.get("category_stats", [])))
                
                # 카테고리별 분포
                st.subheader("📊 카테고리별 이벤트 분포")
                category_stats = stats.get("category_stats", [])
                
                if category_stats:
                    df_category = pd.DataFrame(category_stats)
                    st.bar_chart(df_category.set_index("name")["event_count"])
                
                # 원시 데이터 표시
                with st.expander("🔍 상세 통계 데이터"):
                    st.json(stats)
            else:
                display_api_error("통계 정보 조회에 실패했습니다.")

# 푸터
st.markdown("---")
st.markdown("**💡 API 사용법:**")
st.code("""
# API 서버 실행
python -m kpop_dashboard.api_server

# API 문서 확인
http://localhost:8000/docs

# 헬스 체크
http://localhost:8000/health
""")

st.info("🔗 이 페이지는 백엔드 API와 프론트엔드 Streamlit의 연동을 테스트합니다.")