import streamlit as st
import pandas as pd
from database_postgresql import get_company_by_name, get_groups_and_artists_in_hierarchy # Updated imports
from utils.navigation import navigate_to # New import
import plotly.express as px

st.set_page_config(page_title="기업별 분석", page_icon="🏢")

st.title("🏢 기업별 분석")
st.write("메인 페이지 사이드바에서 선택한 회사의 소속 그룹 및 아티스트 데이터를 분석합니다.")

# session_state에서 선택된 회사 가져오기 (app.py와 공유)
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None

selected_company_name = st.session_state.get('selected_company', None)

if not selected_company_name:
    st.warning("메인 페이지의 사이드바에서 분석할 회사를 먼저 선택해주세요.")
else:
    st.header(f"📈 {selected_company_name} 분석 결과")
    
    # 회사 이름으로 ID 찾기
    company_info = get_company_by_name(selected_company_name)
    
    if not company_info:
        st.error("선택한 회사를 찾을 수 없습니다.")
    else:
        company_id = company_info['company_id']
        
        st.subheader("소속 그룹 및 아티스트")
        
        hierarchy_data = get_groups_and_artists_in_hierarchy(company_id)
        
        if not hierarchy_data:
            st.info(f"{selected_company_name} 및 그 자회사에 소속된 그룹/아티스트 정보가 없습니다. '데이터 관리' 페이지에서 추가해주세요.")
        else:
            for group_info in hierarchy_data:
                st.markdown(f"#### 👥 {group_info['group_name']}")
                
                if group_info['artists']:
                    cols = st.columns(5) # 한 줄에 5명씩 표시
                    for i, artist in enumerate(group_info['artists']):
                        with cols[i % 5]:
                            if st.button(artist['name'], key=f"artist_{artist['artist_id']}"):
                                navigate_to("아티스트별_분석", artist_id=artist['artist_id'])
                else:
                    st.markdown("  - 소속 아티스트 없음")
            
            st.markdown("--- ")
            st.subheader("📊 소속 그룹별 플랫폼 지표 비교")
            
            # 해당 회사의 모든 그룹 데이터 수집
            company_groups_data = []
            for group_info in hierarchy_data:
                group_id = group_info['group_id']
                group_name = group_info['group_name']
                
                # 그룹 메트릭 데이터 가져오기
                from database_postgresql import get_latest_metric_for_group
                yt_metric = get_latest_metric_for_group(group_id, 'youtube')
                sp_metric = get_latest_metric_for_group(group_id, 'spotify')
                
                if yt_metric or sp_metric:  # 데이터가 있는 그룹만 포함
                    company_groups_data.append({
                        '그룹': group_name,
                        'YouTube 구독자': yt_metric.get('subscribers', 0),
                        'YouTube 총 조회수': yt_metric.get('total_views', 0),
                        'Spotify 팔로워': sp_metric.get('followers', 0),
                        'Spotify 월간 청취자': sp_metric.get('monthly_listeners', 0),
                    })
            
            if company_groups_data:
                import pandas as pd
                import plotly.express as px
                
                df = pd.DataFrame(company_groups_data)
                
                # 수치 포맷팅 함수
                def format_number(num):
                    if num >= 1_000_000_000:
                        return f"{num/1_000_000_000:.1f}B"
                    elif num >= 1_000_000:
                        return f"{num/1_000_000:.1f}M" 
                    elif num >= 1_000:
                        return f"{num/1_000:.1f}K"
                    else:
                        return str(num)
                
                # 데이터 테이블
                display_df = df.copy()
                numeric_columns = ['YouTube 구독자', 'YouTube 총 조회수', 'Spotify 팔로워', 'Spotify 월간 청취자']
                for col in numeric_columns:
                    display_df[col] = display_df[col].apply(format_number)
                
                st.dataframe(display_df, width='stretch', hide_index=True)
                
                # 차트
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### YouTube 구독자 비교")
                    if df['YouTube 구독자'].sum() > 0:
                        fig_yt = px.bar(
                            df,
                            x='그룹',
                            y='YouTube 구독자',
                            title=f"{selected_company_name} 소속 그룹 YouTube 구독자 비교",
                            labels={'YouTube 구독자': '구독자 수 (명)'}
                        )
                        fig_yt.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_yt, width='stretch')
                    else:
                        st.info("YouTube 구독자 데이터가 없습니다.")
                
                with col2:
                    st.markdown("#### Spotify 팔로워 비교")
                    if df['Spotify 팔로워'].sum() > 0:
                        fig_sp = px.bar(
                            df,
                            x='그룹',
                            y='Spotify 팔로워',
                            title=f"{selected_company_name} 소속 그룹 Spotify 팔로워 비교",
                            labels={'Spotify 팔로워': '팔로워 수 (명)'}
                        )
                        fig_sp.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_sp, width='stretch')
                    else:
                        st.info("Spotify 팔로워 데이터가 없습니다.")
                        
                # 종합 비교 차트
                if len(df) > 1:  # 2개 이상 그룹이 있을 때만
                    st.markdown("#### 종합 플랫폼 지표 비교")
                    main_metrics = ['YouTube 구독자', 'Spotify 팔로워']
                    df_main = df[['그룹'] + main_metrics]
                    df_melted = df_main.melt(id_vars=['그룹'], var_name='플랫폼', value_name='수치')
                    
                    fig_combined = px.bar(
                        df_melted,
                        x='그룹',
                        y='수치',
                        color='플랫폼',
                        barmode='group',
                        title=f"{selected_company_name} 소속 그룹 플랫폼 지표 종합 비교",
                        labels={'수치': '팔로워/구독자 수 (명)'}
                    )
                    fig_combined.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_combined, width='stretch')
            else:
                st.info(f"{selected_company_name} 소속 그룹들의 플랫폼 데이터가 아직 수집되지 않았습니다.")