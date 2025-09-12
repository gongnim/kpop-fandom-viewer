import streamlit as st
import pandas as pd
import plotly.express as px
from database_postgresql import get_all_groups_with_details, get_latest_metric_for_group

st.set_page_config(page_title="플랫폼 비교 분석", page_icon="🆚")

st.title("🆚 그룹별 플랫폼 비교 분석")
st.write("여러 K-Pop 그룹을 선택하여 주요 플랫폼에서의 팬덤 규모를 비교해보세요.")

# 그룹 선택
groups = get_all_groups_with_details()
if not groups:
    st.warning("먼저 '데이터 관리' 페이지에서 그룹을 추가해주세요.")
else:
    group_options = {f"{group['group_name']} ({group['company_name']})": group['group_id'] for group in groups}
    selected_groups_display = st.multiselect(
        "비교할 그룹을 선택하세요 (최대 6개)",
        options=list(group_options.keys()),
        max_selections=6,
        help="그룹 단위로 비교하면 더 큰 팬덤 규모를 확인할 수 있습니다"
    )
    
    if selected_groups_display:
        st.markdown("---")
        
        # 데이터 수집
        comparison_data = []
        for display_name in selected_groups_display:
            group_id = group_options[display_name]
            group_name = display_name.split(' (')[0]
            company_name = display_name.split(' (')[1].rstrip(')')
            
            # 각 플랫폼별 최신 메트릭 가져오기
            yt_metric = get_latest_metric_for_group(group_id, 'youtube')
            sp_metric = get_latest_metric_for_group(group_id, 'spotify')
            
            comparison_data.append({
                '그룹': group_name,
                '소속사': company_name,
                'YouTube 구독자': yt_metric.get('subscribers', 0),
                'YouTube 총 조회수': yt_metric.get('total_views', 0),
                'YouTube 비디오 수': yt_metric.get('video_count', 0),
                'Spotify 팔로워': sp_metric.get('followers', 0),
                'Spotify 월간 청취자': sp_metric.get('monthly_listeners', 0),
            })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
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
            
            # 데이터 테이블 표시
            st.subheader("📊 그룹별 플랫폼 지표 비교")
            
            # 포맷팅된 데이터프레임 생성
            display_df = df.copy()
            numeric_columns = ['YouTube 구독자', 'YouTube 총 조회수', 'YouTube 비디오 수', 'Spotify 팔로워', 'Spotify 월간 청취자']
            for col in numeric_columns:
                display_df[col] = display_df[col].apply(format_number)
            
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            # 차트 섹션
            st.markdown("---")
            st.subheader("📈 시각화 비교")
            
            # 탭으로 차트 구분
            tab1, tab2, tab3 = st.tabs(["📺 YouTube", "🎵 Spotify", "📊 종합 비교"])
            
            with tab1:
                st.markdown("#### YouTube 플랫폼 지표")
                
                # YouTube 구독자 비교
                fig_yt_subs = px.bar(
                    df, 
                    x='그룹', 
                    y='YouTube 구독자',
                    color='소속사',
                    title="그룹별 YouTube 구독자 수 비교",
                    labels={'YouTube 구독자': '구독자 수 (명)'}
                )
                fig_yt_subs.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_yt_subs, width='stretch')
                
                # YouTube 총 조회수 비교
                fig_yt_views = px.bar(
                    df,
                    x='그룹',
                    y='YouTube 총 조회수', 
                    color='소속사',
                    title="그룹별 YouTube 총 조회수 비교",
                    labels={'YouTube 총 조회수': '총 조회수'}
                )
                fig_yt_views.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_yt_views, width='stretch')
                
            with tab2:
                st.markdown("#### Spotify 플랫폼 지표")
                
                # Spotify 팔로워 비교
                fig_sp_followers = px.bar(
                    df,
                    x='그룹',
                    y='Spotify 팔로워',
                    color='소속사', 
                    title="그룹별 Spotify 팔로워 수 비교",
                    labels={'Spotify 팔로워': '팔로워 수 (명)'}
                )
                fig_sp_followers.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_sp_followers, width='stretch')
                
                # Spotify 월간 청취자 비교
                fig_sp_listeners = px.bar(
                    df,
                    x='그룹',
                    y='Spotify 월간 청취자',
                    color='소속사',
                    title="그룹별 Spotify 월간 청취자 수 비교", 
                    labels={'Spotify 월간 청취자': '월간 청취자 수 (명)'}
                )
                fig_sp_listeners.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_sp_listeners, width='stretch')
                
            with tab3:
                st.markdown("#### 주요 지표 종합 비교")
                
                # 주요 지표만 선택해서 멜트
                main_metrics = ['YouTube 구독자', 'Spotify 팔로워', 'Spotify 월간 청취자']
                df_main = df[['그룹', '소속사'] + main_metrics]
                df_melted = df_main.melt(id_vars=['그룹', '소속사'], var_name='플랫폼 지표', value_name='수치')
                
                fig_combined = px.bar(
                    df_melted,
                    x='그룹',
                    y='수치',
                    color='플랫폼 지표',
                    barmode='group',
                    title="그룹별 주요 플랫폼 지표 종합 비교",
                    labels={'수치': '팔로워/구독자 수 (명)'}
                )
                fig_combined.update_layout(
                    xaxis_tickangle=-45,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right", 
                        x=1
                    )
                )
                st.plotly_chart(fig_combined, width='stretch')
                
                # 소속사별 평균 비교
                st.markdown("#### 소속사별 평균 팬덤 규모")
                company_avg = df.groupby('소속사')[main_metrics].mean().reset_index()
                company_avg_melted = company_avg.melt(id_vars=['소속사'], var_name='플랫폼 지표', value_name='평균 수치')
                
                fig_company = px.bar(
                    company_avg_melted,
                    x='소속사',
                    y='평균 수치', 
                    color='플랫폼 지표',
                    barmode='group',
                    title="소속사별 평균 팬덤 규모 비교",
                    labels={'평균 수치': '평균 팔로워/구독자 수 (명)'}
                )
                st.plotly_chart(fig_company, width='stretch')
                
        else:
            st.info("선택한 그룹들의 플랫폼 데이터가 없습니다. 데이터 수집을 실행해주세요.")
            
    else:
        st.info("💡 **K-Pop 그룹 비교 분석을 시작해보세요!**\n\n"
                "위에서 비교하고 싶은 그룹들을 선택하시면:\n"
                "- 그룹별 YouTube 구독자/조회수 비교\n" 
                "- 그룹별 Spotify 팔로워/청취자 비교\n"
                "- 소속사별 평균 팬덤 규모 분석\n\n"
                "등의 인사이트를 확인할 수 있습니다!")