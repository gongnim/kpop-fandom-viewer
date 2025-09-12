
import streamlit as st
import pandas as pd
from database_postgresql import get_all_artists_with_details, get_all_metrics_for_artist, get_all_metrics_for_group, get_events_for_artist
from utils.charts import create_time_series_chart

st.set_page_config(page_title="아티스트별 분석", page_icon="👤")

st.title("👤 아티스트별 분석")
st.write("분석하고 싶은 아티스트를 선택하여 개인과 그룹의 성장 추이를 함께 확인하세요.")

# 아티스트 선택
artists = get_all_artists_with_details()
if not artists:
    st.warning("먼저 '데이터 관리' 페이지에서 아티스트를 추가해주세요.")
else:
    # artist_options에 group_id와 group_name도 저장
    artist_options = {
        f"{artist['artist_name']} ({artist['group_name'] or '솔로'})": {
            'artist_id': artist['artist_id'],
            'group_id': artist.get('group_id'),
            'group_name': artist.get('group_name')
        } for artist in artists
    }
    selected_artist_display = st.selectbox("아티스트 선택", options=list(artist_options.keys()))
    
    if selected_artist_display:
        selection = artist_options[selected_artist_display]
        artist_id = selection['artist_id']
        group_id = selection['group_id']
        group_name = selection['group_name']
        selected_artist_name = selected_artist_display.split(' (')[0]
        
        st.header(f"📊 {selected_artist_name} (그룹: {group_name or '솔로'}) 성장 추이")
        
        # 1. 아티스트 개인 지표 가져오기
        artist_metrics = get_all_metrics_for_artist(artist_id)
        df_artist = pd.DataFrame(artist_metrics) if artist_metrics else pd.DataFrame()
        if not df_artist.empty:
            df_artist['legend'] = selected_artist_name

        # 2. 그룹 지표 가져오기 (그룹에 속한 경우)
        df_group = pd.DataFrame()
        if group_id:
            group_metrics = get_all_metrics_for_group(group_id)
            if group_metrics:
                df_group = pd.DataFrame(group_metrics)
                df_group['legend'] = group_name

        # 3. 데이터 통합 및 차트 생성
        if df_artist.empty and df_group.empty:
            st.info(f"{selected_artist_name}의 수집된 데이터가 없습니다. 데이터 수집을 실행해주세요.")
        else:
            df_combined = pd.concat([df_artist, df_group]).reset_index(drop=True)
            df_combined['collected_at'] = pd.to_datetime(df_combined['collected_at'])

            # 이벤트 데이터 가져오기
            events = get_events_for_artist(artist_id)
            events_df = pd.DataFrame(events) if events else pd.DataFrame()

            # 플랫폼별, 메트릭 타입별로 차트 생성
            platforms = df_combined['platform'].unique()
            for platform in platforms:
                st.subheader(f"{platform.capitalize()} 데이터")
                platform_df = df_combined[df_combined['platform'] == platform]
                metric_types = platform_df['metric_type'].unique()
                
                for metric_type in metric_types:
                    metric_df = platform_df[platform_df['metric_type'] == metric_type]
                    if not metric_df.empty:
                        chart_title = f"{selected_artist_name} & {group_name or ''} {platform.capitalize()} {metric_type.capitalize()}"
                        fig = create_time_series_chart(
                            metric_df.set_index('collected_at'), 
                            y_column='value', 
                            title=chart_title, 
                            color_column='legend', # 아티스트와 그룹을 색으로 구분
                            events_df=events_df
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info(f"{platform}의 {metric_type} 데이터가 없습니다.")
                st.markdown("--- ")
