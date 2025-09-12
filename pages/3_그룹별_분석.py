import streamlit as st
import pandas as pd
from database_postgresql import get_groups, get_all_metrics_for_group, get_events_for_group
from utils.charts import create_time_series_chart
from logger_config import logger

st.set_page_config(page_title="그룹별 분석", page_icon="👥")

st.title("👥 그룹별 분석")
st.write("분석하고 싶은 그룹을 선택하여 플랫폼별 성장 추이를 확인하세요.")

# 그룹 선택
groups = get_groups()
if not groups:
    st.warning("먼저 '데이터 관리' 페이지에서 그룹을 추가해주세요.")
else:
    group_options = {group['name']: group['group_id'] for group in groups}
    
    # 쿼리 파라미터에서 group_id 가져오기
    pre_selected_group_id = st.query_params.get("group_id")
    
    selected_group_name = None
    if pre_selected_group_id:
        for name, group_id in group_options.items():
            if str(group_id) == pre_selected_group_id:
                selected_group_name = name
                break
    
    selected_group_display = st.selectbox("그룹 선택", options=list(group_options.keys()), index=list(group_options.keys()).index(selected_group_name) if selected_group_name else 0)
    
    if selected_group_display:
        group_id = group_options[selected_group_display]
        
        st.header(f"📊 {selected_group_display} 성장 추이")
        
        # 그룹 전체의 지표 데이터 가져오기
        all_group_metrics = get_all_metrics_for_group(group_id)
        
        if not all_group_metrics:
            st.info(f"{selected_group_display} 그룹의 수집된 데이터가 없습니다. 데이터 수집을 실행해주세요.")
        else:
            df_aggregated = pd.DataFrame(all_group_metrics)
            df_aggregated['collected_at'] = pd.to_datetime(df_aggregated['collected_at'])
            
            # 그룹 이벤트 가져오기
            group_events = get_events_for_group(group_id)
            events_df = pd.DataFrame(group_events) if group_events else pd.DataFrame()
            
            # 플랫폼별, 메트릭 타입별로 데이터 분리하여 차트 생성
            platforms = df_aggregated['platform'].unique()
            for platform in platforms:
                platform_df = df_aggregated[df_aggregated['platform'] == platform]
                metric_types = platform_df['metric_type'].unique()
                st.subheader(f"{platform.capitalize()} 데이터")
                for metric_type in metric_types:
                    metric_df = platform_df[platform_df['metric_type'] == metric_type].set_index('collected_at')
                    if not metric_df.empty:
                        chart_title = f"{selected_group_display} {platform.capitalize()} {metric_type.capitalize()}"
                        fig = create_time_series_chart(metric_df, 'value', chart_title, events_df=events_df)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info(f"{platform}의 {metric_type} 데이터가 없습니다.")
                st.markdown("--- ")
