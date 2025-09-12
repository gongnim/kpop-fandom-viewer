import plotly.express as px
import pandas as pd

def create_time_series_chart(df: pd.DataFrame, y_column: str, title: str, color_column: str = None, events_df: pd.DataFrame = None):
    """
    Pandas DataFrame을 받아 시계열 라인 차트를 생성합니다.
    
    Args:
        df (pd.DataFrame): 인덱스가 날짜(datetime)인 데이터프레임
        y_column (str): Y축에 사용할 컬럼명
        title (str): 차트 제목
        color_column (str): 데이터를 구분할 색상 컬럼명 (선택 사항)
        events_df (pd.DataFrame): 이벤트 데이터프레임 (선택 사항). 'event_date', 'name', 'event_type' 컬럼 포함.
    """
    if y_column not in df.columns:
        return None
        
    fig = px.line(
        df,
        x=df.index,
        y=y_column,
        color=color_column, # 색상으로 데이터 구분
        title=title,
        labels={'x': '날짜', y_column: '수치', color_column: '범례'},
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title="",
        legend_title="플랫폼",
        title_x=0.5
    )

    # Add events as vertical lines and annotations
    if events_df is not None and not events_df.empty:
        for _, event in events_df.iterrows():
            event_date = pd.to_datetime(event['event_date'])
            event_name = event['name']
            event_type = event['event_type']
            
            fig.add_vline(
                x=event_date.timestamp() * 1000, # Plotly expects milliseconds for datetime
                line_width=1, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"{event_name} ({event_type})",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="red"
            )
    
    return fig
