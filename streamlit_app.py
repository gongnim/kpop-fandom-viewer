import streamlit as st
import pandas as pd
from fanpower_tracker import collect

st.set_page_config(page_title="K-POP 팬덤 지표", layout="wide")

st.title("K-POP 아티스트 팬덤/콘텐츠 분석기")
st.markdown("아티스트 또는 그룹명을 입력하고 팬덤 지표를 확인하세요.")

input_text = st.text_input("예: BANGTANTV, BLACKPINK, IVE, NewJeans")

if st.button("조회하기") and input_text:
    with st.spinner("조회 중..."):
        artist_list = [x.strip() for x in input_text.split(",") if x.strip()]
        df = collect(artist_list)
        st.success("조회 완료")
        st.dataframe(df, use_container_width=True)
        st.download_button("결과 CSV 다운로드", df.to_csv(index=False), "fanpower.csv", "text/csv")
