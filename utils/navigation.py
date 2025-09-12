import streamlit as st

def navigate_to(page_name, **kwargs):
    """
    Streamlit 페이지로 이동하고 쿼리 파라미터를 설정합니다.
    
    Args:
        page_name (str): 이동할 페이지의 이름 (예: "아티스트별_분석").
                         Streamlit 멀티페이지 앱의 파일명과 일치해야 합니다.
        **kwargs: 쿼리 파라미터로 전달할 키-값 쌍 (예: artist_id=123).
    """
    # Streamlit 멀티페이지 앱에서 페이지 이동을 직접 제어하는 표준 방법은 없습니다.
    # 대신, 쿼리 파라미터를 설정하고 사용자에게 페이지를 수동으로 클릭하도록 안내하거나,
    # st.rerun()을 사용하여 앱을 다시 로드하고 URL을 변경하는 방식을 사용합니다.
    # 여기서는 쿼리 파라미터를 설정하고, 페이지를 다시 로드하여 URL을 변경하는 방식을 사용합니다.
    
    # 쿼리 파라미터 설정
    for key, value in kwargs.items():
        st.query_params[key] = value
        
    # 페이지 이동을 위한 URL 구성 (Streamlit의 내부 동작에 의존)
    # Streamlit은 페이지 파일명에 따라 URL을 생성합니다.
    # 예: 1_기업별_분석.py -> /기업별_분석
    # 이 부분은 Streamlit의 향후 업데이트에 따라 변경될 수 있습니다.
    
    # st.rerun()을 호출하여 앱을 다시 로드하고 URL 변경을 유도
    st.rerun()