import streamlit as st
import pandas as pd
import datetime
from database_postgresql import (
    get_companies,
    add_company,
    get_groups,
    add_group,
    get_artists,
    add_artist,
    add_artist_account,
    get_all_artists_with_details # We will use this for a detailed view
)

st.set_page_config(page_title="데이터 관리", page_icon="⚙️")

st.title("⚙️ 데이터 관리")
st.write("대시보드에 사용될 기본 데이터를 관리합니다. 회사, 그룹, 아티스트, 그리고 각 아티스트의 플랫폼별 계정 정보를 추가할 수 있습니다.")

st.markdown("--- ")

# --- 회사 관리 --- #
st.subheader("🏢 회사 관리")
with st.expander("회사 관리 펼치기/접기"):
    # For simplicity, we only add root companies here. Parent-child relations are handled by the seed script.
    with st.form("company_form", clear_on_submit=True):
        new_company_name = st.text_input("회사 이름", placeholder="예: 안테나")
        submitted = st.form_submit_button("추가")
        if submitted and new_company_name:
            add_company(new_company_name)
            st.success(f"회사 '{new_company_name}'을(를) 추가했습니다.")
            st.rerun()
        elif submitted:
            st.warning("회사 이름을 입력해주세요.")

    st.write("**현재 회사 목록**")
    st.dataframe(pd.DataFrame(get_companies()), width='stretch', hide_index=True)

st.markdown("--- ")

# --- 그룹 관리 --- #
st.subheader("🎤 그룹 관리")
with st.expander("그룹 관리 펼치기/접기"):
    with st.form("group_form", clear_on_submit=True):
        companies = get_companies()
        company_dict = {c['name']: c['company_id'] for c in companies}
        
        selected_company_name = st.selectbox("소속사 선택", options=list(company_dict.keys()), index=None, placeholder="회사를 선택하세요")
        new_group_name = st.text_input("그룹 이름", placeholder="예: RIIZE")
        debut_date = st.date_input("데뷔 일자", value=None, min_value=datetime.date(1990, 1, 1), max_value=datetime.date.today())
        submitted = st.form_submit_button("추가")
        if submitted:
            if not selected_company_name or not new_group_name:
                st.warning("소속사와 그룹 이름을 모두 입력해주세요.")
            else:
                company_id = company_dict[selected_company_name]
                add_group(new_group_name, company_id, debut_date)
                st.success(f"그룹 '{new_group_name}'을(를) 추가했습니다.")
                st.rerun()

    st.write("**현재 그룹 목록**")
    st.dataframe(pd.DataFrame(get_groups()), width='stretch', hide_index=True)

st.markdown("--- ")

# --- 아티스트 관리 --- #
st.subheader("👤 아티스트 관리")
with st.expander("아티스트 관리 펼치기/접기"):
    with st.form("artist_form", clear_on_submit=True):
        groups = get_groups()
        group_dict = {g['name']: g['group_id'] for g in groups}
        selected_group_name = st.selectbox("그룹 선택", options=list(group_dict.keys()), index=None, placeholder="그룹을 선택하세요")
        
        st.text("아티스트 정보")
        name = st.text_input("활동명(영문)", placeholder="예: WONYOUNG")
        name_kr = st.text_input("활동명(한글)", placeholder="예: 원영")
        fullname_kr = st.text_input("본명(한글)", placeholder="예: 장원영")
        nationality_name = st.text_input("국적명", placeholder="예: 대한민국")
        nationality_code = st.text_input("국적코드", placeholder="예: KR")

        submitted = st.form_submit_button("추가")
        if submitted:
            if not selected_group_name or not name:
                st.warning("그룹과 아티스트 활동명(영문)을 모두 입력해주세요.")
            else:
                group_id = group_dict[selected_group_name]
                add_artist(name, name_kr, fullname_kr, group_id, nationality_name, nationality_code)
                st.success(f"아티스트 '{name}'을(를) 추가했습니다.")
                st.rerun()

    st.write("**현재 아티스트 목록**")
    st.dataframe(pd.DataFrame(get_all_artists_with_details()), width='stretch', hide_index=True)

st.markdown("--- ")

# --- 아티스트 계정 관리 --- #
st.subheader("🔗 아티스트 플랫폼 계정 관리")
with st.expander("아티스트 계정 관리 펼치기/접기"):
    with st.form("account_form", clear_on_submit=True):
        artists = get_artists()
        artist_dict = {f"{a['name']}": a['artist_id'] for a in artists}
        selected_artist_name = st.selectbox("아티스트 선택", options=list(artist_dict.keys()), index=None, placeholder="계정을 추가할 아티스트를 선택하세요")
        platform = st.selectbox("플랫폼 선택", options=["youtube", "spotify", "twitter"], index=None, placeholder="플랫폼을 선택하세요")
        account_identifier = st.text_input("플랫폼 계정 ID / 유저네임", placeholder="예: UC3IZKseVpdzPSBaWxBxundA")
        submitted = st.form_submit_button("추가")
        if submitted:
            if not selected_artist_name or not platform or not account_identifier:
                st.warning("모든 필드를 입력해주세요.")
            else:
                artist_id = artist_dict[selected_artist_name]
                add_artist_account(platform=platform, account_identifier=account_identifier, artist_id=artist_id)
                st.success(f"'{selected_artist_name}'의 계정을 추가했습니다.")
                st.rerun()

    # For simplicity, we don't display all accounts here. This can be a future enhancement.