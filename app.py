import streamlit as st
from database_postgresql import init_db, get_companies, get_listed_companies # Added get_listed_companies
from scheduler import start_scheduler
from logger_config import logger
from config import Config

# Import responsive styles
try:
    from assets.responsive_styles import get_responsive_css
    RESPONSIVE_STYLES_AVAILABLE = True
except ImportError:
    RESPONSIVE_STYLES_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="K-Pop Analytics Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info("========================================")
logger.info(" streamlit application started ")
logger.info("========================================")

# 스케줄러가 한 번만 실행되도록 session_state를 사용
if 'scheduler_started' not in st.session_state:
    logger.info("Starting scheduler...")
    start_scheduler()
    st.session_state.scheduler_started = True

def main():
    # Apply responsive styles
    if RESPONSIVE_STYLES_AVAILABLE:
        st.markdown(get_responsive_css(), unsafe_allow_html=True)
    
    # 디버그: 데이터베이스 설정 정보 출력
    logger.info("=== Database Configuration Debug ===")
    db_config = Config.debug_config()
    # st.sidebar.info(f"🔧 DB Host: {db_config['host']}:{db_config['port']}")
    
    # 앱 시작 시 데이터베이스 초기화
    init_db()
    
    st.title("🎵 K-Pop Entertainment Analysis Dashboard")
    st.caption("Welcome to the K-Pop Entertainment Analysis Dashboard.")

    # --- 사이드바 --- #
    with st.sidebar:
        st.header("필터 옵션")
        
        # 회사 선택
        companies = get_listed_companies() # Use get_listed_companies
        if not companies:
            st.warning("상장사 데이터가 없습니다. 데이터 관리 페이지에서 상장사를 추가해주세요.") # Updated warning message
            st.session_state.selected_company = None
        else:
            # 전체 회사 목록에서 이름만 추출하여 옵션으로 사용
            company_names = [c['name'] for c in companies]
            selected_company_name = st.selectbox(
                "회사 선택", 
                company_names,
                index=None,
                placeholder="회사를 선택하세요..."
            )
            # 선택된 회사 이름을 session_state에 저장하여 다른 페이지와 공유
            st.session_state.selected_company = selected_company_name
        
        st.info("사이드바에서 필터를 선택하여 데이터를 조회하세요.")

        # Admin Refresh Button
        if st.session_state.get("password_correct"):
            st.markdown("---")
            st.subheader("관리자 기능")
            if st.button("데이터 즉시 수집 (관리자용)"):
                from scheduler import collect_all_data # Import here to avoid circular dependency if scheduler imports app
                with st.spinner("데이터 수집 중..."):
                    collect_all_data()
                st.success("데이터 수집 완료! 페이지를 새로고침하여 최신 데이터를 확인하세요.")
                st.rerun() # Rerun the app to show updated data

    # --- 메인 대시보드 --- #
    st.header("종합 현황")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # To-Do: DB에서 실제 데이터 가져오기
        st.metric("전체 아티스트 수", "0", help="데이터베이스에 등록된 총 아티스트 수")
    with col2:
        # To-Do: DB에서 실제 데이터 가져오기
        st.metric("전체 플랫폼 구독자 합계", "0 M", help="YouTube, Spotify, Twitter 등 모든 플랫폼의 구독자/팔로워 합계")
    with col3:
        st.metric("데이터 수집 상태", "정상", "-5%")

    st.markdown("--- ")
    st.markdown("### 📊 페이지 안내")
    
    # 페이지를 카테고리별로 그룹화
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 분석 대시보드")
        st.write("""
        - **기업별 분석**: 엔터테인먼트사 소속 아티스트들의 데이터를 종합적으로 비교 분석합니다
        - **아티스트별 분석**: 특정 아티스트의 플랫폼별 성장 추이를 상세히 살펴봅니다  
        - **그룹별 분석**: 그룹 단위의 성과 및 멤버별 기여도를 분석합니다
        - **플랫폼 비교**: 여러 아티스트를 선택하여 플랫폼 간의 영향력을 비교합니다
        """)
        
        st.markdown("#### 📈 고급 분석")
        st.write("""
        - **성장률 분석**: K-Pop 아티스트의 플랫폼별 성장률을 다양한 방법으로 분석합니다
        - **경영진 KPI**: 경영진을 위한 핵심 성과 지표 및 전략적 인사이트를 제공합니다
        """)
    
    with col2:
        st.markdown("#### ⚙️ 관리 도구")
        st.write("""
        - **데이터 관리**: 대시보드에 사용될 회사, 그룹, 아티스트 정보를 직접 관리합니다
        - **이벤트 관리**: K-Pop 관련 이벤트 및 일정을 관리하고 분석에 활용합니다
        - **API 테스트**: 외부 API 연동 상태를 확인하고 테스트합니다
        """)
        
        st.markdown("#### 🎯 접근 방법")
        st.info("""
        💡 **팁**: 왼쪽 사이드바에서 회사를 선택한 후 각 분석 페이지를 이용하면 더 구체적인 인사이트를 얻을 수 있습니다.
        
        🎵 **새로운 기능**: 성장률 분석과 경영진 KPI 페이지가 추가되었습니다!
        """)

def check_password():
    """비밀번호를 확인하여 인증 상태를 반환합니다."""
    def password_entered():
        # 사용자가 입력한 비밀번호가 secrets.toml에 있는지 확인
        if st.session_state["password"] in st.secrets["passwords"].values():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # 비밀번호를 메모리에서 삭제
        else:
            st.session_state["password_correct"] = False

    # st.session_state에 "password_correct"가 없으면, 아직 인증 전
    if "password_correct" not in st.session_state:
        logger.info("New user connection. Requesting password.")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    # 비밀번호가 틀렸을 경우
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        logger.warning("Incorrect password attempt.")
        return False
    # 인증 성공
    else:
        logger.info("Password correct. Authentication successful.")
        return True

if __name__ == "__main__":
    if check_password():
        main()
