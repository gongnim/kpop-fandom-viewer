# -*- coding: utf-8 -*-
"""
Interactive Preview Dashboard for Template System
Real-time template preview with live editing and theme switching
"""

from typing import Dict, List, Any, Optional
import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from .template_engine import KPOPTemplateEngine, RenderConfig, TemplateContext
from ..branding import KPOPBrandSystem

class TemplatePreviewDashboard:
    """Interactive Streamlit dashboard for template preview"""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent.parent
        self.engine = KPOPTemplateEngine(str(self.template_dir))
        self.brand_system = KPOPBrandSystem()
        
        # Initialize session state
        if 'current_template' not in st.session_state:
            st.session_state.current_template = 'reports/html/weekly_report.html'
        if 'current_theme' not in st.session_state:
            st.session_state.current_theme = 'kpop_vibrant'
        if 'preview_data' not in st.session_state:
            st.session_state.preview_data = self._generate_sample_data()
    
    def run(self):
        """Run the preview dashboard"""
        st.set_page_config(
            page_title="K-POP Analytics Template Preview",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        self._apply_custom_css()
        
        # Main header
        st.title("📊 K-POP Analytics Template Preview")
        st.markdown("실시간 템플릿 미리보기 및 편집 시스템")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 3])
        
        with col1:
            self._render_control_panel()
        
        with col2:
            self._render_preview_area()
        
        # Footer with statistics
        self._render_footer()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling"""
        theme = self.brand_system.get_theme(st.session_state.current_theme)
        colors = theme['colors']
        
        css = f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, {colors['background']} 0%, {colors['surface']} 100%);
        }}
        
        .main-header {{
            background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: {colors['background']};
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid {colors['primary']};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}
        
        .template-preview {{
            border: 2px solid {colors['border']};
            border-radius: 8px;
            background: {colors['background']};
            min-height: 600px;
        }}
        
        .success-message {{
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid {colors['success']};
            padding: 1rem;
            border-radius: 4px;
        }}
        
        .error-message {{
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid {colors['error']};
            padding: 1rem;
            border-radius: 4px;
        }}
        
        .sidebar-section {{
            background: {colors['surface']};
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("🎨 Template Settings")
            
            # Template selection
            st.subheader("템플릿 선택")
            available_templates = self.engine.get_available_templates()
            
            template_options = []
            for category, templates in available_templates.items():
                for template in templates:
                    template_options.append(f"{category}/{template}")
            
            selected_template = st.selectbox(
                "템플릿",
                template_options,
                index=template_options.index(f"html/{st.session_state.current_template.split('/')[-1]}") 
                if f"html/{st.session_state.current_template.split('/')[-1]}" in template_options else 0
            )
            
            if selected_template != st.session_state.current_template:
                st.session_state.current_template = selected_template.replace('html/', 'reports/html/')
                st.rerun()
            
            # Theme selection
            st.subheader("테마 설정")
            theme_options = list(self.brand_system.themes.keys())
            theme_names = {
                'kpop_vibrant': '🎨 K-POP Vibrant',
                'kpop_neon': '🌈 K-POP Neon', 
                'kpop_pastel': '🎀 K-POP Pastel',
                'kpop_luxury': '💎 K-POP Luxury'
            }
            
            selected_theme = st.selectbox(
                "테마",
                theme_options,
                format_func=lambda x: theme_names.get(x, x),
                index=theme_options.index(st.session_state.current_theme)
            )
            
            if selected_theme != st.session_state.current_theme:
                st.session_state.current_theme = selected_theme
                st.rerun()
            
            # Output format
            st.subheader("출력 형식")
            output_format = st.radio(
                "형식",
                ['html', 'pdf', 'json'],
                index=0
            )
            
            # Advanced settings
            with st.expander("고급 설정"):
                include_charts = st.checkbox("차트 포함", value=True)
                cache_enabled = st.checkbox("캐시 사용", value=True)
                responsive = st.checkbox("반응형 디자인", value=True)
            
            # Theme preview
            st.subheader("테마 미리보기")
            self._render_theme_preview()
            
            # Cache management
            st.subheader("캐시 관리")
            cache_stats = self.engine.get_cache_stats()
            st.metric("캐시 크기", f"{cache_stats['size']}/{cache_stats['max_size']}")
            st.metric("적중률", f"{cache_stats['hit_rate']:.1%}")
            
            if st.button("캐시 지우기"):
                self.engine.clear_cache()
                st.success("캐시가 지워졌습니다")
                st.rerun()
    
    def _render_control_panel(self):
        """Render control panel"""
        st.header("🛠️ 컨트롤 패널")
        
        # Data editing
        with st.expander("📊 샘플 데이터 편집", expanded=True):
            st.subheader("기본 정보")
            
            col1, col2 = st.columns(2)
            with col1:
                period_start = st.date_input(
                    "시작일",
                    value=datetime.now().date() - timedelta(days=7)
                )
            with col2:
                period_end = st.date_input(
                    "종료일", 
                    value=datetime.now().date()
                )
            
            # Update data
            if st.button("데이터 업데이트"):
                st.session_state.preview_data.update({
                    'period_start': period_start.isoformat(),
                    'period_end': period_end.isoformat()
                })
                st.success("데이터가 업데이트되었습니다")
                st.rerun()
            
            # Metric cards editing
            st.subheader("메트릭 카드")
            if 'summary_cards' in st.session_state.preview_data:
                for i, card in enumerate(st.session_state.preview_data['summary_cards']):
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            card['title'] = st.text_input(f"제목 {i+1}", value=card['title'])
                        with col2:
                            card['value'] = st.text_input(f"값 {i+1}", value=card['value'])
                        with col3:
                            card['change'] = st.number_input(f"변화율 {i+1}", value=card['change'])
            
            # Performance data
            st.subheader("성과 데이터")
            if st.checkbox("랜덤 데이터 생성"):
                st.session_state.preview_data = self._generate_random_data()
                st.success("랜덤 데이터가 생성되었습니다")
                st.rerun()
        
        # Template validation
        with st.expander("✅ 템플릿 검증"):
            if st.button("템플릿 검증"):
                is_valid, errors = self.engine.validate_template(st.session_state.current_template)
                
                if is_valid:
                    st.success("✅ 템플릿이 유효합니다")
                else:
                    st.error("❌ 템플릿 오류 발견")
                    for error in errors:
                        st.error(error)
        
        # Export options
        with st.expander("📥 내보내기 옵션"):
            export_format = st.selectbox("내보내기 형식", ['html', 'pdf', 'json'])
            
            if st.button("내보내기"):
                self._export_template(export_format)
    
    def _render_preview_area(self):
        """Render preview area"""
        st.header("👀 미리보기")
        
        # Render controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 새로고침"):
                st.rerun()
        with col2:
            auto_refresh = st.checkbox("자동 새로고침")
        with col3:
            if st.button("🔍 전체화면"):
                self._show_fullscreen_preview()
        
        # Template preview
        try:
            config = RenderConfig(
                template_name=st.session_state.current_template,
                output_format='html',
                theme=st.session_state.current_theme,
                include_charts=True,
                cache_enabled=True,
                preview_mode=True
            )
            
            context = TemplateContext(
                data=st.session_state.preview_data,
                metadata={
                    'report_type': self._get_report_type(st.session_state.current_template)
                }
            )
            
            result = self.engine.render_template(config, context)
            
            if result.success:
                # Show render stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("렌더링 시간", f"{result.render_time:.3f}초")
                with col2:
                    st.metric("캐시", "✅ Hit" if result.cache_hit else "❌ Miss")
                with col3:
                    st.metric("크기", f"{len(result.content):,} bytes")
                
                # Display preview
                st.markdown("---")
                
                # Render HTML in iframe for better isolation
                iframe_html = f"""
                <div class="template-preview">
                    <iframe srcdoc='{result.content.replace("'", "&apos;")}' 
                            width="100%" height="800" frameborder="0">
                    </iframe>
                </div>
                """
                st.markdown(iframe_html, unsafe_allow_html=True)
                
            else:
                st.error("❌ 템플릿 렌더링 실패")
                for error in result.errors:
                    st.error(error)
                
                # Show debugging info
                with st.expander("디버깅 정보"):
                    st.json({
                        'template': st.session_state.current_template,
                        'theme': st.session_state.current_theme,
                        'errors': result.errors,
                        'render_time': result.render_time
                    })
        
        except Exception as e:
            st.error(f"미리보기 오류: {str(e)}")
            st.exception(e)
        
        # Auto-refresh
        if auto_refresh:
            import time
            time.sleep(2)
            st.rerun()
    
    def _render_theme_preview(self):
        """Render theme color preview"""
        theme = self.brand_system.get_theme(st.session_state.current_theme)
        colors = theme['colors']
        
        # Create color palette
        color_items = []
        for name, color in colors.items():
            color_items.append({
                'name': name.replace('_', ' ').title(),
                'color': color,
                'contrast': '#ffffff' if self._is_dark_color(color) else '#000000'
            })
        
        # Display colors in a grid
        for i in range(0, len(color_items), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                if i < len(color_items):
                    item = color_items[i]
                    st.markdown(
                        f"""
                        <div style="background: {item['color']}; color: {item['contrast']}; 
                                    padding: 0.5rem; border-radius: 4px; text-align: center; margin-bottom: 0.5rem;">
                            {item['name']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with col2:
                if i + 1 < len(color_items):
                    item = color_items[i + 1]
                    st.markdown(
                        f"""
                        <div style="background: {item['color']}; color: {item['contrast']}; 
                                    padding: 0.5rem; border-radius: 4px; text-align: center; margin-bottom: 0.5rem;">
                            {item['name']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    def _render_footer(self):
        """Render footer with statistics"""
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            available_templates = self.engine.get_available_templates()
            total_templates = sum(len(templates) for templates in available_templates.values())
            st.metric("사용 가능한 템플릿", total_templates)
        
        with col2:
            st.metric("지원하는 테마", len(self.brand_system.themes))
        
        with col3:
            cache_stats = self.engine.get_cache_stats()
            st.metric("캐시 적중률", f"{cache_stats['hit_rate']:.1%}")
        
        with col4:
            st.metric("현재 시간", datetime.now().strftime("%H:%M:%S"))
        
        # Version info
        st.markdown(
            """
            <div style="text-align: center; color: #6b7280; font-size: 0.8rem; margin-top: 2rem;">
                K-POP Analytics Template Preview Dashboard v1.0<br>
                Built with Streamlit and Jinja2
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate comprehensive sample data"""
        return {
            'period_start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'period_end': datetime.now().strftime('%Y-%m-%d'),
            'report_type': 'weekly',
            'summary_cards': [
                {'title': '총 팔로워', 'value': '2.8M', 'change': 5.2, 'status': 'positive'},
                {'title': '참여율', 'value': '8.1%', 'change': -0.3, 'status': 'negative'},
                {'title': '조회수', 'value': '15.2M', 'change': 12.5, 'status': 'positive'},
                {'title': '신규 구독자', 'value': '45.2K', 'change': 8.7, 'status': 'positive'}
            ],
            'top_performers': [
                {'rank': 1, 'entity_name': 'NewJeans', 'metric_value': 15200000, 'change': 12.5, 'key_strengths': ['글로벌 인기']},
                {'rank': 2, 'entity_name': 'IVE', 'metric_value': 12800000, 'change': 8.3, 'key_strengths': ['국내 팬층']},
                {'rank': 3, 'entity_name': 'ITZY', 'metric_value': 11500000, 'change': 6.1, 'key_strengths': ['해외 진출']}
            ],
            'trends': {
                'follower_growth': {'change_percent': 5.2, 'trend': 'increasing'},
                'engagement_rate': {'change_percent': -0.3, 'trend': 'decreasing'}, 
                'content_uploads': {'change_percent': 15.0, 'trend': 'increasing'}
            },
            'platform_data': {
                'platforms': ['YouTube', 'Instagram', 'TikTok', 'Spotify'],
                'follower_counts': [28, 19, 32, 11],
                'title': '플랫폼별 팔로워 수 (단위: 백만)'
            },
            'attention_items': [
                {
                    'entity_name': 'Artist A',
                    'entity_type': 'Solo',
                    'issues': ['참여율 하락', '조회수 감소'],
                    'severity_score': 75,
                    'severity_level': 'warning',
                    'recommended_actions': ['콘텐츠 전략 재검토', '팬 소통 강화']
                }
            ],
            'portfolio_growth': 5.2,
            'top_contribution': 68,
            'engagement_rate': 7.8
        }
    
    def _generate_random_data(self) -> Dict[str, Any]:
        """Generate random sample data"""
        import random
        
        base_data = self._generate_sample_data()
        
        # Randomize metrics
        for card in base_data['summary_cards']:
            card['change'] = random.uniform(-10, 15)
            card['status'] = 'positive' if card['change'] > 0 else 'negative'
        
        # Randomize performer data
        for performer in base_data['top_performers']:
            performer['metric_value'] = random.randint(5000000, 20000000)
            performer['change'] = random.uniform(-5, 20)
        
        # Randomize platform data
        base_data['platform_data']['follower_counts'] = [
            random.randint(10, 50) for _ in range(4)
        ]
        
        return base_data
    
    def _get_report_type(self, template_name: str) -> str:
        """Extract report type from template name"""
        if 'weekly' in template_name:
            return 'weekly'
        elif 'monthly' in template_name:
            return 'monthly'
        elif 'quarterly' in template_name:
            return 'quarterly'
        else:
            return 'weekly'
    
    def _is_dark_color(self, color: str) -> bool:
        """Check if color is dark (for contrast calculation)"""
        if color.startswith('#'):
            color = color[1:]
        
        try:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16) 
            b = int(color[4:6], 16)
            
            # Calculate luminance
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return luminance < 0.5
        except:
            return False
    
    def _export_template(self, format_type: str):
        """Export template to specified format"""
        try:
            config = RenderConfig(
                template_name=st.session_state.current_template,
                output_format=format_type,
                theme=st.session_state.current_theme,
                include_charts=True,
                cache_enabled=False
            )
            
            context = TemplateContext(
                data=st.session_state.preview_data,
                metadata={
                    'report_type': self._get_report_type(st.session_state.current_template)
                }
            )
            
            result = self.engine.render_template(config, context)
            
            if result.success:
                # Provide download
                filename = f"kpop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
                
                if format_type == 'html':
                    st.download_button(
                        label=f"📥 {filename} 다운로드",
                        data=result.content,
                        file_name=filename,
                        mime='text/html'
                    )
                elif format_type == 'pdf':
                    st.download_button(
                        label=f"📥 {filename} 다운로드", 
                        data=result.content,
                        file_name=filename,
                        mime='application/pdf'
                    )
                elif format_type == 'json':
                    st.download_button(
                        label=f"📥 {filename} 다운로드",
                        data=result.content,
                        file_name=filename,
                        mime='application/json'
                    )
                
                st.success(f"✅ {format_type.upper()} 파일이 준비되었습니다")
            else:
                st.error("❌ 내보내기 실패")
                for error in result.errors:
                    st.error(error)
                    
        except Exception as e:
            st.error(f"내보내기 오류: {str(e)}")
    
    def _show_fullscreen_preview(self):
        """Show fullscreen preview in new tab"""
        st.info("전체화면 미리보기는 브라우저 새 탭에서 열립니다")
        
        # This would typically open a new window/tab
        # For now, we'll show a placeholder
        st.markdown("전체화면 미리보기 기능은 개발 중입니다")

# Main function for running the dashboard
def run_preview_dashboard():
    """Run the template preview dashboard"""
    dashboard = TemplatePreviewDashboard()
    dashboard.run()

if __name__ == "__main__":
    run_preview_dashboard()