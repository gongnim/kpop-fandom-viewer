# -*- coding: utf-8 -*-
"""
Template System Demo Script
Demonstrates the complete template system with all features
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import webbrowser
import tempfile
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from templates.rendering import (
    create_template_engine, RenderConfig, TemplateContext, 
    start_preview_server, run_preview_dashboard
)
from templates.branding import export_all_themes
from templates.reports.chart_integration import generate_weekly_charts

class TemplateSystemDemo:
    """Comprehensive demo of the template system"""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent.parent
        self.engine = create_template_engine(str(self.template_dir))
        self.demo_data = self._create_demo_data()
    
    def run_full_demo(self):
        """Run complete system demonstration"""
        print("🎵 K-POP Analytics Template System Demo")
        print("=" * 50)
        
        # 1. System Overview
        self._show_system_overview()
        
        # 2. Template Validation
        self._demo_template_validation()
        
        # 3. Theme System
        self._demo_theme_system()
        
        # 4. Chart Generation
        self._demo_chart_generation()
        
        # 5. Template Rendering
        self._demo_template_rendering()
        
        # 6. Batch Operations
        self._demo_batch_rendering()
        
        # 7. Cache Performance
        self._demo_cache_performance()
        
        # 8. Export Capabilities
        self._demo_export_features()
        
        # 9. Preview Server
        self._demo_preview_server()
        
        print("\n✅ Demo completed successfully!")
        print("Template system is ready for production use.")
    
    def _show_system_overview(self):
        """Show system overview and capabilities"""
        print("\n📊 System Overview")
        print("-" * 30)
        
        # Available templates
        templates = self.engine.get_available_templates()
        total_templates = sum(len(t) for t in templates.values())
        
        print(f"Available Templates: {total_templates}")
        for category, template_list in templates.items():
            print(f"  {category.title()}: {len(template_list)} templates")
        
        # Theme information
        from templates.branding import KPOPBrandSystem
        brand_system = KPOPBrandSystem()
        print(f"Available Themes: {len(brand_system.themes)}")
        for theme_name in brand_system.themes.keys():
            print(f"  • {theme_name}")
        
        # Cache statistics
        cache_stats = self.engine.get_cache_stats()
        print(f"Cache Configuration: {cache_stats['max_size']} items, {cache_stats['ttl']}s TTL")
    
    def _demo_template_validation(self):
        """Demonstrate template validation"""
        print("\n✅ Template Validation")
        print("-" * 30)
        
        templates_to_validate = [
            'reports/html/weekly_report.html',
            'reports/html/monthly_report.html',
            'reports/html/quarterly_report.html'
        ]
        
        for template in templates_to_validate:
            is_valid, errors = self.engine.validate_template(template)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"{template}: {status}")
            
            if errors:
                for error in errors[:2]:  # Show first 2 errors
                    print(f"    • {error}")
    
    def _demo_theme_system(self):
        """Demonstrate theme system"""
        print("\n🎨 Theme System")
        print("-" * 30)
        
        from templates.branding import KPOPBrandSystem, generate_theme_css
        
        brand_system = KPOPBrandSystem()
        
        # Show theme colors
        for theme_name in ['kpop_vibrant', 'kpop_neon']:
            print(f"\nTheme: {theme_name}")
            theme = brand_system.get_theme(theme_name)
            colors = theme['colors']
            
            print(f"  Primary: {colors['primary']}")
            print(f"  Secondary: {colors['secondary']}")
            print(f"  Accent: {colors['accent']}")
        
        # Generate CSS for theme
        css_output = generate_theme_css('kpop_vibrant')
        print(f"\nGenerated CSS length: {len(css_output):,} characters")
        
        # Export all themes
        export_dir = tempfile.mkdtemp()
        export_all_themes(export_dir)
        print(f"All themes exported to: {export_dir}")
    
    def _demo_chart_generation(self):
        """Demonstrate chart generation"""
        print("\n📈 Chart Generation")
        print("-" * 30)
        
        # Generate charts for weekly report
        charts = generate_weekly_charts(self.demo_data)
        
        print(f"Generated {len(charts)} charts:")
        for chart_name, chart_data in charts.items():
            has_html = 'html' in chart_data
            has_json = 'json' in chart_data
            has_base64 = 'base64_png' in chart_data
            
            print(f"  • {chart_name}: HTML={has_html}, JSON={has_json}, Base64={has_base64}")
        
        # Chart integration stats
        print(f"Chart data size: {sum(len(str(v)) for v in charts.values()):,} bytes")
    
    def _demo_template_rendering(self):
        """Demonstrate template rendering"""
        print("\n🖨️ Template Rendering")
        print("-" * 30)
        
        # Render different report types
        report_configs = [
            ('weekly', 'reports/html/weekly_report.html'),
            ('monthly', 'reports/html/monthly_report.html'),
            ('quarterly', 'reports/html/quarterly_report.html')
        ]
        
        for report_type, template_name in report_configs:
            config = RenderConfig(
                template_name=template_name,
                output_format='html',
                theme='kpop_vibrant',
                include_charts=True,
                cache_enabled=True
            )
            
            context = TemplateContext(
                data=self.demo_data,
                metadata={'report_type': report_type}
            )
            
            result = self.engine.render_template(config, context)
            
            if result.success:
                print(f"  ✅ {report_type.title()} Report: {len(result.content):,} bytes, {result.render_time:.3f}s")
            else:
                print(f"  ❌ {report_type.title()} Report: {result.errors[0] if result.errors else 'Unknown error'}")
    
    def _demo_batch_rendering(self):
        """Demonstrate batch rendering"""
        print("\n⚡ Batch Rendering")
        print("-" * 30)
        
        # Prepare batch requests
        batch_requests = []
        
        themes = ['kpop_vibrant', 'kpop_neon', 'kpop_pastel']
        template_name = 'reports/html/weekly_report.html'
        
        for theme in themes:
            config = RenderConfig(
                template_name=template_name,
                output_format='html',
                theme=theme,
                include_charts=False,  # Faster for demo
                cache_enabled=True
            )
            
            context = TemplateContext(
                data=self.demo_data,
                metadata={'report_type': 'weekly'}
            )
            
            batch_requests.append((config, context))
        
        # Execute batch rendering
        start_time = datetime.now()
        results = self.engine.batch_render(batch_requests)
        batch_time = (datetime.now() - start_time).total_seconds()
        
        successful = sum(1 for r in results if r.success)
        total_size = sum(len(r.content) for r in results if r.success)
        
        print(f"Batch Results: {successful}/{len(results)} successful")
        print(f"Total Time: {batch_time:.3f}s")
        print(f"Total Size: {total_size:,} bytes")
        print(f"Average per template: {batch_time/len(results):.3f}s")
    
    def _demo_cache_performance(self):
        """Demonstrate cache performance"""
        print("\n⚡ Cache Performance")
        print("-" * 30)
        
        config = RenderConfig(
            template_name='reports/html/weekly_report.html',
            output_format='html',
            theme='kpop_vibrant',
            include_charts=False,
            cache_enabled=True
        )
        
        context = TemplateContext(
            data=self.demo_data,
            metadata={'report_type': 'weekly'}
        )
        
        # First render (cache miss)
        result1 = self.engine.render_template(config, context)
        print(f"First render: {result1.render_time:.3f}s (cache_hit={result1.cache_hit})")
        
        # Second render (cache hit)
        result2 = self.engine.render_template(config, context)
        print(f"Second render: {result2.render_time:.3f}s (cache_hit={result2.cache_hit})")
        
        # Performance improvement
        if result1.render_time > 0:
            speedup = result1.render_time / result2.render_time
            print(f"Cache speedup: {speedup:.1f}x faster")
        
        # Cache statistics
        cache_stats = self.engine.get_cache_stats()
        print(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    
    def _demo_export_features(self):
        """Demonstrate export capabilities"""
        print("\n📥 Export Features")
        print("-" * 30)
        
        export_formats = ['html', 'json']  # PDF requires additional setup
        
        for format_type in export_formats:
            config = RenderConfig(
                template_name='reports/html/weekly_report.html',
                output_format=format_type,
                theme='kpop_vibrant',
                include_charts=format_type == 'html',
                cache_enabled=False
            )
            
            context = TemplateContext(
                data=self.demo_data,
                metadata={'report_type': 'weekly'}
            )
            
            result = self.engine.render_template(config, context)
            
            if result.success:
                # Save to temporary file
                suffix = '.html' if format_type == 'html' else f'.{format_type}'
                with tempfile.NamedTemporaryFile(mode='w' if format_type != 'pdf' else 'wb', 
                                               suffix=suffix, delete=False) as f:
                    if format_type == 'pdf':
                        f.write(result.content)
                    else:
                        f.write(result.content)
                    temp_path = f.name
                
                file_size = os.path.getsize(temp_path)
                print(f"  ✅ {format_type.upper()}: {file_size:,} bytes → {temp_path}")
                
                # Clean up
                os.unlink(temp_path)
            else:
                print(f"  ❌ {format_type.upper()}: Export failed")
    
    def _demo_preview_server(self):
        """Demonstrate preview server"""
        print("\n🌐 Preview Server")
        print("-" * 30)
        
        print("Starting preview server...")
        
        # Note: This would normally start a server
        # For demo purposes, we'll just show the concept
        print("Preview server would be available at:")
        print("  • http://localhost:8080/ - Template gallery")
        print("  • http://localhost:8080/preview/html/weekly_report.html - Live preview")
        print("  • http://localhost:8080/api/templates - Template API")
        
        # In production, you would uncomment:
        # server = start_preview_server(str(self.template_dir))
        # input("Press Enter to stop preview server...")
        # server.stop()
        
        print("Preview server demo completed")
    
    def _create_demo_data(self) -> Dict[str, Any]:
        """Create comprehensive demo data"""
        base_date = datetime.now()
        
        return {
            'period_start': (base_date - timedelta(days=7)).strftime('%Y-%m-%d'),
            'period_end': base_date.strftime('%Y-%m-%d'),
            'report_type': 'weekly',
            'generated_at': base_date.isoformat(),
            
            # Summary metrics
            'summary_cards': [
                {
                    'title': '총 팔로워',
                    'value': '2.8M', 
                    'unit': '',
                    'change': 5.2,
                    'status': 'positive'
                },
                {
                    'title': '참여율',
                    'value': '8.1',
                    'unit': '%', 
                    'change': -0.3,
                    'status': 'negative'
                },
                {
                    'title': '조회수',
                    'value': '15.2M',
                    'unit': '',
                    'change': 12.5, 
                    'status': 'positive'
                },
                {
                    'title': '신규 구독자',
                    'value': '45.2K',
                    'unit': '',
                    'change': 8.7,
                    'status': 'positive'
                }
            ],
            
            # Top performers
            'top_performers': {
                'artists': [
                    {
                        'rank': 1,
                        'entity_name': 'NewJeans',
                        'entity_type': 'Group',
                        'metric_value': 15200000,
                        'metric_unit': 'views',
                        'change': 12.5,
                        'key_strengths': ['글로벌 인기', '젊은 팬층']
                    },
                    {
                        'rank': 2, 
                        'entity_name': 'IVE',
                        'entity_type': 'Group',
                        'metric_value': 12800000,
                        'metric_unit': 'views',
                        'change': 8.3,
                        'key_strengths': ['국내 팬층', '음악성']
                    },
                    {
                        'rank': 3,
                        'entity_name': 'ITZY', 
                        'entity_type': 'Group',
                        'metric_value': 11500000,
                        'metric_unit': 'views',
                        'change': 6.1,
                        'key_strengths': ['해외 진출', '퍼포먼스']
                    }
                ]
            },
            
            # Trend data
            'trends': {
                'follower_growth': {
                    'change_percent': 5.2,
                    'trend': 'increasing'
                },
                'engagement_rate': {
                    'change_percent': -0.3, 
                    'trend': 'decreasing'
                },
                'content_uploads': {
                    'change_percent': 15.0,
                    'trend': 'increasing'
                }
            },
            
            # Platform data
            'platform_data': {
                'platforms': ['YouTube', 'Instagram', 'TikTok', 'Spotify'],
                'values': [28, 19, 32, 11],
                'title': '플랫폼별 팔로워 수 (단위: 백만)'
            },
            
            # Performance data for charts
            'daily_scores': [85, 92, 88, 95, 102, 98, 105],
            'dates': [(base_date - timedelta(days=6-i)).strftime('%Y-%m-%d') for i in range(7)],
            'engagement_rates': [7.2, 8.1, 7.8, 8.5, 9.2, 8.8, 9.5],
            
            # Attention areas
            'attention_items': [
                {
                    'entity_name': '(G)I-DLE',
                    'entity_type': 'Group',
                    'issues': ['참여율 하락', '조회수 감소'],
                    'severity_score': 75,
                    'severity_level': 'warning',
                    'recommended_actions': [
                        '콘텐츠 전략 재검토',
                        '팬 소통 강화',
                        '새로운 플랫폼 활용'
                    ]
                }
            ],
            
            # Additional metrics
            'portfolio_growth': 5.2,
            'top_contribution': 68,
            'engagement_rate': 7.8,
            
            # Monthly/Quarterly specific data
            'monthly_highlights': [
                '신규 뮤직비디오 5편이 평균 24시간 내 100만 조회수 달성',
                '글로벌 팬베이스 15% 증가, 특히 동남아시아 지역 성장 두드러짐',
                '브랜드 파트너십을 통한 수익 다각화 성공'
            ],
            
            'quarterly_highlights': [
                '전 세계 15개국에서 콘서트 투어 성공적 완료', 
                '스트리밍 누적 재생 수 10억 회 돌파',
                '글로벌 브랜드와의 전략적 파트너십 5건 체결'
            ],
            
            'strategic_initiatives': [
                {
                    'title': 'AI 기반 음악 제작 플랫폼 구축',
                    'status': 'progress',
                    'timeline': '2024 Q4 완료 예정',
                    'impact': '제작 효율성 40% 향상 기대'
                },
                {
                    'title': '글로벌 팬 커뮤니티 플랫폼 런칭',
                    'status': 'planning', 
                    'timeline': '2025 Q1 시작',
                    'impact': '팬 참여도 25% 증가 목표'
                }
            ]
        }

def main():
    """Main demo function"""
    print("🎵 K-POP Analytics Template System")
    print("Choose demo mode:")
    print("1. Full System Demo")
    print("2. Interactive Preview Dashboard")
    print("3. Quick Template Test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        demo = TemplateSystemDemo()
        demo.run_full_demo()
    elif choice == '2':
        print("Starting interactive preview dashboard...")
        print("Note: This requires Streamlit to be installed")
        try:
            run_preview_dashboard()
        except ImportError:
            print("❌ Streamlit not installed. Install with: pip install streamlit")
    elif choice == '3':
        demo = TemplateSystemDemo()
        demo._demo_template_rendering()
    else:
        print("Invalid choice. Running full demo...")
        demo = TemplateSystemDemo()
        demo.run_full_demo()

if __name__ == "__main__":
    main()