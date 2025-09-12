# -*- coding: utf-8 -*-
"""
Chart Integration System for Report Templates
Seamless integration between chart generation and template systems
"""

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template
from typing import Dict, List, Any, Optional
import json
import base64
from datetime import datetime
import os
from ..utils.chart_generator import AdvancedChartGenerator, ChartConfig, ChartTemplateLibrary

class ChartIntegrationManager:
    """Manages chart integration across HTML and PDF templates"""
    
    def __init__(self):
        self.chart_generator = AdvancedChartGenerator()
        self.template_library = ChartTemplateLibrary()
        
        # Chart configuration for different report types
        self.chart_configs = {
            'weekly': {
                'performance_trend': ChartConfig(
                    title='주간 성과 추이',
                    chart_type='line',
                    width=800,
                    height=400,
                    theme='kpop'
                ),
                'platform_performance': ChartConfig(
                    title='플랫폼별 성과',
                    chart_type='bar',
                    width=700,
                    height=350,
                    theme='kpop'
                )
            },
            'monthly': {
                'growth_analysis': ChartConfig(
                    title='월간 성장 분석',
                    chart_type='trend_analysis',
                    width=900,
                    height=500,
                    theme='kpop'
                ),
                'portfolio_matrix': ChartConfig(
                    title='포트폴리오 매트릭스',
                    chart_type='bubble',
                    width=800,
                    height=600,
                    theme='kpop'
                ),
                'market_share': ChartConfig(
                    title='시장 점유율 분석',
                    chart_type='pie',
                    width=600,
                    height=500,
                    theme='kpop'
                )
            },
            'quarterly': {
                'strategic_overview': ChartConfig(
                    title='전략적 개요',
                    chart_type='performance_dashboard',
                    width=1200,
                    height=800,
                    theme='kpop'
                ),
                'competitive_positioning': ChartConfig(
                    title='경쟁사 포지셔닝',
                    chart_type='bubble',
                    width=1000,
                    height=700,
                    theme='kpop'
                ),
                'financial_waterfall': ChartConfig(
                    title='재무 성과 분석',
                    chart_type='waterfall',
                    width=900,
                    height=600,
                    theme='professional'
                )
            }
        }
    
    def generate_charts_for_report(self, report_type: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate all charts needed for a specific report type"""
        
        charts = {}
        
        if report_type not in self.chart_configs:
            return charts
        
        for chart_name, config in self.chart_configs[report_type].items():
            try:
                chart_data = self._prepare_chart_data(chart_name, data)
                fig = self._create_chart(config.chart_type, chart_data, config)
                
                # Convert to different formats
                charts[chart_name] = {
                    'html': fig.to_html(include_plotlyjs='cdn', div_id=f'chart_{chart_name}'),
                    'json': fig.to_json(),
                    'base64_png': self.chart_generator.get_chart_as_base64(fig, 'png'),
                    'base64_svg': self.chart_generator.get_chart_as_base64(fig, 'svg')
                }
                
            except Exception as e:
                print(f"Error generating chart {chart_name}: {str(e)}")
                charts[chart_name] = self._create_placeholder_chart(chart_name)
        
        return charts
    
    def _prepare_chart_data(self, chart_name: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data specific to each chart type"""
        
        data_preparers = {
            'performance_trend': self._prepare_performance_trend_data,
            'platform_performance': self._prepare_platform_data,
            'growth_analysis': self._prepare_growth_data,
            'portfolio_matrix': self._prepare_portfolio_data,
            'market_share': self._prepare_market_share_data,
            'strategic_overview': self._prepare_dashboard_data,
            'competitive_positioning': self._prepare_competitive_data,
            'financial_waterfall': self._prepare_financial_data
        }
        
        preparer = data_preparers.get(chart_name, self._prepare_default_data)
        return preparer(raw_data)
    
    def _prepare_performance_trend_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare performance trend data"""
        return {
            'main_trend': {
                'dates': raw_data.get('dates', []),
                'values': raw_data.get('daily_scores', [85, 92, 88, 95, 102, 98, 105]),
                'name': '일일 성과 점수'
            },
            'additional_trends': [
                {
                    'dates': raw_data.get('dates', []),
                    'values': raw_data.get('engagement_rates', [7.2, 8.1, 7.8, 8.5, 9.2, 8.8, 9.5]),
                    'name': '참여율'
                }
            ],
            'annotations': raw_data.get('annotations', [])
        }
    
    def _prepare_platform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare platform performance data"""
        return {
            'platforms': raw_data.get('platforms', ['YouTube', 'Instagram', 'TikTok', 'Spotify']),
            'values': raw_data.get('follower_counts', [28, 19, 32, 11]),
            'title': '플랫폼별 팔로워 수 (단위: 백만)'
        }
    
    def _prepare_growth_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare growth analysis data"""
        return {
            'main_trend': {
                'dates': raw_data.get('monthly_dates', []),
                'values': raw_data.get('cumulative_growth', []),
                'name': '누적 성장률'
            },
            'additional_trends': [
                {
                    'dates': raw_data.get('monthly_dates', []),
                    'values': raw_data.get('monthly_growth_rates', []),
                    'name': '월간 성장률'
                }
            ]
        }
    
    def _prepare_portfolio_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare portfolio matrix data"""
        return {
            'datasets': [{
                'x_values': raw_data.get('market_influence', [75, 85, 90, 65, 80]),
                'y_values': raw_data.get('growth_rates', [20, 15, 25, 10, 18]),
                'sizes': raw_data.get('revenue_contribution', [30, 40, 50, 25, 35]),
                'labels': raw_data.get('artist_names', ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E']),
                'name': '아티스트'
            }],
            'x_label': '시장 영향력',
            'y_label': '성장률 (%)',
            'size_label': '수익 기여도',
            'size_ref': 2
        }
    
    def _prepare_market_share_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market share pie chart data"""
        return {
            'labels': raw_data.get('categories', ['국내 시장', '아시아 시장', '북미 시장', '유럽 시장', '기타']),
            'values': raw_data.get('percentages', [35, 28, 20, 12, 5]),
            'title': '지역별 시장 점유율'
        }
    
    def _prepare_dashboard_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive dashboard data"""
        return {
            'daily_performance': {
                'dates': raw_data.get('quarter_dates', []),
                'scores': raw_data.get('quarterly_scores', []),
                'engagement': raw_data.get('quarterly_engagement', [])
            },
            'platform_performance': {
                'platforms': ['YouTube', 'Instagram', 'TikTok', 'Spotify', 'Apple Music'],
                'followers': raw_data.get('platform_followers', [28, 19, 32, 11, 8])
            },
            'artist_rankings': {
                'influence_score': raw_data.get('influence_scores', []),
                'growth_rate': raw_data.get('artist_growth_rates', []),
                'artist_names': raw_data.get('top_artists', []),
                'marker_sizes': raw_data.get('marker_sizes', []),
                'performance_scores': raw_data.get('performance_scores', [])
            },
            'growth_distribution': {
                'categories': ['신인', '중견', '톱티어', '레전드'],
                'percentages': raw_data.get('tier_distribution', [25, 35, 30, 10])
            }
        }
    
    def _prepare_competitive_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare competitive analysis data"""
        return {
            'datasets': [{
                'x_values': raw_data.get('market_positions', [75, 85, 90, 65, 80, 70]),
                'y_values': raw_data.get('growth_rates', [20, 15, 25, 10, 18, 22]),
                'sizes': raw_data.get('market_caps', [50, 45, 60, 30, 40, 35]),
                'labels': raw_data.get('companies', ['HYBE', 'SM', 'YG', 'JYP', 'CUBE', 'Others']),
                'name': '기획사'
            }],
            'x_label': '시장 포지션',
            'y_label': '성장률 (%)',
            'size_label': '시가총액',
            'size_ref': 1.5
        }
    
    def _prepare_financial_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare financial waterfall data"""
        return {
            'categories': ['Q1 매출', '앨범 판매', '콘서트', '굿즈', '스트리밍', '광고', '기타', 'Q2 매출'],
            'values': raw_data.get('financial_breakdown', [1000, 200, 300, 150, 100, 80, 50, 1880])
        }
    
    def _prepare_default_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default data preparation"""
        return raw_data
    
    def _create_chart(self, chart_type: str, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create chart using the advanced chart generator"""
        
        chart_methods = {
            'line': lambda d, c: self.chart_generator.create_trend_analysis_chart(d, c),
            'bar': lambda d, c: self._create_bar_chart(d, c),
            'pie': lambda d, c: self._create_pie_chart(d, c),
            'trend_analysis': lambda d, c: self.chart_generator.create_trend_analysis_chart(d, c),
            'bubble': lambda d, c: self.chart_generator.create_bubble_chart(d, c),
            'performance_dashboard': lambda d, c: self.chart_generator.create_performance_dashboard(d, c),
            'waterfall': lambda d, c: self.chart_generator.create_waterfall_chart(d, c),
            'heatmap': lambda d, c: self.chart_generator.create_heatmap_matrix(d, c),
            'radar': lambda d, c: self.chart_generator.create_radar_chart(d, c)
        }
        
        if chart_type not in chart_methods:
            return self._create_placeholder_figure(config.title)
        
        return chart_methods[chart_type](data, config)
    
    def _create_bar_chart(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create simple bar chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data.get('platforms', []),
            y=data.get('values', []),
            name=data.get('title', 'Data'),
            marker=dict(
                color=['#ff0000', '#e4405f', '#000000', '#1db954', '#1da1f2'][:len(data.get('platforms', []))]
            )
        ))
        
        theme = self.chart_generator.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def _create_pie_chart(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create simple pie chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=data.get('labels', []),
            values=data.get('values', []),
            name=data.get('title', 'Distribution')
        ))
        
        theme = self.chart_generator.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def _create_placeholder_chart(self, chart_name: str) -> Dict[str, str]:
        """Create placeholder when chart generation fails"""
        fig = self._create_placeholder_figure(f'차트 생성 오류: {chart_name}')
        
        return {
            'html': fig.to_html(include_plotlyjs='cdn'),
            'json': fig.to_json(),
            'base64_png': self.chart_generator.get_chart_as_base64(fig, 'png'),
            'base64_svg': self.chart_generator.get_chart_as_base64(fig, 'svg')
        }
    
    def _create_placeholder_figure(self, title: str) -> go.Figure:
        """Create placeholder figure"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"차트를 불러올 수 없습니다<br>{title}",
            showarrow=False,
            font=dict(size=16, color='gray'),
            xref="paper", yref="paper"
        )
        
        fig.update_layout(
            title=title,
            width=600,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def create_interactive_chart_html(self, chart_name: str, chart_data: Dict[str, str]) -> str:
        """Create interactive HTML for charts with controls"""
        
        html_template = """
        <div class="chart-container" id="container-{{ chart_name }}">
            <div class="chart-controls">
                <button onclick="toggleChartType('{{ chart_name }}')" class="chart-btn">
                    차트 타입 변경
                </button>
                <button onclick="exportChart('{{ chart_name }}')" class="chart-btn">
                    내보내기
                </button>
            </div>
            <div id="chart-{{ chart_name }}" class="chart-content">
                {{ chart_html|safe }}
            </div>
        </div>
        
        <script>
            function toggleChartType(chartName) {
                // Chart type toggle functionality
                console.log('Toggling chart type for:', chartName);
            }
            
            function exportChart(chartName) {
                // Export functionality
                const chartDiv = document.getElementById('chart-' + chartName);
                if (chartDiv && window.Plotly) {
                    Plotly.downloadImage(chartDiv, {
                        format: 'png',
                        width: 1200,
                        height: 800,
                        filename: chartName + '_chart'
                    });
                }
            }
        </script>
        """
        
        template = Template(html_template)
        return template.render(
            chart_name=chart_name,
            chart_html=chart_data.get('html', '')
        )
    
    def create_chart_summary(self, charts: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Create summary information about generated charts"""
        
        summary = {
            'total_charts': len(charts),
            'successful_charts': 0,
            'failed_charts': 0,
            'chart_list': [],
            'generation_time': datetime.now().isoformat()
        }
        
        for chart_name, chart_data in charts.items():
            if 'html' in chart_data and len(chart_data['html']) > 100:
                summary['successful_charts'] += 1
                status = 'success'
            else:
                summary['failed_charts'] += 1
                status = 'failed'
            
            summary['chart_list'].append({
                'name': chart_name,
                'status': status,
                'has_html': 'html' in chart_data,
                'has_json': 'json' in chart_data,
                'has_base64': 'base64_png' in chart_data
            })
        
        return summary

# Convenience functions for template integration
def generate_weekly_charts(data: Dict[str, Any]) -> Dict[str, str]:
    """Generate charts for weekly reports"""
    manager = ChartIntegrationManager()
    return manager.generate_charts_for_report('weekly', data)

def generate_monthly_charts(data: Dict[str, Any]) -> Dict[str, str]:
    """Generate charts for monthly reports"""
    manager = ChartIntegrationManager()
    return manager.generate_charts_for_report('monthly', data)

def generate_quarterly_charts(data: Dict[str, Any]) -> Dict[str, str]:
    """Generate charts for quarterly reports"""
    manager = ChartIntegrationManager()
    return manager.generate_charts_for_report('quarterly', data)