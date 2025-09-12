# -*- coding: utf-8 -*-
"""
Advanced Chart Generator for K-POP Analytics Dashboard
Automated chart generation system with multiple output formats and interactive features
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import base64
import io
from dataclasses import dataclass
import colorsys
import json

# K-POP Brand Colors
KPOP_COLORS = {
    'primary': '#6366f1',
    'secondary': '#ec4899', 
    'accent': '#f59e0b',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'text_primary': '#1f2937',
    'text_secondary': '#6b7280',
    'bg_primary': '#ffffff',
    'bg_secondary': '#f9fafb',
    'border': '#e5e7eb'
}

# Color palettes for different chart types
COLOR_PALETTES = {
    'kpop_gradient': ['#6366f1', '#8b5cf6', '#a855f7', '#c084fc', '#d8b4fe'],
    'performance': ['#10b981', '#34d399', '#6ee7b7', '#9df3c4', '#c6f6d5'],
    'platforms': ['#ff0000', '#e4405f', '#000000', '#1db954', '#1da1f2'],
    'analytics': ['#6366f1', '#ec4899', '#f59e0b', '#10b981', '#8b5cf6'],
    'heatmap': ['#dbeafe', '#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8'],
    'sequential': ['#f0f9ff', '#c7d2fe', '#a5b4fc', '#818cf8', '#6366f1']
}

@dataclass
class ChartConfig:
    """Chart configuration data class"""
    title: str
    chart_type: str
    width: int = 800
    height: int = 600
    theme: str = 'kpop'
    interactive: bool = True
    export_formats: List[str] = None
    color_palette: str = 'kpop_gradient'
    show_legend: bool = True
    responsive: bool = True

class ChartThemeManager:
    """Manage chart themes and styling"""
    
    def __init__(self):
        self.themes = {
            'kpop': self._create_kpop_theme(),
            'professional': self._create_professional_theme(),
            'dark': self._create_dark_theme(),
            'minimal': self._create_minimal_theme()
        }
    
    def _create_kpop_theme(self) -> dict:
        """Create K-POP themed styling"""
        return {
            'layout': {
                'paper_bgcolor': KPOP_COLORS['bg_primary'],
                'plot_bgcolor': KPOP_COLORS['bg_primary'],
                'font': {
                    'family': 'Inter, Noto Sans KR, sans-serif',
                    'size': 12,
                    'color': KPOP_COLORS['text_primary']
                },
                'title': {
                    'font': {
                        'family': 'Inter, Noto Sans KR, sans-serif',
                        'size': 18,
                        'color': KPOP_COLORS['primary']
                    },
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'colorway': COLOR_PALETTES['kpop_gradient'],
                'hovermode': 'closest',
                'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60}
            },
            'axes': {
                'xaxis': {
                    'gridcolor': KPOP_COLORS['border'],
                    'linecolor': KPOP_COLORS['border'],
                    'tickcolor': KPOP_COLORS['text_secondary'],
                    'tickfont': {'color': KPOP_COLORS['text_secondary']},
                    'titlefont': {'color': KPOP_COLORS['text_primary'], 'size': 14}
                },
                'yaxis': {
                    'gridcolor': KPOP_COLORS['border'],
                    'linecolor': KPOP_COLORS['border'],
                    'tickcolor': KPOP_COLORS['text_secondary'],
                    'tickfont': {'color': KPOP_COLORS['text_secondary']},
                    'titlefont': {'color': KPOP_COLORS['text_primary'], 'size': 14}
                }
            }
        }
    
    def _create_professional_theme(self) -> dict:
        """Create professional business theme"""
        return {
            'layout': {
                'paper_bgcolor': '#ffffff',
                'plot_bgcolor': '#ffffff',
                'font': {'family': 'Arial, sans-serif', 'size': 11, 'color': '#333333'},
                'title': {'font': {'size': 16, 'color': '#333333'}, 'x': 0.5},
                'colorway': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                'margin': {'l': 50, 'r': 50, 't': 70, 'b': 50}
            },
            'axes': {
                'xaxis': {'gridcolor': '#e0e0e0', 'linecolor': '#cccccc'},
                'yaxis': {'gridcolor': '#e0e0e0', 'linecolor': '#cccccc'}
            }
        }
    
    def _create_dark_theme(self) -> dict:
        """Create dark theme for presentations"""
        return {
            'layout': {
                'paper_bgcolor': '#1a1a1a',
                'plot_bgcolor': '#1a1a1a',
                'font': {'family': 'Inter, sans-serif', 'size': 12, 'color': '#ffffff'},
                'title': {'font': {'size': 18, 'color': '#ffffff'}, 'x': 0.5},
                'colorway': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'],
                'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60}
            },
            'axes': {
                'xaxis': {'gridcolor': '#333333', 'linecolor': '#666666', 'tickcolor': '#cccccc'},
                'yaxis': {'gridcolor': '#333333', 'linecolor': '#666666', 'tickcolor': '#cccccc'}
            }
        }
    
    def _create_minimal_theme(self) -> dict:
        """Create minimal clean theme"""
        return {
            'layout': {
                'paper_bgcolor': '#ffffff',
                'plot_bgcolor': '#ffffff',
                'font': {'family': 'Helvetica, sans-serif', 'size': 10, 'color': '#666666'},
                'title': {'font': {'size': 14, 'color': '#333333'}, 'x': 0.5},
                'colorway': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                'showlegend': False,
                'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40}
            },
            'axes': {
                'xaxis': {'gridcolor': '#f0f0f0', 'linecolor': '#dddddd', 'showgrid': False},
                'yaxis': {'gridcolor': '#f0f0f0', 'linecolor': '#dddddd', 'showgrid': False}
            }
        }
    
    def get_theme(self, theme_name: str = 'kpop') -> dict:
        """Get theme configuration"""
        return self.themes.get(theme_name, self.themes['kpop'])

class AdvancedChartGenerator:
    """Advanced chart generation system with multiple visualization types"""
    
    def __init__(self):
        self.theme_manager = ChartThemeManager()
        self.color_palettes = COLOR_PALETTES
        
    def create_performance_dashboard(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create comprehensive performance dashboard with multiple subplots"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['일일 성과 추이', '플랫폼별 성과', '아티스트 순위', '성장률 분석'],
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        # 1. Daily Performance Trend (Line Chart)
        daily_data = data.get('daily_performance', {})
        if daily_data:
            fig.add_trace(
                go.Scatter(
                    x=daily_data.get('dates', []),
                    y=daily_data.get('scores', []),
                    mode='lines+markers',
                    name='성과 점수',
                    line=dict(color=KPOP_COLORS['primary'], width=3),
                    marker=dict(color=KPOP_COLORS['secondary'], size=8)
                ),
                row=1, col=1
            )
            
            # Add secondary axis for engagement
            if 'engagement' in daily_data:
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.get('dates', []),
                        y=daily_data.get('engagement', []),
                        mode='lines',
                        name='참여율',
                        line=dict(color=KPOP_COLORS['success'], width=2, dash='dot'),
                        yaxis='y2'
                    ),
                    row=1, col=1, secondary_y=True
                )
        
        # 2. Platform Performance (Bar Chart)
        platform_data = data.get('platform_performance', {})
        if platform_data:
            fig.add_trace(
                go.Bar(
                    x=platform_data.get('platforms', []),
                    y=platform_data.get('followers', []),
                    name='팔로워 수',
                    marker=dict(color=COLOR_PALETTES['platforms'][:len(platform_data.get('platforms', []))])
                ),
                row=1, col=2
            )
        
        # 3. Artist Rankings (Scatter Plot)
        ranking_data = data.get('artist_rankings', {})
        if ranking_data:
            fig.add_trace(
                go.Scatter(
                    x=ranking_data.get('influence_score', []),
                    y=ranking_data.get('growth_rate', []),
                    mode='markers+text',
                    text=ranking_data.get('artist_names', []),
                    textposition='top center',
                    name='아티스트 포지션',
                    marker=dict(
                        size=ranking_data.get('marker_sizes', []),
                        color=ranking_data.get('performance_scores', []),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="성과 점수")
                    )
                ),
                row=2, col=1
            )
        
        # 4. Growth Distribution (Pie Chart)
        growth_data = data.get('growth_distribution', {})
        if growth_data:
            fig.add_trace(
                go.Pie(
                    labels=growth_data.get('categories', []),
                    values=growth_data.get('percentages', []),
                    name='성장 분포',
                    marker=dict(colors=COLOR_PALETTES['kpop_gradient'])
                ),
                row=2, col=2
            )
        
        # Apply theme
        theme = self.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_trend_analysis_chart(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create advanced trend analysis with multiple metrics"""
        
        fig = go.Figure()
        
        # Main trend line
        main_data = data.get('main_trend', {})
        if main_data:
            fig.add_trace(
                go.Scatter(
                    x=main_data.get('dates', []),
                    y=main_data.get('values', []),
                    mode='lines+markers',
                    name=main_data.get('name', '주요 지표'),
                    line=dict(color=KPOP_COLORS['primary'], width=4),
                    marker=dict(size=10, color=KPOP_COLORS['primary']),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: %{y:,.0f}<br>' +
                                 '<extra></extra>'
                )
            )
        
        # Additional trend lines
        additional_trends = data.get('additional_trends', [])
        colors = COLOR_PALETTES['analytics']
        
        for i, trend in enumerate(additional_trends):
            fig.add_trace(
                go.Scatter(
                    x=trend.get('dates', []),
                    y=trend.get('values', []),
                    mode='lines',
                    name=trend.get('name', f'지표 {i+1}'),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: %{y:,.0f}<br>' +
                                 '<extra></extra>'
                )
            )
        
        # Add trend annotations
        annotations = data.get('annotations', [])
        for annotation in annotations:
            fig.add_annotation(
                x=annotation.get('x'),
                y=annotation.get('y'),
                text=annotation.get('text'),
                showarrow=True,
                arrowhead=2,
                arrowcolor=KPOP_COLORS['accent'],
                bgcolor=KPOP_COLORS['bg_secondary'],
                bordercolor=KPOP_COLORS['border'],
                font=dict(color=KPOP_COLORS['text_primary'])
            )
        
        # Apply theme and styling
        theme = self.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis_title='기간',
            yaxis_title='지표값',
            hovermode='x unified'
        )
        
        return fig
    
    def create_heatmap_matrix(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create correlation heatmap matrix"""
        
        matrix_data = data.get('matrix', [])
        x_labels = data.get('x_labels', [])
        y_labels = data.get('y_labels', [])
        
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix_data,
                x=x_labels,
                y=y_labels,
                colorscale=COLOR_PALETTES['heatmap'],
                showscale=True,
                hoverongaps=False,
                hovertemplate='<b>%{y} vs %{x}</b><br>' +
                             'Correlation: %{z:.2f}<br>' +
                             '<extra></extra>'
            )
        )
        
        # Add text annotations
        if data.get('show_values', True):
            annotations = []
            for i, row in enumerate(matrix_data):
                for j, value in enumerate(row):
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=str(round(value, 2)),
                            showarrow=False,
                            font=dict(color='white' if abs(value) > 0.5 else 'black')
                        )
                    )
            fig.update_layout(annotations=annotations)
        
        theme = self.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_radar_chart(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create radar/spider chart for multi-dimensional analysis"""
        
        categories = data.get('categories', [])
        datasets = data.get('datasets', [])
        
        fig = go.Figure()
        
        colors = COLOR_PALETTES['kpop_gradient']
        
        for i, dataset in enumerate(datasets):
            fig.add_trace(
                go.Scatterpolar(
                    r=dataset.get('values', []) + [dataset.get('values', [])[0]],  # Close the shape
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=dataset.get('name', f'Dataset {i+1}'),
                    line=dict(color=colors[i % len(colors)]),
                    fillcolor=colors[i % len(colors)].replace('1)', '0.2)'),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 '%{theta}: %{r}<br>' +
                                 '<extra></extra>'
                )
            )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, data.get('max_value', 100)]
                )
            ),
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_waterfall_chart(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create waterfall chart for showing cumulative effects"""
        
        categories = data.get('categories', [])
        values = data.get('values', [])
        
        # Calculate cumulative values
        cumulative = [values[0]]
        for i in range(1, len(values)):
            cumulative.append(cumulative[-1] + values[i])
        
        fig = go.Figure()
        
        # Add bars
        colors = []
        for i, value in enumerate(values):
            if i == 0 or i == len(values) - 1:  # Start and end
                colors.append(KPOP_COLORS['text_primary'])
            elif value > 0:  # Positive
                colors.append(KPOP_COLORS['success'])
            else:  # Negative
                colors.append(KPOP_COLORS['error'])
        
        fig.add_trace(
            go.Waterfall(
                name="",
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(values) - 2) + ["total"],
                x=categories,
                textposition="outside",
                text=[f"{v:+.0f}" if i > 0 and i < len(values)-1 else f"{v:.0f}" 
                      for i, v in enumerate(values)],
                y=values,
                connector={"line": {"color": KPOP_COLORS['border']}},
                decreasing={"marker": {"color": KPOP_COLORS['error']}},
                increasing={"marker": {"color": KPOP_COLORS['success']}},
                totals={"marker": {"color": KPOP_COLORS['primary']}}
            )
        )
        
        theme = self.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=False
        )
        
        return fig
    
    def create_bubble_chart(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create bubble chart for 3D data visualization"""
        
        fig = go.Figure()
        
        datasets = data.get('datasets', [])
        colors = COLOR_PALETTES['analytics']
        
        for i, dataset in enumerate(datasets):
            fig.add_trace(
                go.Scatter(
                    x=dataset.get('x_values', []),
                    y=dataset.get('y_values', []),
                    mode='markers',
                    name=dataset.get('name', f'그룹 {i+1}'),
                    marker=dict(
                        size=dataset.get('sizes', []),
                        color=colors[i % len(colors)],
                        sizemode='diameter',
                        sizeref=data.get('size_ref', 1),
                        opacity=0.7,
                        line=dict(width=2, color='white')
                    ),
                    text=dataset.get('labels', []),
                    hovertemplate='<b>%{text}</b><br>' +
                                 f"{data.get('x_label', 'X')}: %{{x}}<br>" +
                                 f"{data.get('y_label', 'Y')}: %{{y}}<br>" +
                                 f"{data.get('size_label', 'Size')}: %{{marker.size}}<br>" +
                                 '<extra></extra>'
                )
            )
        
        theme = self.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis_title=data.get('x_label', 'X축'),
            yaxis_title=data.get('y_label', 'Y축')
        )
        
        return fig
    
    def create_gantt_chart(self, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
        """Create Gantt chart for timeline visualization"""
        
        tasks = data.get('tasks', [])
        
        fig = go.Figure()
        
        colors = COLOR_PALETTES['analytics']
        
        for i, task in enumerate(tasks):
            fig.add_trace(
                go.Scatter(
                    x=[task.get('start'), task.get('end')],
                    y=[task.get('name'), task.get('name')],
                    mode='lines+markers',
                    name=task.get('name'),
                    line=dict(color=colors[i % len(colors)], width=20),
                    marker=dict(size=12, color=colors[i % len(colors)]),
                    hovertemplate='<b>%{y}</b><br>' +
                                 'Start: %{x[0]}<br>' +
                                 'End: %{x[1]}<br>' +
                                 '<extra></extra>',
                    showlegend=False
                )
            )
        
        theme = self.theme_manager.get_theme(config.theme)
        fig.update_layout(theme['layout'])
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis_title='기간',
            yaxis_title='작업'
        )
        
        return fig
    
    def export_chart(self, fig: go.Figure, format: str, filename: str) -> str:
        """Export chart to various formats"""
        
        if format.lower() == 'html':
            return fig.to_html(filename)
        elif format.lower() == 'png':
            fig.write_image(filename, format='png', width=1200, height=800, scale=2)
            return filename
        elif format.lower() == 'pdf':
            fig.write_image(filename, format='pdf', width=1200, height=800)
            return filename
        elif format.lower() == 'svg':
            fig.write_image(filename, format='svg', width=1200, height=800)
            return filename
        elif format.lower() == 'json':
            with open(filename, 'w') as f:
                f.write(fig.to_json())
            return filename
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_chart_as_base64(self, fig: go.Figure, format: str = 'png') -> str:
        """Get chart as base64 encoded string for embedding"""
        
        img_bytes = fig.to_image(format=format, width=1200, height=800, scale=2)
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/{format};base64,{img_b64}"

class ChartTemplateLibrary:
    """Pre-configured chart templates for common analytics scenarios"""
    
    def __init__(self):
        self.generator = AdvancedChartGenerator()
    
    def get_kpi_dashboard_template(self) -> Dict[str, Any]:
        """Template for KPI dashboard"""
        return {
            'type': 'performance_dashboard',
            'config': ChartConfig(
                title='KPI 대시보드',
                chart_type='dashboard',
                width=1200,
                height=800,
                theme='kpop',
                color_palette='kpop_gradient'
            ),
            'sample_data': {
                'daily_performance': {
                    'dates': pd.date_range('2024-01-01', periods=30, freq='D').tolist(),
                    'scores': np.random.randint(80, 120, 30).tolist(),
                    'engagement': np.random.uniform(5, 15, 30).tolist()
                },
                'platform_performance': {
                    'platforms': ['YouTube', 'Instagram', 'TikTok', 'Spotify'],
                    'followers': [2800000, 1900000, 3200000, 1100000]
                }
            }
        }
    
    def get_growth_analysis_template(self) -> Dict[str, Any]:
        """Template for growth analysis"""
        return {
            'type': 'trend_analysis_chart',
            'config': ChartConfig(
                title='성장 분석',
                chart_type='line',
                width=1000,
                height=600,
                theme='kpop',
                color_palette='performance'
            ),
            'sample_data': {
                'main_trend': {
                    'dates': pd.date_range('2024-01-01', periods=12, freq='M').tolist(),
                    'values': np.cumsum(np.random.randint(5, 20, 12)).tolist(),
                    'name': '총 팔로워 성장'
                }
            }
        }
    
    def get_competitive_analysis_template(self) -> Dict[str, Any]:
        """Template for competitive analysis"""
        return {
            'type': 'bubble_chart',
            'config': ChartConfig(
                title='경쟁사 포지셔닝 분석',
                chart_type='bubble',
                width=900,
                height=700,
                theme='kpop',
                color_palette='analytics'
            ),
            'sample_data': {
                'datasets': [{
                    'x_values': [75, 85, 90, 65, 80],
                    'y_values': [20, 15, 25, 10, 18],
                    'sizes': [30, 40, 50, 25, 35],
                    'labels': ['HYBE', 'SM', 'YG', 'JYP', 'Others'],
                    'name': '기획사'
                }],
                'x_label': '시장 영향력',
                'y_label': '성장률 (%)',
                'size_label': '시장 점유율'
            }
        }

# Usage example and factory function
def create_chart(chart_type: str, data: Dict[str, Any], config: ChartConfig) -> go.Figure:
    """Factory function to create charts"""
    
    generator = AdvancedChartGenerator()
    
    chart_methods = {
        'performance_dashboard': generator.create_performance_dashboard,
        'trend_analysis': generator.create_trend_analysis_chart,
        'heatmap': generator.create_heatmap_matrix,
        'radar': generator.create_radar_chart,
        'waterfall': generator.create_waterfall_chart,
        'bubble': generator.create_bubble_chart,
        'gantt': generator.create_gantt_chart
    }
    
    if chart_type not in chart_methods:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    return chart_methods[chart_type](data, config)