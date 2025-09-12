# -*- coding: utf-8 -*-
"""
PDF Report Generator for K-POP Analytics Dashboard
Professional PDF generation using ReportLab with K-POP themed styling
"""

from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, NextPageTemplate, PageBreak,
    Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
    FrameBreak, Flowable, XBox
)
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.colors import Color, HexColor
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics import renderPDF
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime, timedelta
import io
import tempfile
from typing import Dict, List, Any, Optional
from .styles import PDFStyleManager, PDFPageTemplate

class PDFChartGenerator:
    """Generate charts for PDF reports using ReportLab graphics"""
    
    def __init__(self, style_manager: PDFStyleManager):
        self.style_manager = style_manager
        self.colors = style_manager.colors
    
    def create_performance_chart(self, data: Dict[str, Any], width: float = 6*inch, height: float = 3*inch) -> Drawing:
        """Create performance trend chart"""
        drawing = Drawing(width, height)
        
        chart = HorizontalLineChart()
        chart.x = 0.5 * inch
        chart.y = 0.5 * inch
        chart.width = width - 1 * inch
        chart.height = height - 1 * inch
        
        # Sample data structure
        chart.data = [
            data.get('daily_scores', [85, 92, 88, 95, 102, 98, 105])
        ]
        chart.categoryAxis.categoryNames = data.get('days', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Styling
        chart.lines[0].strokeColor = self.colors['primary']
        chart.lines[0].strokeWidth = 3
        chart.lines[0].symbol.kind = 'FilledCircle'
        chart.lines[0].symbol.size = 6
        chart.lines[0].symbol.fillColor = self.colors['secondary']
        
        # Axes styling
        chart.categoryAxis.labels.fontName = 'Helvetica'
        chart.categoryAxis.labels.fontSize = 8
        chart.valueAxis.labels.fontName = 'Helvetica'
        chart.valueAxis.labels.fontSize = 8
        
        # Grid
        chart.categoryAxis.visibleGrid = True
        chart.valueAxis.visibleGrid = True
        chart.categoryAxis.gridStrokeColor = colors.lightgrey
        chart.valueAxis.gridStrokeColor = colors.lightgrey
        
        drawing.add(chart)
        
        # Title
        title = String(width/2, height - 0.3*inch, data.get('title', 'Performance Trend'))
        title.fontName = 'Helvetica-Bold'
        title.fontSize = 12
        title.fillColor = self.colors['text_primary']
        title.textAnchor = 'middle'
        drawing.add(title)
        
        return drawing
    
    def create_platform_chart(self, data: Dict[str, Any], width: float = 6*inch, height: float = 3*inch) -> Drawing:
        """Create platform comparison bar chart"""
        drawing = Drawing(width, height)
        
        chart = VerticalBarChart()
        chart.x = 0.7 * inch
        chart.y = 0.5 * inch
        chart.width = width - 1.4 * inch
        chart.height = height - 1 * inch
        
        # Data
        platforms = data.get('platforms', ['YouTube', 'Instagram', 'TikTok', 'Spotify'])
        values = data.get('values', [28, 19, 32, 11])
        
        chart.data = [values]
        chart.categoryAxis.categoryNames = platforms
        
        # Colors for different platforms
        platform_colors = [
            HexColor('#ff0000'),  # YouTube red
            HexColor('#e4405f'),  # Instagram pink
            HexColor('#000000'),  # TikTok black
            HexColor('#1db954')   # Spotify green
        ]
        
        for i, color in enumerate(platform_colors[:len(values)]):
            chart.bars[0][i].fillColor = color
        
        # Styling
        chart.categoryAxis.labels.fontName = 'Helvetica'
        chart.categoryAxis.labels.fontSize = 8
        chart.categoryAxis.labels.angle = 45
        chart.valueAxis.labels.fontName = 'Helvetica'
        chart.valueAxis.labels.fontSize = 8
        
        # Grid
        chart.valueAxis.visibleGrid = True
        chart.valueAxis.gridStrokeColor = colors.lightgrey
        
        drawing.add(chart)
        
        # Title
        title = String(width/2, height - 0.3*inch, data.get('title', 'Platform Performance'))
        title.fontName = 'Helvetica-Bold'
        title.fontSize = 12
        title.fillColor = self.colors['text_primary']
        title.textAnchor = 'middle'
        drawing.add(title)
        
        return drawing
    
    def create_pie_chart(self, data: Dict[str, Any], width: float = 4*inch, height: float = 3*inch) -> Drawing:
        """Create pie chart for distribution analysis"""
        drawing = Drawing(width, height)
        
        pie = Pie()
        pie.x = 0.5 * inch
        pie.y = 0.5 * inch
        pie.width = 2.5 * inch
        pie.height = 2 * inch
        
        # Data
        pie.data = data.get('values', [30, 25, 20, 15, 10])
        pie.labels = data.get('labels', ['A', 'B', 'C', 'D', 'E'])
        
        # Colors
        pie.slices.fillColor = self.colors['primary']
        for i, slice in enumerate(pie.slices):
            colors_list = [self.colors['primary'], self.colors['secondary'], 
                          self.colors['accent'], self.colors['success'], self.colors['warning']]
            pie.slices[i].fillColor = colors_list[i % len(colors_list)]
        
        drawing.add(pie)
        
        # Legend
        legend = Legend()
        legend.x = 3.2 * inch
        legend.y = 1.5 * inch
        legend.dx = 8
        legend.dy = 8
        legend.fontName = 'Helvetica'
        legend.fontSize = 8
        legend.boxAnchor = 'w'
        legend.columnMaximum = 5
        legend.strokeWidth = 0
        legend.deltax = 75
        legend.deltay = 10
        legend.autoXPadding = 5
        legend.yGap = 0
        legend.dxTextSpace = 5
        legend.alignment = 'left'
        legend.dividerLines = 1
        legend.dividerOffsY = 4.5
        legend.subCols.rpad = 30
        
        legend.colorNamePairs = [(pie.slices[i].fillColor, pie.labels[i]) for i in range(len(pie.labels))]
        
        drawing.add(legend)
        
        # Title
        title = String(width/2, height - 0.3*inch, data.get('title', 'Distribution'))
        title.fontName = 'Helvetica-Bold'
        title.fontSize = 12
        title.fillColor = self.colors['text_primary']
        title.textAnchor = 'middle'
        drawing.add(title)
        
        return drawing

class MetricCard(Flowable):
    """Custom flowable for metric cards in PDF"""
    
    def __init__(self, title: str, value: str, change: str = "", 
                 status: str = "neutral", width: float = 2*inch, height: float = 1.5*inch):
        self.title = title
        self.value = value
        self.change = change
        self.status = status
        self.width = width
        self.height = height
        self.style_manager = PDFStyleManager()
    
    def draw(self):
        """Draw the metric card"""
        canvas = self.canv
        
        # Background
        canvas.setFillColor(self.style_manager.colors['bg_primary'])
        canvas.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        
        # Border based on status
        border_colors = {
            'positive': self.style_manager.colors['success'],
            'negative': self.style_manager.colors['error'],
            'neutral': self.style_manager.colors['warning'],
            'default': self.style_manager.colors['border']
        }
        
        canvas.setStrokeColor(border_colors.get(self.status, border_colors['default']))
        canvas.setLineWidth(2)
        canvas.rect(0, 0, self.width, self.height, fill=0, stroke=1)
        
        # Left accent bar
        canvas.setFillColor(border_colors.get(self.status, border_colors['default']))
        canvas.rect(0, 0, 4, self.height, fill=1, stroke=0)
        
        # Title
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(self.style_manager.colors['text_secondary'])
        canvas.drawCentredText(self.width/2, self.height - 0.3*inch, self.title)
        
        # Value
        canvas.setFont('Helvetica-Bold', 16)
        canvas.setFillColor(self.style_manager.colors['primary'])
        canvas.drawCentredText(self.width/2, self.height/2, self.value)
        
        # Change indicator
        if self.change:
            canvas.setFont('Helvetica', 8)
            change_color = border_colors.get(self.status, self.style_manager.colors['text_secondary'])
            canvas.setFillColor(change_color)
            canvas.drawCentredText(self.width/2, 0.2*inch, self.change)

class KPOPPDFGenerator:
    """Main PDF generator for K-POP analytics reports"""
    
    def __init__(self):
        self.style_manager = PDFStyleManager()
        self.page_template = PDFPageTemplate(self.style_manager)
        self.chart_generator = PDFChartGenerator(self.style_manager)
        self.styles = self.style_manager.styles
    
    def generate_weekly_report(self, data: Dict[str, Any], output_path: str) -> str:
        """Generate weekly performance report"""
        doc = BaseDocTemplate(
            output_path,
            pagesize=(8.5*inch, 11*inch),
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Add page templates
        doc.addPageTemplates([
            self.page_template.create_page_template('cover'),
            self.page_template.create_page_template('report')
        ])
        
        story = []
        
        # Cover page
        story.append(NextPageTemplate('cover'))
        story.extend(self._create_cover_page(data, 'weekly'))
        story.append(PageBreak())
        
        # Report content
        story.append(NextPageTemplate('report'))
        story.extend(self._create_executive_summary(data))
        story.extend(self._create_performance_section(data))
        story.extend(self._create_platform_analysis(data))
        story.extend(self._create_attention_areas(data))
        
        doc.build(story)
        return output_path
    
    def generate_monthly_report(self, data: Dict[str, Any], output_path: str) -> str:
        """Generate monthly strategic report"""
        doc = BaseDocTemplate(
            output_path,
            pagesize=(8.5*inch, 11*inch),
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        doc.addPageTemplates([
            self.page_template.create_page_template('cover'),
            self.page_template.create_page_template('report')
        ])
        
        story = []
        
        # Cover page
        story.append(NextPageTemplate('cover'))
        story.extend(self._create_cover_page(data, 'monthly'))
        story.append(PageBreak())
        
        # Report content
        story.append(NextPageTemplate('report'))
        story.extend(self._create_executive_summary(data))
        story.extend(self._create_monthly_highlights(data))
        story.extend(self._create_portfolio_analysis(data))
        story.extend(self._create_market_analysis(data))
        story.extend(self._create_strategic_recommendations(data))
        
        doc.build(story)
        return output_path
    
    def generate_quarterly_report(self, data: Dict[str, Any], output_path: str) -> str:
        """Generate quarterly strategic report"""
        doc = BaseDocTemplate(
            output_path,
            pagesize=(8.5*inch, 11*inch),
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        doc.addPageTemplates([
            self.page_template.create_page_template('cover'),
            self.page_template.create_page_template('report')
        ])
        
        story = []
        
        # Cover page
        story.append(NextPageTemplate('cover'))
        story.extend(self._create_cover_page(data, 'quarterly'))
        story.append(PageBreak())
        
        # Report content
        story.append(NextPageTemplate('report'))
        story.extend(self._create_ceo_briefing(data))
        story.extend(self._create_quarterly_highlights(data))
        story.extend(self._create_competitive_analysis(data))
        story.extend(self._create_financial_performance(data))
        story.extend(self._create_strategic_initiatives(data))
        
        doc.build(story)
        return output_path
    
    def _create_cover_page(self, data: Dict[str, Any], report_type: str) -> List:
        """Create report cover page"""
        story = []
        
        # Title
        title_map = {
            'weekly': 'K-POP Analytics 주간 리포트',
            'monthly': 'K-POP Analytics 월간 전략 리포트',
            'quarterly': 'K-POP Analytics 분기별 경영 리포트'
        }
        
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(title_map.get(report_type, 'K-POP Analytics Report'), self.styles['title']))
        story.append(Spacer(1, 0.5*inch))
        
        # Subtitle with period
        period_text = f"{data.get('period_start', '')} ~ {data.get('period_end', '')}"
        story.append(Paragraph(period_text, self.styles['subtitle']))
        story.append(Spacer(1, 1*inch))
        
        # Summary stats
        if 'summary_stats' in data:
            stats = data['summary_stats']
            story.append(Paragraph('리포트 요약', self.styles['section_header']))
            
            summary_data = [
                ['총 아티스트 수', str(stats.get('total_artists', 0))],
                ['분석 기간', period_text],
                ['주요 지표 수', str(stats.get('metrics_count', 0))],
                ['성과 개선율', f"+{stats.get('improvement_rate', 0)}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(self.style_manager.get_table_style('summary'))
            story.append(summary_table)
        
        story.append(Spacer(1, 2*inch))
        
        # Generation info
        gen_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
        story.append(Paragraph(f'생성 일시: {gen_time}', self.styles['footer']))
        story.append(Paragraph('K-POP Analytics Dashboard', self.styles['footer']))
        
        return story
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph('경영진 요약', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Metric cards
        if 'summary_cards' in data:
            cards_data = []
            for card in data['summary_cards'][:4]:  # Show top 4 cards
                metric_card = MetricCard(
                    title=card.get('title', ''),
                    value=card.get('value', ''),
                    change=card.get('change_text', ''),
                    status=card.get('status', 'neutral')
                )
                cards_data.append([metric_card])
            
            # Arrange cards in 2x2 grid
            if len(cards_data) >= 4:
                cards_table = Table([
                    [cards_data[0][0], cards_data[1][0]],
                    [cards_data[2][0], cards_data[3][0]]
                ], colWidths=[3.5*inch, 3.5*inch], rowHeights=[1.5*inch, 1.5*inch])
                story.append(cards_table)
                story.append(Spacer(1, 0.3*inch))
        
        # Key insights
        if 'key_insights' in data:
            story.append(Paragraph('주요 인사이트', self.styles['subsection_header']))
            for insight in data['key_insights'][:3]:
                story.append(Paragraph(f"• {insight}", self.styles['body_text']))
            story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_performance_section(self, data: Dict[str, Any]) -> List:
        """Create performance analysis section"""
        story = []
        
        story.append(Paragraph('성과 분석', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Performance chart
        if 'performance_trends' in data:
            chart = self.chart_generator.create_performance_chart(
                data['performance_trends'], width=6*inch, height=3*inch
            )
            story.append(chart)
            story.append(Spacer(1, 0.3*inch))
        
        # Top performers table
        if 'top_performers' in data:
            story.append(Paragraph('베스트 퍼포머', self.styles['subsection_header']))
            
            performers_data = [['순위', '아티스트', '지표값', '변화율']]
            for i, performer in enumerate(data['top_performers'][:5], 1):
                performers_data.append([
                    str(i),
                    performer.get('name', ''),
                    str(performer.get('metric_value', '')),
                    performer.get('change_text', '')
                ])
            
            performers_table = Table(performers_data, colWidths=[0.8*inch, 2*inch, 1.5*inch, 1.2*inch])
            performers_table.setStyle(self.style_manager.get_table_style('performance'))
            story.append(performers_table)
        
        return story
    
    def _create_platform_analysis(self, data: Dict[str, Any]) -> List:
        """Create platform analysis section"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph('플랫폼별 성과 분석', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Platform chart
        if 'platform_data' in data:
            chart = self.chart_generator.create_platform_chart(
                data['platform_data'], width=6*inch, height=3*inch
            )
            story.append(chart)
            story.append(Spacer(1, 0.3*inch))
        
        # Platform summary table
        platform_summary_data = [
            ['플랫폼', '팔로워 수', '주간 증가', '참여율']
        ]
        
        sample_platforms = [
            ['YouTube', '2.8M', '+5.2%', '8.1%'],
            ['Instagram', '1.9M', '+3.8%', '7.3%'],
            ['TikTok', '3.2M', '+8.7%', '12.4%'],
            ['Spotify', '1.1M', '+2.1%', '6.8%']
        ]
        
        platform_summary_data.extend(sample_platforms)
        
        platform_table = Table(platform_summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        platform_table.setStyle(self.style_manager.get_table_style('metrics'))
        story.append(platform_table)
        
        return story
    
    def _create_attention_areas(self, data: Dict[str, Any]) -> List:
        """Create attention needed areas section"""
        story = []
        
        if 'attention_items' in data and data['attention_items']:
            story.append(Paragraph('주의 필요 영역', self.styles['section_header']))
            story.append(Spacer(1, 0.2*inch))
            
            for item in data['attention_items'][:3]:
                # Alert box based on severity
                severity = item.get('severity_level', 'warning')
                style_name = f'alert_{severity}'
                if style_name in self.styles:
                    alert_style = self.styles[style_name]
                else:
                    alert_style = self.styles['alert_warning']
                
                alert_text = f"<b>{item.get('entity_name', '')}</b><br/>"
                alert_text += f"이슈: {', '.join(item.get('issues', []))}<br/>"
                alert_text += f"심각도: {item.get('severity_score', 0)}/100"
                
                story.append(Paragraph(alert_text, alert_style))
                story.append(Spacer(1, 0.1*inch))
        
        return story
    
    def _create_monthly_highlights(self, data: Dict[str, Any]) -> List:
        """Create monthly highlights section"""
        story = []
        
        story.append(Paragraph('월간 하이라이트', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        highlights = data.get('monthly_highlights', [
            "신규 뮤직비디오 5편이 평균 24시간 내 100만 조회수 달성",
            "글로벌 팬베이스 15% 증가, 특히 동남아시아 지역 성장 두드러짐",
            "브랜드 파트너십을 통한 수익 다각화 성공"
        ])
        
        for highlight in highlights:
            story.append(Paragraph(f"• {highlight}", self.styles['body_text']))
        
        return story
    
    def _create_portfolio_analysis(self, data: Dict[str, Any]) -> List:
        """Create portfolio analysis section"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph('포트폴리오 분석', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Portfolio distribution pie chart
        if 'portfolio_distribution' in data:
            chart = self.chart_generator.create_pie_chart(
                data['portfolio_distribution'], width=6*inch, height=3*inch
            )
            story.append(chart)
        
        return story
    
    def _create_market_analysis(self, data: Dict[str, Any]) -> List:
        """Create market analysis section"""
        story = []
        
        story.append(Paragraph('시장 분석', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        market_insights = data.get('market_insights', [
            "K-POP 글로벌 시장 규모 전년 대비 23% 성장",
            "스트리밍 플랫폼 중심의 음악 소비 패턴 지속",
            "Z세대 타겟 콘텐츠 수요 급증"
        ])
        
        for insight in market_insights:
            story.append(Paragraph(f"• {insight}", self.styles['body_text']))
        
        return story
    
    def _create_strategic_recommendations(self, data: Dict[str, Any]) -> List:
        """Create strategic recommendations section"""
        story = []
        
        story.append(Paragraph('전략적 권장사항', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        recommendations = data.get('recommendations', [
            "디지털 마케팅 예산 20% 증액하여 글로벌 확장 가속화",
            "데이터 분석 역량 강화를 통한 팬 인사이트 활용 확대",
            "신흥 시장(동남아시아, 남미) 진출 전략 수립"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['body_text']))
        
        return story
    
    def _create_ceo_briefing(self, data: Dict[str, Any]) -> List:
        """Create CEO briefing section for quarterly reports"""
        story = []
        
        story.append(Paragraph('CEO 브리핑', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Executive summary for CEO
        ceo_summary = data.get('ceo_summary', {
            'quarter_performance': '+18% 성장',
            'key_achievements': ['글로벌 확장 성공', '신규 아티스트 데뷔', '수익 다각화'],
            'strategic_focus': ['AI 기반 콘텐츠 제작', '글로벌 파트너십 확대', '팬 경험 혁신']
        })
        
        story.append(Paragraph(f'분기 성과: {ceo_summary["quarter_performance"]}', self.styles['metric_value']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph('주요 성과', self.styles['subsection_header']))
        for achievement in ceo_summary['key_achievements']:
            story.append(Paragraph(f"• {achievement}", self.styles['body_text']))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph('전략적 중점 영역', self.styles['subsection_header']))
        for focus in ceo_summary['strategic_focus']:
            story.append(Paragraph(f"• {focus}", self.styles['body_text']))
        
        return story
    
    def _create_quarterly_highlights(self, data: Dict[str, Any]) -> List:
        """Create quarterly highlights section"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph('분기별 하이라이트', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        highlights = data.get('quarterly_highlights', [
            "전 세계 15개국에서 콘서트 투어 성공적 완료",
            "스트리밍 누적 재생 수 10억 회 돌파",
            "글로벌 브랜드와의 전략적 파트너십 5건 체결"
        ])
        
        for highlight in highlights:
            story.append(Paragraph(f"• {highlight}", self.styles['highlight_positive']))
            story.append(Spacer(1, 0.1*inch))
        
        return story
    
    def _create_competitive_analysis(self, data: Dict[str, Any]) -> List:
        """Create competitive analysis section"""
        story = []
        
        story.append(Paragraph('경쟁사 분석', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Competitive positioning table
        competitive_data = [
            ['순위', '기획사', '시장 점유율', '성장률', '강점']
        ]
        
        sample_competitors = [
            ['1', 'HYBE', '23%', '+15%', '글로벌 확장'],
            ['2', 'SM Entertainment', '18%', '+8%', '아티스트 육성'],
            ['3', 'YG Entertainment', '15%', '+12%', '브랜드 가치'],
            ['4', 'JYP Entertainment', '14%', '+20%', '해외 진출']
        ]
        
        competitive_data.extend(sample_competitors)
        
        competitive_table = Table(competitive_data, colWidths=[0.8*inch, 1.5*inch, 1.2*inch, 1*inch, 1.5*inch])
        competitive_table.setStyle(self.style_manager.get_table_style('metrics'))
        story.append(competitive_table)
        
        return story
    
    def _create_financial_performance(self, data: Dict[str, Any]) -> List:
        """Create financial performance section"""
        story = []
        
        story.append(Paragraph('재무 성과', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        financial_data = [
            ['지표', 'Q1', 'Q2', 'Q3', '변화율']
        ]
        
        sample_financial = [
            ['매출액 (억원)', '1,250', '1,380', '1,520', '+21.6%'],
            ['영업이익 (억원)', '180', '205', '245', '+36.1%'],
            ['순이익 (억원)', '145', '165', '195', '+34.5%'],
            ['EBITDA (억원)', '220', '250', '290', '+31.8%']
        ]
        
        financial_data.extend(sample_financial)
        
        financial_table = Table(financial_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1.5*inch])
        financial_table.setStyle(self.style_manager.get_table_style('metrics'))
        story.append(financial_table)
        
        return story
    
    def _create_strategic_initiatives(self, data: Dict[str, Any]) -> List:
        """Create strategic initiatives section"""
        story = []
        
        story.append(Paragraph('전략적 이니셔티브', self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        initiatives = data.get('strategic_initiatives', [
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
            },
            {
                'title': 'ESG 경영 체계 도입',
                'status': 'progress',
                'timeline': '2024 Q3 완료',
                'impact': '브랜드 가치 및 투자 매력도 향상'
            }
        ])
        
        for initiative in initiatives:
            story.append(Paragraph(f"<b>{initiative['title']}</b>", self.styles['subsection_header']))
            story.append(Paragraph(f"상태: {initiative['status']}", self.styles['body_text']))
            story.append(Paragraph(f"일정: {initiative['timeline']}", self.styles['body_text']))
            story.append(Paragraph(f"기대효과: {initiative['impact']}", self.styles['body_text']))
            story.append(Spacer(1, 0.2*inch))
        
        return story