# -*- coding: utf-8 -*-
"""
PDF Report Styles for K-POP Analytics Dashboard
ReportLab styling system for professional PDF reports
"""

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch, cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.platypus.tableofcontents import TableOfContents
import os
from typing import Dict, Any

# K-POP Brand Colors
KPOP_COLORS = {
    'primary': HexColor('#6366f1'),           # Indigo
    'secondary': HexColor('#ec4899'),         # Pink
    'accent': HexColor('#f59e0b'),            # Amber
    'success': HexColor('#10b981'),           # Green
    'warning': HexColor('#f59e0b'),           # Amber
    'error': HexColor('#ef4444'),             # Red
    'text_primary': HexColor('#1f2937'),      # Dark Gray
    'text_secondary': HexColor('#6b7280'),    # Medium Gray
    'bg_primary': HexColor('#ffffff'),        # White
    'bg_secondary': HexColor('#f9fafb'),      # Light Gray
    'border': HexColor('#e5e7eb'),            # Light Gray Border
    'gradient_start': HexColor('#6366f1'),    # Primary gradient start
    'gradient_end': HexColor('#ec4899'),      # Primary gradient end
}

class PDFStyleManager:
    """Comprehensive PDF styling system for K-POP themed reports"""
    
    def __init__(self):
        self.colors = KPOP_COLORS
        self.styles = {}
        self._setup_fonts()
        self._create_styles()
    
    def _setup_fonts(self):
        """Setup custom fonts for PDF reports"""
        try:
            # Try to register system fonts if available
            font_paths = [
                '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # macOS Korean font
                '/Windows/Fonts/malgun.ttf',  # Windows Korean font
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'  # Linux Korean font
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('KoreanFont', font_path))
                        self.korean_font = 'KoreanFont'
                        break
                    except:
                        continue
            else:
                self.korean_font = 'Helvetica'  # Fallback to system font
                
        except Exception:
            self.korean_font = 'Helvetica'
    
    def _create_styles(self):
        """Create comprehensive paragraph and text styles"""
        base_styles = getSampleStyleSheet()
        
        # Header Styles
        self.styles['title'] = ParagraphStyle(
            'KPOPTitle',
            parent=base_styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            spaceAfter=20,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            borderColor=self.colors['primary'],
            borderWidth=2,
            borderPadding=10
        )
        
        self.styles['subtitle'] = ParagraphStyle(
            'KPOPSubtitle',
            parent=base_styles['Heading2'],
            fontName='Helvetica',
            fontSize=14,
            textColor=self.colors['text_secondary'],
            alignment=TA_CENTER,
            spaceAfter=15
        )
        
        self.styles['section_header'] = ParagraphStyle(
            'KPOPSectionHeader',
            parent=base_styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=self.colors['primary'],
            spaceAfter=12,
            spaceBefore=20,
            borderColor=self.colors['primary'],
            borderWidth=1,
            leftIndent=10,
            borderPadding=(5, 0, 5, 10)
        )
        
        self.styles['subsection_header'] = ParagraphStyle(
            'KPOPSubsectionHeader',
            parent=base_styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=self.colors['text_primary'],
            spaceAfter=8,
            spaceBefore=15
        )
        
        # Body Text Styles
        self.styles['normal'] = ParagraphStyle(
            'KPOPNormal',
            parent=base_styles['Normal'],
            fontName=self.korean_font,
            fontSize=10,
            textColor=self.colors['text_primary'],
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        self.styles['body_text'] = ParagraphStyle(
            'KPOPBodyText',
            parent=self.styles['normal'],
            fontSize=9,
            leading=12,
            spaceAfter=4
        )
        
        self.styles['small_text'] = ParagraphStyle(
            'KPOPSmallText',
            parent=self.styles['normal'],
            fontSize=8,
            textColor=self.colors['text_secondary'],
            spaceAfter=3
        )
        
        # Special Content Styles
        self.styles['metric_value'] = ParagraphStyle(
            'KPOPMetricValue',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=18,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            spaceAfter=2
        )
        
        self.styles['metric_label'] = ParagraphStyle(
            'KPOPMetricLabel',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=self.colors['text_secondary'],
            alignment=TA_CENTER,
            spaceAfter=8
        )
        
        self.styles['highlight_positive'] = ParagraphStyle(
            'KPOPHighlightPositive',
            parent=self.styles['body_text'],
            textColor=self.colors['success'],
            fontName='Helvetica-Bold',
            backgroundColor=Color(0.9, 0.98, 0.94)  # Light green background
        )
        
        self.styles['highlight_negative'] = ParagraphStyle(
            'KPOPHighlightNegative',
            parent=self.styles['body_text'],
            textColor=self.colors['error'],
            fontName='Helvetica-Bold',
            backgroundColor=Color(0.99, 0.95, 0.95)  # Light red background
        )
        
        self.styles['highlight_warning'] = ParagraphStyle(
            'KPOPHighlightWarning',
            parent=self.styles['body_text'],
            textColor=self.colors['warning'],
            fontName='Helvetica-Bold',
            backgroundColor=Color(1.0, 0.98, 0.9)  # Light yellow background
        )
        
        # Alert Box Styles
        self.styles['alert_info'] = ParagraphStyle(
            'KPOPAlertInfo',
            parent=self.styles['body_text'],
            borderColor=self.colors['primary'],
            borderWidth=1,
            borderPadding=10,
            backgroundColor=Color(0.96, 0.97, 1.0),
            leftIndent=15,
            rightIndent=15,
            spaceAfter=10
        )
        
        self.styles['alert_warning'] = ParagraphStyle(
            'KPOPAlertWarning',
            parent=self.styles['body_text'],
            borderColor=self.colors['warning'],
            borderWidth=1,
            borderPadding=10,
            backgroundColor=Color(1.0, 0.98, 0.9),
            leftIndent=15,
            rightIndent=15,
            spaceAfter=10
        )
        
        self.styles['alert_critical'] = ParagraphStyle(
            'KPOPAlertCritical',
            parent=self.styles['body_text'],
            borderColor=self.colors['error'],
            borderWidth=1,
            borderPadding=10,
            backgroundColor=Color(0.99, 0.95, 0.95),
            leftIndent=15,
            rightIndent=15,
            spaceAfter=10
        )
        
        # Footer Style
        self.styles['footer'] = ParagraphStyle(
            'KPOPFooter',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            textColor=self.colors['text_secondary'],
            alignment=TA_CENTER,
            spaceBefore=20
        )
        
        # Table of Contents Style
        self.styles['toc_heading'] = ParagraphStyle(
            'KPOPTOCHeading',
            parent=base_styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=self.colors['primary'],
            spaceAfter=15,
            alignment=TA_LEFT
        )
    
    def get_table_style(self, table_type: str = 'default') -> list:
        """Get table styling configuration"""
        base_style = [
            ('FONT', (0, 0), (-1, -1), self.korean_font, 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['bg_secondary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.colors['text_primary']),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]
        
        if table_type == 'metrics':
            base_style.extend([
                ('BACKGROUND', (0, 1), (-1, -1), self.colors['bg_primary']),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['bg_primary'], Color(0.98, 0.98, 0.99)]),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ])
        elif table_type == 'performance':
            base_style.extend([
                ('BACKGROUND', (0, 1), (-1, -1), self.colors['bg_primary']),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ])
        elif table_type == 'summary':
            base_style.extend([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors['bg_secondary']),
                ('GRID', (0, 0), (-1, -1), 1, self.colors['primary']),
                ('FONT', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ])
        
        return base_style
    
    def get_chart_frame_style(self) -> dict:
        """Get styling for chart frames"""
        return {
            'border_color': self.colors['border'],
            'border_width': 1,
            'background_color': self.colors['bg_primary'],
            'padding': 10
        }
    
    def create_metric_card_style(self) -> dict:
        """Create styling for metric cards"""
        return {
            'border_color': self.colors['primary'],
            'border_width': 2,
            'background_color': self.colors['bg_primary'],
            'corner_radius': 8,
            'padding': 15,
            'shadow_color': Color(0, 0, 0, alpha=0.1),
            'shadow_offset': (2, -2)
        }

class PDFPageTemplate:
    """Custom page template for K-POP themed reports"""
    
    def __init__(self, style_manager: PDFStyleManager):
        self.style_manager = style_manager
    
    def create_page_template(self, template_name: str = 'default') -> PageTemplate:
        """Create a custom page template"""
        if template_name == 'cover':
            return self._create_cover_template()
        elif template_name == 'report':
            return self._create_report_template()
        else:
            return self._create_default_template()
    
    def _create_default_template(self) -> PageTemplate:
        """Create default page template"""
        frame = Frame(
            0.75 * inch, 0.75 * inch,  # x, y
            7 * inch, 9.5 * inch,      # width, height
            leftPadding=0.5 * inch,
            bottomPadding=0.5 * inch,
            rightPadding=0.5 * inch,
            topPadding=0.5 * inch,
            showBoundary=0
        )
        
        return PageTemplate(
            id='default',
            frames=[frame],
            onPage=self._draw_default_page,
            pagesize=(8.5 * inch, 11 * inch)
        )
    
    def _create_cover_template(self) -> PageTemplate:
        """Create cover page template"""
        frame = Frame(
            1 * inch, 1 * inch,        # x, y
            6.5 * inch, 9 * inch,      # width, height
            leftPadding=0,
            bottomPadding=0,
            rightPadding=0,
            topPadding=0,
            showBoundary=0
        )
        
        return PageTemplate(
            id='cover',
            frames=[frame],
            onPage=self._draw_cover_page,
            pagesize=(8.5 * inch, 11 * inch)
        )
    
    def _create_report_template(self) -> PageTemplate:
        """Create report content template"""
        frame = Frame(
            0.75 * inch, 1 * inch,     # x, y
            7 * inch, 9 * inch,        # width, height
            leftPadding=0.3 * inch,
            bottomPadding=0.3 * inch,
            rightPadding=0.3 * inch,
            topPadding=0.3 * inch,
            showBoundary=0
        )
        
        return PageTemplate(
            id='report',
            frames=[frame],
            onPage=self._draw_report_page,
            pagesize=(8.5 * inch, 11 * inch)
        )
    
    def _draw_default_page(self, canvas, doc):
        """Draw default page layout"""
        self._draw_header(canvas, doc)
        self._draw_footer(canvas, doc)
    
    def _draw_cover_page(self, canvas, doc):
        """Draw cover page layout"""
        # Cover pages typically don't need headers/footers
        self._draw_cover_decoration(canvas, doc)
    
    def _draw_report_page(self, canvas, doc):
        """Draw report page layout"""
        self._draw_header(canvas, doc)
        self._draw_footer(canvas, doc)
        self._draw_page_border(canvas, doc)
    
    def _draw_header(self, canvas, doc):
        """Draw page header"""
        canvas.saveState()
        
        # Header background
        canvas.setFillColor(self.style_manager.colors['bg_secondary'])
        canvas.rect(0.5 * inch, 10.2 * inch, 7.5 * inch, 0.5 * inch, fill=1, stroke=0)
        
        # Header border
        canvas.setStrokeColor(self.style_manager.colors['primary'])
        canvas.setLineWidth(2)
        canvas.line(0.5 * inch, 10.2 * inch, 8 * inch, 10.2 * inch)
        
        # Header text
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(self.style_manager.colors['text_primary'])
        canvas.drawString(0.75 * inch, 10.35 * inch, "K-POP Analytics Dashboard")
        
        # Date/time
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(self.style_manager.colors['text_secondary'])
        canvas.drawRightString(7.75 * inch, 10.35 * inch, f"Generated: {current_time}")
        
        canvas.restoreState()
    
    def _draw_footer(self, canvas, doc):
        """Draw page footer"""
        canvas.saveState()
        
        # Footer border
        canvas.setStrokeColor(self.style_manager.colors['border'])
        canvas.setLineWidth(1)
        canvas.line(0.75 * inch, 0.75 * inch, 7.75 * inch, 0.75 * inch)
        
        # Page number
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(self.style_manager.colors['text_secondary'])
        canvas.drawCentredText(4.25 * inch, 0.5 * inch, f"Page {doc.page}")
        
        # Footer text
        canvas.setFont('Helvetica', 7)
        canvas.drawString(0.75 * inch, 0.3 * inch, "K-POP Analytics Dashboard - Confidential Report")
        canvas.drawRightString(7.75 * inch, 0.3 * inch, "analytics@kpop-dashboard.com")
        
        canvas.restoreState()
    
    def _draw_cover_decoration(self, canvas, doc):
        """Draw cover page decorations"""
        canvas.saveState()
        
        # Gradient-like effect using multiple rectangles
        colors = [
            Color(0.39, 0.4, 0.94, alpha=0.8),   # Primary color with transparency
            Color(0.93, 0.28, 0.6, alpha=0.6),   # Secondary color with transparency
            Color(0.96, 0.62, 0.04, alpha=0.4)   # Accent color with transparency
        ]
        
        for i, color in enumerate(colors):
            canvas.setFillColor(color)
            y_pos = 9 * inch - (i * 2.5 * inch)
            canvas.rect(0, y_pos, 8.5 * inch, 2.5 * inch, fill=1, stroke=0)
        
        canvas.restoreState()
    
    def _draw_page_border(self, canvas, doc):
        """Draw decorative page border"""
        canvas.saveState()
        
        # Thin border around content area
        canvas.setStrokeColor(self.style_manager.colors['primary'])
        canvas.setLineWidth(0.5)
        canvas.rect(0.5 * inch, 0.8 * inch, 7.5 * inch, 9.4 * inch, fill=0, stroke=1)
        
        canvas.restoreState()