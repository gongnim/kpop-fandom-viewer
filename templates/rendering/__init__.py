# -*- coding: utf-8 -*-
"""
Template Rendering System Package
Advanced template rendering engine with preview and caching capabilities
"""

from .template_engine import (
    KPOPTemplateEngine,
    RenderConfig,
    TemplateContext, 
    RenderResult,
    TemplateCache,
    TemplatePreviewServer,
    create_template_engine,
    render_weekly_report,
    render_monthly_report,
    start_preview_server
)

from .preview_dashboard import (
    TemplatePreviewDashboard,
    run_preview_dashboard
)

__all__ = [
    'KPOPTemplateEngine',
    'RenderConfig',
    'TemplateContext',
    'RenderResult', 
    'TemplateCache',
    'TemplatePreviewServer',
    'TemplatePreviewDashboard',
    'create_template_engine',
    'render_weekly_report',
    'render_monthly_report',
    'start_preview_server',
    'run_preview_dashboard'
]