# -*- coding: utf-8 -*-
"""
Template Rendering Engine for K-POP Analytics Dashboard
Advanced template rendering with preview, caching, and multi-format support
"""

from jinja2 import (
    Environment, FileSystemLoader, TemplateNotFound, 
    TemplateSyntaxError, BaseLoader, DictLoader
)
from jinja2.meta import find_undeclared_variables
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import webbrowser
import threading
import http.server
import socketserver
from pathlib import Path

from ..branding import KPOPBrandSystem, generate_theme_css
from ..reports.chart_integration import ChartIntegrationManager
from ..reports.pdf import KPOPPDFGenerator

@dataclass
class RenderConfig:
    """Template rendering configuration"""
    template_name: str
    output_format: str = 'html'  # html, pdf, json
    theme: str = 'kpop_vibrant'
    include_charts: bool = True
    cache_enabled: bool = True
    preview_mode: bool = False
    responsive: bool = True
    
@dataclass 
class TemplateContext:
    """Template rendering context"""
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    user_preferences: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.user_preferences is None:
            self.user_preferences = {}

@dataclass
class RenderResult:
    """Template rendering result"""
    success: bool
    content: str = ""
    format: str = ""
    file_path: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    render_time: float = 0.0
    cache_hit: bool = False
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class TemplateCache:
    """Template caching system"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def _generate_key(self, template_name: str, context_hash: str, config: RenderConfig) -> str:
        """Generate cache key"""
        key_data = f"{template_name}:{context_hash}:{config.theme}:{config.output_format}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cached item is expired"""
        return (time.time() - timestamp) > self.ttl
    
    def get(self, key: str) -> Optional[str]:
        """Get item from cache"""
        if key in self.cache:
            timestamp, content = self.cache[key]
            if not self._is_expired(timestamp):
                self.access_times[key] = time.time()
                return content
            else:
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key: str, content: str):
        """Set item in cache"""
        # Clean expired items
        self._cleanup_expired()
        
        # Evict LRU if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        timestamp = time.time()
        self.cache[key] = (timestamp, content)
        self.access_times[key] = timestamp
    
    def _cleanup_expired(self):
        """Remove expired items"""
        current_time = time.time()
        expired_keys = [
            key for key, (timestamp, _) in self.cache.items()
            if self._is_expired(timestamp)
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.access_times:
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_times.clear()

class KPOPTemplateEngine:
    """Advanced template rendering engine"""
    
    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self.cache = TemplateCache()
        self.brand_system = KPOPBrandSystem()
        self.chart_manager = ChartIntegrationManager()
        self.pdf_generator = KPOPPDFGenerator()
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters and functions
        self._setup_custom_filters()
        self._setup_global_functions()
        
        # Thread pool for async rendering
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _setup_custom_filters(self):
        """Setup custom Jinja2 filters"""
        
        def format_korean_number(value):
            """Format number with Korean units"""
            if not isinstance(value, (int, float)):
                return value
            
            if value >= 100000000:  # 억
                return f"{value/100000000:.1f}억"
            elif value >= 10000:    # 만
                return f"{value/10000:.1f}만"
            elif value >= 1000:     # 천
                return f"{value/1000:.1f}천"
            else:
                return str(int(value))
        
        def format_percentage(value, decimals=1):
            """Format percentage with proper sign"""
            if not isinstance(value, (int, float)):
                return value
            
            sign = "+" if value > 0 else ""
            return f"{sign}{value:.{decimals}f}%"
        
        def format_date_korean(value):
            """Format date in Korean style"""
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except:
                    return value
            
            if isinstance(value, datetime):
                return value.strftime("%Y년 %m월 %d일")
            return value
        
        def apply_tier_styling(tier_name):
            """Apply tier-specific styling"""
            from ..branding import BrandTier, apply_tier_styling
            try:
                tier = BrandTier(tier_name.lower())
                return apply_tier_styling(tier, "")
            except:
                return ""
        
        def get_platform_color(platform_name):
            """Get platform-specific color"""
            from ..branding import PlatformType
            try:
                platform = PlatformType(platform_name.lower())
                colors = self.brand_system.get_platform_styling(platform)
                return colors.get('primary', '#6b7280')
            except:
                return '#6b7280'
        
        def truncate_text(text, length=100):
            """Truncate text with ellipsis"""
            if len(text) <= length:
                return text
            return text[:length] + "..."
        
        # Register filters
        filters = {
            'korean_number': format_korean_number,
            'percentage': format_percentage,
            'korean_date': format_date_korean,
            'tier_styling': apply_tier_styling,
            'platform_color': get_platform_color,
            'truncate': truncate_text
        }
        
        for name, filter_func in filters.items():
            self.env.filters[name] = filter_func
    
    def _setup_global_functions(self):
        """Setup global template functions"""
        
        def get_theme_colors(theme_name='kpop_vibrant'):
            """Get theme colors"""
            theme = self.brand_system.get_theme(theme_name)
            return theme['colors']
        
        def get_brand_assets():
            """Get brand assets"""
            return self.brand_system.brand_assets.__dict__
        
        def format_currency(amount, currency='KRW'):
            """Format currency"""
            if currency == 'KRW':
                return f"₩{amount:,.0f}"
            elif currency == 'USD':
                return f"${amount:,.2f}"
            else:
                return f"{amount:,.2f} {currency}"
        
        def calculate_growth_rate(current, previous):
            """Calculate growth rate"""
            if previous == 0:
                return 0
            return ((current - previous) / previous) * 100
        
        def get_status_badge(value, thresholds=None):
            """Get status badge based on value and thresholds"""
            if thresholds is None:
                thresholds = {'good': 80, 'warning': 60}
            
            if value >= thresholds['good']:
                return 'positive'
            elif value >= thresholds['warning']:
                return 'neutral'
            else:
                return 'negative'
        
        # Register global functions
        globals_dict = {
            'get_theme_colors': get_theme_colors,
            'get_brand_assets': get_brand_assets,
            'format_currency': format_currency,
            'calculate_growth_rate': calculate_growth_rate,
            'get_status_badge': get_status_badge,
            'now': datetime.now,
            'today': datetime.now().date()
        }
        
        for name, func in globals_dict.items():
            self.env.globals[name] = func
    
    def validate_template(self, template_name: str) -> Tuple[bool, List[str]]:
        """Validate template syntax and variables"""
        errors = []
        
        try:
            # Load template
            template = self.env.get_template(template_name)
            
            # Parse template to find undeclared variables
            template_source = self.env.loader.get_source(self.env, template_name)[0]
            parsed = self.env.parse(template_source)
            undeclared = find_undeclared_variables(parsed)
            
            # Check for required variables
            required_vars = {'period_start', 'period_end', 'report_type'}
            missing_required = required_vars - set(self.env.globals.keys())
            
            if missing_required:
                errors.append(f"Missing required template variables: {missing_required}")
            
            if undeclared:
                # Filter out variables that are likely to be provided at runtime
                runtime_vars = {'data', 'charts', 'summary_cards', 'top_performers'}
                unexpected = undeclared - runtime_vars - set(self.env.globals.keys())
                if unexpected:
                    errors.append(f"Undeclared variables found: {unexpected}")
        
        except TemplateNotFound:
            errors.append(f"Template not found: {template_name}")
        except TemplateSyntaxError as e:
            errors.append(f"Template syntax error: {e}")
        except Exception as e:
            errors.append(f"Template validation error: {e}")
        
        return len(errors) == 0, errors
    
    def render_template(self, config: RenderConfig, context: TemplateContext) -> RenderResult:
        """Render template with given configuration and context"""
        start_time = time.time()
        
        try:
            # Validate template
            is_valid, validation_errors = self.validate_template(config.template_name)
            if not is_valid:
                return RenderResult(
                    success=False,
                    errors=validation_errors,
                    render_time=time.time() - start_time
                )
            
            # Generate context hash for caching
            context_data = {**context.data, **context.metadata}
            context_hash = hashlib.md5(
                json.dumps(context_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check cache
            cache_key = None
            if config.cache_enabled:
                cache_key = self.cache._generate_key(
                    config.template_name, context_hash, config
                )
                cached_content = self.cache.get(cache_key)
                if cached_content:
                    return RenderResult(
                        success=True,
                        content=cached_content,
                        format=config.output_format,
                        render_time=time.time() - start_time,
                        cache_hit=True
                    )
            
            # Prepare rendering context
            render_context = self._prepare_render_context(context, config)
            
            # Generate charts if needed
            if config.include_charts:
                charts = self.chart_manager.generate_charts_for_report(
                    context.metadata.get('report_type', 'weekly'),
                    context.data
                )
                render_context['charts'] = charts
            
            # Render template based on format
            if config.output_format == 'html':
                content = self._render_html(config, render_context)
            elif config.output_format == 'pdf':
                content = self._render_pdf(config, render_context)
            elif config.output_format == 'json':
                content = self._render_json(config, render_context)
            else:
                raise ValueError(f"Unsupported output format: {config.output_format}")
            
            # Cache result
            if config.cache_enabled and cache_key:
                self.cache.set(cache_key, content)
            
            return RenderResult(
                success=True,
                content=content,
                format=config.output_format,
                render_time=time.time() - start_time,
                cache_hit=False
            )
            
        except Exception as e:
            return RenderResult(
                success=False,
                errors=[str(e)],
                render_time=time.time() - start_time
            )
    
    def _prepare_render_context(self, context: TemplateContext, config: RenderConfig) -> Dict[str, Any]:
        """Prepare rendering context with additional variables"""
        render_context = {
            **context.data,
            **context.metadata,
            'theme': config.theme,
            'theme_colors': self.brand_system.get_theme(config.theme)['colors'],
            'brand_assets': self.brand_system.brand_assets.__dict__,
            'generated_at': context.timestamp,
            'user_preferences': context.user_preferences,
            'preview_mode': config.preview_mode,
            'responsive': config.responsive
        }
        
        # Add theme-specific CSS
        if config.output_format == 'html':
            render_context['theme_css'] = generate_theme_css(config.theme)
        
        return render_context
    
    def _render_html(self, config: RenderConfig, context: Dict[str, Any]) -> str:
        """Render HTML template"""
        template = self.env.get_template(config.template_name)
        return template.render(**context)
    
    def _render_pdf(self, config: RenderConfig, context: Dict[str, Any]) -> str:
        """Render PDF template"""
        # First render HTML content
        html_content = self._render_html(config, context)
        
        # Generate PDF using the PDF generator
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            report_type = context.get('report_type', 'weekly')
            if report_type == 'weekly':
                self.pdf_generator.generate_weekly_report(context, temp_path)
            elif report_type == 'monthly':
                self.pdf_generator.generate_monthly_report(context, temp_path)
            elif report_type == 'quarterly':
                self.pdf_generator.generate_quarterly_report(context, temp_path)
            
            with open(temp_path, 'rb') as f:
                pdf_content = f.read()
            
            os.unlink(temp_path)
            return pdf_content
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    def _render_json(self, config: RenderConfig, context: Dict[str, Any]) -> str:
        """Render JSON template"""
        # For JSON output, return the context data as JSON
        output_data = {
            'template': config.template_name,
            'theme': config.theme,
            'generated_at': context['generated_at'].isoformat(),
            'data': context
        }
        return json.dumps(output_data, indent=2, ensure_ascii=False, default=str)
    
    async def render_async(self, config: RenderConfig, context: TemplateContext) -> RenderResult:
        """Render template asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.render_template, 
            config, 
            context
        )
    
    def batch_render(self, render_requests: List[Tuple[RenderConfig, TemplateContext]]) -> List[RenderResult]:
        """Render multiple templates in batch"""
        results = []
        
        # Use thread pool for parallel rendering
        futures = []
        for config, context in render_requests:
            future = self.executor.submit(self.render_template, config, context)
            futures.append(future)
        
        # Collect results
        for future in futures:
            results.append(future.result())
        
        return results
    
    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get list of available templates by category"""
        templates = {
            'html': [],
            'pdf': [],
            'components': []
        }
        
        # Scan template directory
        for template_path in self.template_dir.rglob('*.html'):
            rel_path = template_path.relative_to(self.template_dir)
            template_name = str(rel_path)
            
            if 'html' in template_path.parts:
                templates['html'].append(template_name)
            elif 'pdf' in template_path.parts:
                templates['pdf'].append(template_name)
            elif 'components' in template_path.parts:
                templates['components'].append(template_name)
        
        return templates
    
    def clear_cache(self):
        """Clear template cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache.cache),
            'max_size': self.cache.max_size,
            'hit_rate': getattr(self.cache, 'hit_count', 0) / max(getattr(self.cache, 'total_requests', 1), 1),
            'ttl': self.cache.ttl
        }

class TemplatePreviewServer:
    """Live preview server for templates"""
    
    def __init__(self, template_engine: KPOPTemplateEngine, port: int = 8080):
        self.template_engine = template_engine
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start(self):
        """Start preview server"""
        handler = self._create_handler()
        
        try:
            self.server = socketserver.TCPServer(("", self.port), handler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            print(f"Preview server started at http://localhost:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to start preview server: {e}")
            return False
    
    def stop(self):
        """Stop preview server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join()
            print("Preview server stopped")
    
    def _create_handler(self):
        """Create HTTP request handler"""
        template_engine = self.template_engine
        
        class PreviewHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_preview_index()
                elif self.path.startswith('/preview/'):
                    self.send_template_preview()
                elif self.path.startswith('/api/'):
                    self.send_api_response()
                else:
                    super().do_GET()
            
            def send_preview_index(self):
                """Send preview index page"""
                templates = template_engine.get_available_templates()
                
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>K-POP Analytics - Template Preview</title>
                    <style>
                        body {{ font-family: 'Inter', sans-serif; margin: 40px; }}
                        .header {{ text-align: center; margin-bottom: 40px; }}
                        .template-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                        .template-card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; }}
                        .template-card h3 {{ color: #6366f1; margin-top: 0; }}
                        .template-list {{ list-style: none; padding: 0; }}
                        .template-item {{ margin: 10px 0; }}
                        .template-link {{ color: #ec4899; text-decoration: none; }}
                        .template-link:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>📊 K-POP Analytics Template Preview</h1>
                        <p>실시간 템플릿 미리보기 시스템</p>
                    </div>
                    <div class="template-grid">
                        <div class="template-card">
                            <h3>HTML 템플릿</h3>
                            <ul class="template-list">
                                {self._generate_template_links(templates['html'], 'html')}
                            </ul>
                        </div>
                        <div class="template-card">
                            <h3>PDF 템플릿</h3>
                            <ul class="template-list">
                                {self._generate_template_links(templates['pdf'], 'pdf')}
                            </ul>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            
            def _generate_template_links(self, templates, format_type):
                """Generate template links"""
                links = []
                for template in templates:
                    template_name = template.replace('.html', '')
                    links.append(
                        f'<li class="template-item">'
                        f'<a href="/preview/{format_type}/{template}" class="template-link">{template_name}</a>'
                        f'</li>'
                    )
                return '\n'.join(links)
            
            def send_template_preview(self):
                """Send template preview"""
                path_parts = self.path.strip('/').split('/')
                if len(path_parts) >= 3:
                    format_type = path_parts[1]
                    template_name = '/'.join(path_parts[2:])
                    
                    # Generate sample data
                    sample_data = self._generate_sample_data()
                    
                    # Create render config
                    config = RenderConfig(
                        template_name=template_name,
                        output_format=format_type,
                        preview_mode=True
                    )
                    
                    # Create context
                    context = TemplateContext(
                        data=sample_data,
                        metadata={'report_type': 'weekly'}
                    )
                    
                    # Render template
                    result = template_engine.render_template(config, context)
                    
                    if result.success:
                        if format_type == 'html':
                            self.send_response(200)
                            self.send_header('Content-type', 'text/html; charset=utf-8')
                            self.end_headers()
                            self.wfile.write(result.content.encode('utf-8'))
                        elif format_type == 'pdf':
                            self.send_response(200)
                            self.send_header('Content-type', 'application/pdf')
                            self.send_header('Content-Disposition', f'inline; filename="{template_name}.pdf"')
                            self.end_headers()
                            self.wfile.write(result.content)
                    else:
                        self.send_error(500, f"Template render error: {result.errors}")
                else:
                    self.send_error(404)
            
            def _generate_sample_data(self):
                """Generate sample data for preview"""
                return {
                    'period_start': '2024-01-01',
                    'period_end': '2024-01-07',
                    'summary_cards': [
                        {'title': '총 팔로워', 'value': '2.8M', 'change': 5.2, 'status': 'positive'},
                        {'title': '참여율', 'value': '8.1%', 'change': -0.3, 'status': 'negative'},
                        {'title': '조회수', 'value': '15.2M', 'change': 12.5, 'status': 'positive'},
                        {'title': '신규 구독자', 'value': '45.2K', 'change': 8.7, 'status': 'positive'}
                    ],
                    'top_performers': [
                        {'rank': 1, 'name': 'NewJeans', 'metric_value': 15200000, 'change': 12.5},
                        {'rank': 2, 'name': 'IVE', 'metric_value': 12800000, 'change': 8.3},
                        {'rank': 3, 'name': 'ITZY', 'metric_value': 11500000, 'change': 6.1}
                    ],
                    'platform_data': {
                        'platforms': ['YouTube', 'Instagram', 'TikTok', 'Spotify'],
                        'followers': [2800000, 1900000, 3200000, 1100000]
                    },
                    'trends': {
                        'follower_growth': {'change_percent': 5.2, 'trend': 'increasing'},
                        'engagement_rate': {'change_percent': -0.3, 'trend': 'decreasing'},
                        'content_uploads': {'change_percent': 15.0, 'trend': 'increasing'}
                    }
                }
            
            def send_api_response(self):
                """Send API response"""
                if self.path == '/api/templates':
                    templates = template_engine.get_available_templates()
                    response = json.dumps(templates, ensure_ascii=False)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(response.encode('utf-8'))
                elif self.path == '/api/cache/stats':
                    stats = template_engine.get_cache_stats()
                    response = json.dumps(stats, ensure_ascii=False)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(response.encode('utf-8'))
                else:
                    self.send_error(404)
        
        return PreviewHandler

# Convenience functions
def create_template_engine(template_dir: str) -> KPOPTemplateEngine:
    """Create template engine instance"""
    return KPOPTemplateEngine(template_dir)

def render_weekly_report(template_engine: KPOPTemplateEngine, data: Dict[str, Any]) -> RenderResult:
    """Render weekly report"""
    config = RenderConfig(
        template_name='reports/html/weekly_report.html',
        output_format='html'
    )
    context = TemplateContext(
        data=data,
        metadata={'report_type': 'weekly'}
    )
    return template_engine.render_template(config, context)

def render_monthly_report(template_engine: KPOPTemplateEngine, data: Dict[str, Any]) -> RenderResult:
    """Render monthly report"""
    config = RenderConfig(
        template_name='reports/html/monthly_report.html',
        output_format='html'
    )
    context = TemplateContext(
        data=data,
        metadata={'report_type': 'monthly'}
    )
    return template_engine.render_template(config, context)

def start_preview_server(template_dir: str, port: int = 8080) -> TemplatePreviewServer:
    """Start preview server"""
    engine = create_template_engine(template_dir)
    server = TemplatePreviewServer(engine, port)
    server.start()
    return server