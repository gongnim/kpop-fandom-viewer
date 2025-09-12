# -*- coding: utf-8 -*-
"""
Dynamic Style Generator for K-POP Analytics Dashboard
Generates CSS, SCSS, and custom styles based on brand system
"""

from typing import Dict, List, Any, Optional
from .brand_system import KPOPBrandSystem, BrandTier, PlatformType
import json
import os
from datetime import datetime

class KPOPStyleGenerator:
    """Generate dynamic styles for various components and themes"""
    
    def __init__(self):
        self.brand_system = KPOPBrandSystem()
        self.current_theme = 'kpop_vibrant'
    
    def set_theme(self, theme_name: str):
        """Set active theme"""
        if theme_name in self.brand_system.themes:
            self.current_theme = theme_name
    
    def generate_complete_css(self, theme_name: Optional[str] = None) -> str:
        """Generate complete CSS stylesheet"""
        theme = theme_name or self.current_theme
        theme_data = self.brand_system.get_theme(theme)
        
        css_parts = [
            self._generate_css_reset(),
            self._generate_css_variables(theme),
            self._generate_base_styles(theme_data),
            self._generate_component_styles(theme_data),
            self._generate_chart_styles(theme_data),
            self._generate_responsive_styles(),
            self._generate_utility_classes(theme_data),
            self._generate_animation_keyframes()
        ]
        
        return '\n\n'.join(css_parts)
    
    def _generate_css_reset(self) -> str:
        """Generate CSS reset styles"""
        return """/* K-POP Analytics CSS Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    font-size: 16px;
    scroll-behavior: smooth;
}

body {
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

button, input, select, textarea {
    font-family: inherit;
}

img {
    max-width: 100%;
    height: auto;
}

a {
    text-decoration: none;
    color: inherit;
}

ul, ol {
    list-style: none;
}"""
    
    def _generate_css_variables(self, theme_name: str) -> str:
        """Generate CSS custom properties"""
        theme = self.brand_system.get_theme(theme_name)
        variables = []
        
        # Color variables
        for key, value in theme['colors'].items():
            variables.append(f"  --color-{key.replace('_', '-')}: {value};")
        
        # Typography variables
        typography = theme['typography']
        variables.append(f"  --font-primary: '{typography['primary_font']}', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;")
        variables.append(f"  --font-secondary: '{typography['secondary_font']}', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;")
        variables.append(f"  --font-korean: '{typography['korean_font']}', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;")
        
        for size_key, size_value in typography['font_sizes'].items():
            variables.append(f"  --font-size-{size_key}: {size_value};")
        
        for weight_key, weight_value in typography['heading_weights'].items():
            variables.append(f"  --font-weight-{weight_key}: {weight_value};")
        
        # Spacing variables
        for space_key, space_value in theme['spacing'].items():
            variables.append(f"  --spacing-{space_key}: {space_value};")
        
        # Border radius variables
        for radius_key, radius_value in theme['border_radius'].items():
            variables.append(f"  --radius-{radius_key}: {radius_value};")
        
        # Shadow variables
        for shadow_key, shadow_value in theme['shadows'].items():
            variables.append(f"  --shadow-{shadow_key}: {shadow_value};")
        
        # Brand gradients
        for grad_key, grad_value in self.brand_system.brand_assets.gradients.items():
            variables.append(f"  --gradient-{grad_key.replace('_', '-')}: {grad_value};")
        
        return """/* CSS Custom Properties */
:root {
""" + '\n'.join(variables) + """
}"""
    
    def _generate_base_styles(self, theme_data: Dict[str, Any]) -> str:
        """Generate base styling"""
        return f"""/* Base Styles */
body {{
    font-family: var(--font-primary);
    font-size: var(--font-size-base);
    line-height: 1.6;
    color: var(--color-text-primary);
    background-color: var(--color-background);
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: var(--font-primary);
    font-weight: var(--font-weight-semibold);
    line-height: 1.25;
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-md);
}}

h1 {{ font-size: var(--font-size-3xl); font-weight: var(--font-weight-bold); }}
h2 {{ font-size: var(--font-size-2xl); font-weight: var(--font-weight-semibold); }}
h3 {{ font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold); }}
h4 {{ font-size: var(--font-size-lg); font-weight: var(--font-weight-medium); }}
h5 {{ font-size: var(--font-size-base); font-weight: var(--font-weight-medium); }}
h6 {{ font-size: var(--font-size-sm); font-weight: var(--font-weight-medium); }}

p {{
    margin-bottom: var(--spacing-md);
    color: var(--color-text-primary);
}}

.korean-text {{
    font-family: var(--font-korean);
}}

.text-primary {{ color: var(--color-primary); }}
.text-secondary {{ color: var(--color-secondary); }}
.text-accent {{ color: var(--color-accent); }}
.text-success {{ color: var(--color-success); }}
.text-warning {{ color: var(--color-warning); }}
.text-error {{ color: var(--color-error); }}

.bg-primary {{ background-color: var(--color-primary); }}
.bg-secondary {{ background-color: var(--color-secondary); }}
.bg-surface {{ background-color: var(--color-surface); }}"""
    
    def _generate_component_styles(self, theme_data: Dict[str, Any]) -> str:
        """Generate component styles"""
        return """/* Component Styles */

/* Buttons */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-sm) var(--spacing-md);
    border: none;
    border-radius: var(--radius-md);
    font-family: var(--font-primary);
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.btn-primary {
    background: var(--gradient-primary);
    color: white;
    box-shadow: var(--shadow-md);
}

.btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-secondary {
    background: var(--color-secondary);
    color: white;
}

.btn-outline {
    background: transparent;
    border: 2px solid var(--color-primary);
    color: var(--color-primary);
}

.btn-outline:hover:not(:disabled) {
    background: var(--color-primary);
    color: white;
}

/* Cards */
.card {
    background: var(--color-background);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-md);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl);
}

.card-elevated {
    background: var(--color-surface);
    border: none;
    box-shadow: var(--shadow-lg);
}

.card-header {
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    padding-bottom: var(--spacing-md);
    margin-bottom: var(--spacing-lg);
}

.card-title {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-xs);
}

/* Metric Cards */
.metric-card {
    background: var(--color-background);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--color-primary);
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.metric-card.positive::before {
    background: var(--color-success);
}

.metric-card.negative::before {
    background: var(--color-error);
}

.metric-card.neutral::before {
    background: var(--color-warning);
}

.metric-value {
    font-size: var(--font-size-3xl);
    font-weight: var(--font-weight-bold);
    color: var(--color-primary);
    margin-bottom: var(--spacing-xs);
}

.metric-label {
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
    margin-bottom: var(--spacing-sm);
}

.metric-change {
    font-size: var(--font-size-xs);
    font-weight: var(--font-weight-medium);
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--radius-full);
    display: inline-block;
}

.metric-change.positive {
    background: rgba(16, 185, 129, 0.1);
    color: var(--color-success);
}

.metric-change.negative {
    background: rgba(239, 68, 68, 0.1);
    color: var(--color-error);
}

/* Alerts */
.alert {
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    border: 1px solid;
    margin-bottom: var(--spacing-md);
}

.alert-info {
    background: rgba(99, 102, 241, 0.1);
    border-color: var(--color-primary);
    color: #4338ca;
}

.alert-success {
    background: rgba(16, 185, 129, 0.1);
    border-color: var(--color-success);
    color: #065f46;
}

.alert-warning {
    background: rgba(245, 158, 11, 0.1);
    border-color: var(--color-warning);
    color: #92400e;
}

.alert-error {
    background: rgba(239, 68, 68, 0.1);
    border-color: var(--color-error);
    color: #991b1b;
}

/* Tables */
.table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: var(--spacing-lg);
}

.table th,
.table td {
    padding: var(--spacing-sm) var(--spacing-md);
    text-align: left;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.table th {
    background: var(--color-surface);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
    font-size: var(--font-size-sm);
}

.table tbody tr:hover {
    background: rgba(99, 102, 241, 0.05);
}

/* Navigation */
.nav {
    display: flex;
    align-items: center;
    padding: var(--spacing-md) 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.nav-brand {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-bold);
    color: var(--color-primary);
}

.nav-links {
    display: flex;
    gap: var(--spacing-lg);
    margin-left: auto;
}

.nav-link {
    color: var(--color-text-secondary);
    font-weight: var(--font-weight-medium);
    transition: color 0.2s ease;
}

.nav-link:hover {
    color: var(--color-primary);
}

.nav-link.active {
    color: var(--color-primary);
}"""
    
    def _generate_chart_styles(self, theme_data: Dict[str, Any]) -> str:
        """Generate chart-specific styles"""
        return """/* Chart Styles */
.chart-container {
    background: var(--color-background);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
    box-shadow: var(--shadow-md);
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-md);
    padding-bottom: var(--spacing-sm);
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.chart-title {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
}

.chart-controls {
    display: flex;
    gap: var(--spacing-sm);
}

.chart-btn {
    padding: var(--spacing-xs) var(--spacing-sm);
    background: var(--color-surface);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: var(--radius-sm);
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
}

.chart-btn:hover {
    background: var(--color-primary);
    color: white;
}

.chart-content {
    position: relative;
    min-height: 300px;
}

.chart-loading {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: var(--color-text-secondary);
}

/* Platform badges */
.platform-badge {
    display: inline-flex;
    align-items: center;
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--radius-full);
    font-size: var(--font-size-xs);
    font-weight: var(--font-weight-semibold);
    color: white;
    margin-right: var(--spacing-xs);
}

.platform-badge.youtube {
    background: #ff0000;
}

.platform-badge.instagram {
    background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
}

.platform-badge.tiktok {
    background: #000000;
}

.platform-badge.spotify {
    background: #1db954;
}

/* Tier indicators */
.tier-indicator {
    display: inline-block;
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--radius-sm);
    font-size: var(--font-size-xs);
    font-weight: var(--font-weight-bold);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.tier-legendary {
    background: linear-gradient(135deg, #ffd700, #ffed4e);
    color: #1a1a1a;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
}

.tier-top-tier {
    background: linear-gradient(135deg, #c084fc, #ec4899);
    color: white;
    box-shadow: 0 0 15px rgba(192, 132, 252, 0.5);
}

.tier-rising-star {
    background: linear-gradient(135deg, #10b981, #34d399);
    color: white;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.5);
}

.tier-rookie {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    color: white;
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.5);
}

.tier-trainee {
    background: linear-gradient(135deg, #6b7280, #9ca3af);
    color: white;
}"""
    
    def _generate_responsive_styles(self) -> str:
        """Generate responsive styles"""
        breakpoints = self.brand_system.create_responsive_breakpoints()
        
        return f"""/* Responsive Styles */

/* Mobile Styles */
@media {breakpoints['mobile']} {{
    .container {{
        padding: var(--spacing-sm);
    }}
    
    h1 {{ font-size: var(--font-size-2xl); }}
    h2 {{ font-size: var(--font-size-xl); }}
    
    .metric-cards {{
        grid-template-columns: 1fr;
    }}
    
    .chart-container {{
        padding: var(--spacing-md);
    }}
    
    .nav-links {{
        display: none;
    }}
    
    .table {{
        font-size: var(--font-size-sm);
    }}
    
    .btn {{
        width: 100%;
        justify-content: center;
    }}
}}

/* Tablet Styles */
@media {breakpoints['tablet']} {{
    .container {{
        padding: var(--spacing-md);
    }}
    
    .metric-cards {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}

/* Desktop Styles */
@media {breakpoints['desktop']} {{
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: var(--spacing-lg);
    }}
    
    .metric-cards {{
        grid-template-columns: repeat(4, 1fr);
    }}
    
    .chart-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: var(--spacing-lg);
    }}
}}

/* Large Desktop */
@media {breakpoints['large']} {{
    .container {{
        max-width: 1400px;
    }}
    
    .chart-grid {{
        grid-template-columns: repeat(3, 1fr);
    }}
}}"""
    
    def _generate_utility_classes(self, theme_data: Dict[str, Any]) -> str:
        """Generate utility classes"""
        return """/* Utility Classes */

/* Spacing */
.m-0 { margin: 0; }
.m-1 { margin: var(--spacing-xs); }
.m-2 { margin: var(--spacing-sm); }
.m-3 { margin: var(--spacing-md); }
.m-4 { margin: var(--spacing-lg); }
.m-5 { margin: var(--spacing-xl); }

.p-0 { padding: 0; }
.p-1 { padding: var(--spacing-xs); }
.p-2 { padding: var(--spacing-sm); }
.p-3 { padding: var(--spacing-md); }
.p-4 { padding: var(--spacing-lg); }
.p-5 { padding: var(--spacing-xl); }

.mb-0 { margin-bottom: 0; }
.mb-1 { margin-bottom: var(--spacing-xs); }
.mb-2 { margin-bottom: var(--spacing-sm); }
.mb-3 { margin-bottom: var(--spacing-md); }
.mb-4 { margin-bottom: var(--spacing-lg); }
.mb-5 { margin-bottom: var(--spacing-xl); }

/* Text alignment */
.text-left { text-align: left; }
.text-center { text-align: center; }
.text-right { text-align: right; }

/* Font weights */
.font-light { font-weight: var(--font-weight-light); }
.font-normal { font-weight: var(--font-weight-regular); }
.font-medium { font-weight: var(--font-weight-medium); }
.font-semibold { font-weight: var(--font-weight-semibold); }
.font-bold { font-weight: var(--font-weight-bold); }

/* Display */
.d-none { display: none; }
.d-block { display: block; }
.d-flex { display: flex; }
.d-grid { display: grid; }

/* Flexbox */
.flex-row { flex-direction: row; }
.flex-col { flex-direction: column; }
.justify-start { justify-content: flex-start; }
.justify-center { justify-content: center; }
.justify-end { justify-content: flex-end; }
.justify-between { justify-content: space-between; }
.items-start { align-items: flex-start; }
.items-center { align-items: center; }
.items-end { align-items: flex-end; }

/* Grid */
.grid { display: grid; }
.grid-cols-1 { grid-template-columns: repeat(1, 1fr); }
.grid-cols-2 { grid-template-columns: repeat(2, 1fr); }
.grid-cols-3 { grid-template-columns: repeat(3, 1fr); }
.grid-cols-4 { grid-template-columns: repeat(4, 1fr); }
.gap-1 { gap: var(--spacing-xs); }
.gap-2 { gap: var(--spacing-sm); }
.gap-3 { gap: var(--spacing-md); }
.gap-4 { gap: var(--spacing-lg); }

/* Border radius */
.rounded-none { border-radius: var(--radius-none); }
.rounded-sm { border-radius: var(--radius-sm); }
.rounded { border-radius: var(--radius-md); }
.rounded-lg { border-radius: var(--radius-lg); }
.rounded-xl { border-radius: var(--radius-xl); }
.rounded-full { border-radius: var(--radius-full); }

/* Shadows */
.shadow-none { box-shadow: none; }
.shadow-sm { box-shadow: var(--shadow-sm); }
.shadow { box-shadow: var(--shadow-md); }
.shadow-lg { box-shadow: var(--shadow-lg); }
.shadow-xl { box-shadow: var(--shadow-xl); }

/* Opacity */
.opacity-0 { opacity: 0; }
.opacity-25 { opacity: 0.25; }
.opacity-50 { opacity: 0.5; }
.opacity-75 { opacity: 0.75; }
.opacity-100 { opacity: 1; }

/* Width */
.w-auto { width: auto; }
.w-full { width: 100%; }
.w-1/2 { width: 50%; }
.w-1/3 { width: 33.333333%; }
.w-2/3 { width: 66.666667%; }
.w-1/4 { width: 25%; }
.w-3/4 { width: 75%; }

/* Height */
.h-auto { height: auto; }
.h-full { height: 100%; }
.h-screen { height: 100vh; }

/* Position */
.relative { position: relative; }
.absolute { position: absolute; }
.fixed { position: fixed; }
.sticky { position: sticky; }

/* Overflow */
.overflow-hidden { overflow: hidden; }
.overflow-auto { overflow: auto; }
.overflow-scroll { overflow: scroll; }"""
    
    def _generate_animation_keyframes(self) -> str:
        """Generate animation keyframes"""
        return """/* Animation Keyframes */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideLeft {
    from {
        opacity: 0;
        transform: translateX(20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideRight {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes bounce {
    0%, 20%, 53%, 80%, 100% {
        animation-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000);
        transform: translate3d(0,0,0);
    }
    40%, 43% {
        animation-timing-function: cubic-bezier(0.755, 0.050, 0.855, 0.060);
        transform: translate3d(0, -30px, 0);
    }
    70% {
        animation-timing-function: cubic-bezier(0.755, 0.050, 0.855, 0.060);
        transform: translate3d(0, -15px, 0);
    }
    90% {
        transform: translate3d(0,-4px,0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: .5;
    }
}

@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 5px var(--color-primary);
    }
    50% {
        box-shadow: 0 0 20px var(--color-primary), 0 0 30px var(--color-primary);
    }
}

/* Animation utility classes */
.animate-fade-in { animation: fadeIn 0.5s ease-in-out; }
.animate-slide-up { animation: slideUp 0.3s ease-out; }
.animate-slide-down { animation: slideDown 0.3s ease-out; }
.animate-slide-left { animation: slideLeft 0.3s ease-out; }
.animate-slide-right { animation: slideRight 0.3s ease-out; }
.animate-bounce { animation: bounce 2s infinite; }
.animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
.animate-spin { animation: spin 1s linear infinite; }
.animate-glow { animation: glow 2s ease-in-out infinite; }

/* Hover animations */
.hover-lift:hover {
    transform: translateY(-4px);
    transition: transform 0.2s ease;
}

.hover-scale:hover {
    transform: scale(1.05);
    transition: transform 0.2s ease;
}

.hover-glow:hover {
    box-shadow: var(--shadow-glow);
    transition: box-shadow 0.3s ease;
}"""
    
    def generate_component_library(self, theme_name: Optional[str] = None) -> Dict[str, str]:
        """Generate complete component library styles"""
        theme = theme_name or self.current_theme
        
        components = {
            'buttons': self._generate_button_variants(theme),
            'cards': self._generate_card_variants(theme),
            'forms': self._generate_form_styles(theme),
            'navigation': self._generate_navigation_styles(theme),
            'metrics': self._generate_metric_components(theme),
            'alerts': self._generate_alert_variants(theme),
            'badges': self._generate_badge_styles(theme)
        }
        
        return components
    
    def _generate_button_variants(self, theme_name: str) -> str:
        """Generate button variant styles"""
        theme = self.brand_system.get_theme(theme_name)
        
        return f"""/* Button Variants */
.btn-gradient {{
    background: var(--gradient-primary);
    color: white;
    border: none;
    position: relative;
    overflow: hidden;
}}

.btn-gradient::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}}

.btn-gradient:hover::before {{
    left: 100%;
}}

.btn-neon {{
    background: transparent;
    border: 2px solid var(--color-secondary);
    color: var(--color-secondary);
    text-shadow: 0 0 10px var(--color-secondary);
    box-shadow: 0 0 10px var(--color-secondary);
}}

.btn-neon:hover {{
    background: var(--color-secondary);
    color: white;
    box-shadow: 0 0 20px var(--color-secondary);
}}

.btn-glass {{
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: var(--color-text-primary);
}}

.btn-3d {{
    background: var(--color-primary);
    color: white;
    border: none;
    box-shadow: 0 4px 0 var(--color-text-secondary);
    transform: translateY(0);
}}

.btn-3d:hover {{
    transform: translateY(2px);
    box-shadow: 0 2px 0 var(--color-text-secondary);
}}"""
    
    def _generate_card_variants(self, theme_name: str) -> str:
        """Generate card variant styles"""
        return """/* Card Variants */
.card-glass {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.card-neon {
    background: var(--color-background);
    border: 2px solid var(--color-primary);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
}

.card-gradient {
    background: var(--gradient-primary);
    color: white;
    border: none;
}

.card-gradient .card-title {
    color: white;
}

.card-floating {
    background: var(--color-background);
    border: none;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    transform: translateY(0);
    transition: all 0.3s ease;
}

.card-floating:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
}"""
    
    def _generate_form_styles(self, theme_name: str) -> str:
        """Generate form styles"""
        return """/* Form Styles */
.form-group {
    margin-bottom: var(--spacing-md);
}

.form-label {
    display: block;
    font-weight: var(--font-weight-medium);
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-xs);
    font-size: var(--font-size-sm);
}

.form-input {
    width: 100%;
    padding: var(--spacing-sm) var(--spacing-md);
    border: 2px solid rgba(0, 0, 0, 0.1);
    border-radius: var(--radius-md);
    font-family: var(--font-primary);
    font-size: var(--font-size-sm);
    background: var(--color-background);
    transition: all 0.2s ease;
}

.form-input:focus {
    outline: none;
    border-color: var(--color-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.form-select {
    width: 100%;
    padding: var(--spacing-sm) var(--spacing-md);
    border: 2px solid rgba(0, 0, 0, 0.1);
    border-radius: var(--radius-md);
    font-family: var(--font-primary);
    font-size: var(--font-size-sm);
    background: var(--color-background);
    cursor: pointer;
}

.form-checkbox {
    margin-right: var(--spacing-xs);
}

.form-help {
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
    margin-top: var(--spacing-xs);
}

.form-error {
    font-size: var(--font-size-xs);
    color: var(--color-error);
    margin-top: var(--spacing-xs);
}"""
    
    def _generate_navigation_styles(self, theme_name: str) -> str:
        """Generate navigation styles"""
        return """/* Navigation Styles */
.navbar {
    background: var(--color-background);
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    padding: var(--spacing-md) 0;
}

.navbar-brand {
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-bold);
    color: var(--color-primary);
}

.navbar-nav {
    display: flex;
    list-style: none;
    gap: var(--spacing-lg);
}

.nav-item {
    position: relative;
}

.nav-link {
    color: var(--color-text-secondary);
    font-weight: var(--font-weight-medium);
    text-decoration: none;
    padding: var(--spacing-sm) 0;
    transition: color 0.2s ease;
    position: relative;
}

.nav-link::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 0;
    height: 2px;
    background: var(--color-primary);
    transition: width 0.3s ease;
}

.nav-link:hover {
    color: var(--color-primary);
}

.nav-link:hover::after,
.nav-link.active::after {
    width: 100%;
}

.nav-link.active {
    color: var(--color-primary);
}"""
    
    def _generate_metric_components(self, theme_name: str) -> str:
        """Generate metric component styles"""
        return """/* Metric Components */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.metric-card-animated {
    background: var(--color-background);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card-animated::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: var(--gradient-primary);
    opacity: 0;
    transform: rotate(45deg);
    transition: opacity 0.3s ease;
    z-index: 0;
}

.metric-card-animated:hover::before {
    opacity: 0.05;
}

.metric-card-animated > * {
    position: relative;
    z-index: 1;
}

.metric-icon {
    font-size: var(--font-size-2xl);
    margin-bottom: var(--spacing-sm);
    color: var(--color-primary);
}

.metric-trend {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-xs);
    margin-top: var(--spacing-sm);
}

.trend-arrow {
    font-size: var(--font-size-sm);
}

.trend-up {
    color: var(--color-success);
}

.trend-down {
    color: var(--color-error);
}

.trend-neutral {
    color: var(--color-warning);
}"""
    
    def _generate_alert_variants(self, theme_name: str) -> str:
        """Generate alert variant styles"""
        return """/* Alert Variants */
.alert-dismissible {
    position: relative;
    padding-right: calc(var(--spacing-xl) + var(--spacing-lg));
}

.alert-close {
    position: absolute;
    right: var(--spacing-md);
    top: var(--spacing-md);
    background: none;
    border: none;
    font-size: var(--font-size-lg);
    cursor: pointer;
    opacity: 0.5;
    transition: opacity 0.2s ease;
}

.alert-close:hover {
    opacity: 1;
}

.alert-icon {
    display: flex;
    align-items: flex-start;
    gap: var(--spacing-sm);
}

.alert-icon::before {
    content: '';
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

.alert-info.alert-icon::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%233b82f6'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' /%3E%3C/svg%3E");
}"""
    
    def _generate_badge_styles(self, theme_name: str) -> str:
        """Generate badge styles"""
        return """/* Badge Styles */
.badge {
    display: inline-flex;
    align-items: center;
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--radius-full);
    font-size: var(--font-size-xs);
    font-weight: var(--font-weight-semibold);
    text-transform: uppercase;
    letter-spacing: 0.025em;
}

.badge-primary {
    background: var(--color-primary);
    color: white;
}

.badge-secondary {
    background: var(--color-secondary);
    color: white;
}

.badge-success {
    background: var(--color-success);
    color: white;
}

.badge-warning {
    background: var(--color-warning);
    color: white;
}

.badge-error {
    background: var(--color-error);
    color: white;
}

.badge-outline {
    background: transparent;
    border: 1px solid currentColor;
}

.badge-dot {
    position: relative;
    padding-left: calc(var(--spacing-sm) + 8px);
}

.badge-dot::before {
    content: '';
    position: absolute;
    left: var(--spacing-xs);
    top: 50%;
    transform: translateY(-50%);
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}"""
    
    def export_styles(self, output_dir: str, theme_name: Optional[str] = None):
        """Export generated styles to files"""
        theme = theme_name or self.current_theme
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate complete CSS
        complete_css = self.generate_complete_css(theme)
        
        # Write main stylesheet
        with open(os.path.join(output_dir, f'{theme}.css'), 'w', encoding='utf-8') as f:
            f.write(complete_css)
        
        # Write CSS variables only
        css_vars = self._generate_css_variables(theme)
        with open(os.path.join(output_dir, f'{theme}-variables.css'), 'w', encoding='utf-8') as f:
            f.write(css_vars)
        
        # Write SCSS variables
        scss_vars = self.brand_system.generate_scss_variables(theme)
        with open(os.path.join(output_dir, f'{theme}-variables.scss'), 'w', encoding='utf-8') as f:
            f.write(scss_vars)
        
        # Write component library
        components = self.generate_component_library(theme)
        for component_name, component_styles in components.items():
            with open(os.path.join(output_dir, f'{theme}-{component_name}.css'), 'w', encoding='utf-8') as f:
                f.write(component_styles)
        
        # Write theme configuration as JSON
        theme_config = self.brand_system.get_theme(theme)
        with open(os.path.join(output_dir, f'{theme}-config.json'), 'w', encoding='utf-8') as f:
            json.dump(theme_config, f, indent=2, ensure_ascii=False)

# Convenience functions
def generate_theme_css(theme_name: str = 'kpop_vibrant') -> str:
    """Generate complete CSS for a theme"""
    generator = KPOPStyleGenerator()
    return generator.generate_complete_css(theme_name)

def export_all_themes(output_dir: str):
    """Export all available themes"""
    generator = KPOPStyleGenerator()
    brand_system = KPOPBrandSystem()
    
    for theme_name in brand_system.themes.keys():
        theme_dir = os.path.join(output_dir, theme_name)
        generator.export_styles(theme_dir, theme_name)