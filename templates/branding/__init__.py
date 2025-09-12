# -*- coding: utf-8 -*-
"""
K-POP Branding System Package
Comprehensive branding and styling system for K-POP Analytics Dashboard
"""

from .brand_system import (
    KPOPBrandSystem, 
    BrandTier, 
    PlatformType, 
    ColorScheme, 
    Typography, 
    BrandAssets,
    get_brand_system,
    apply_tier_styling,
    get_platform_badge_style
)

from .style_generator import (
    KPOPStyleGenerator,
    generate_theme_css,
    export_all_themes
)

__all__ = [
    'KPOPBrandSystem',
    'BrandTier',
    'PlatformType', 
    'ColorScheme',
    'Typography',
    'BrandAssets',
    'KPOPStyleGenerator',
    'get_brand_system',
    'apply_tier_styling',
    'get_platform_badge_style',
    'generate_theme_css',
    'export_all_themes'
]