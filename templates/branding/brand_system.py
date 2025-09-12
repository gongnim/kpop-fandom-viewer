# -*- coding: utf-8 -*-
"""
K-POP Brand System for Analytics Dashboard
Comprehensive branding and styling system with theme management
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import colorsys
import json
import os

class BrandTier(Enum):
    """Brand tier classifications"""
    LEGENDARY = "legendary"
    TOP_TIER = "top_tier"
    RISING_STAR = "rising_star"
    ROOKIE = "rookie"
    TRAINEE = "trainee"

class PlatformType(Enum):
    """Social media platform types"""
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    WEIBO = "weibo"
    TWITTER = "twitter"

@dataclass
class ColorScheme:
    """Color scheme data structure"""
    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    text_primary: str
    text_secondary: str
    success: str
    warning: str
    error: str
    info: str

@dataclass
class Typography:
    """Typography configuration"""
    primary_font: str
    secondary_font: str
    korean_font: str
    english_font: str
    heading_weights: Dict[str, int]
    body_weights: Dict[str, int]
    font_sizes: Dict[str, str]
    line_heights: Dict[str, float]

@dataclass
class BrandAssets:
    """Brand assets and visual elements"""
    logo_primary: str
    logo_secondary: str
    logo_mark: str
    icons: Dict[str, str]
    patterns: Dict[str, str]
    gradients: Dict[str, str]

class KPOPBrandSystem:
    """Comprehensive K-POP branding system"""
    
    def __init__(self):
        self.color_schemes = self._initialize_color_schemes()
        self.typography = self._initialize_typography()
        self.brand_assets = self._initialize_brand_assets()
        self.tier_colors = self._initialize_tier_colors()
        self.platform_colors = self._initialize_platform_colors()
        self.themes = self._initialize_themes()
    
    def _initialize_color_schemes(self) -> Dict[str, ColorScheme]:
        """Initialize different color schemes"""
        return {
            'kpop_vibrant': ColorScheme(
                primary='#6366f1',      # Indigo - Professional yet vibrant
                secondary='#ec4899',    # Pink - K-POP energy
                accent='#f59e0b',       # Amber - Attention-grabbing
                background='#ffffff',   # Pure white
                surface='#f9fafb',      # Light gray
                text_primary='#1f2937', # Dark gray
                text_secondary='#6b7280', # Medium gray
                success='#10b981',      # Green
                warning='#f59e0b',      # Amber
                error='#ef4444',        # Red
                info='#3b82f6'          # Blue
            ),
            'kpop_neon': ColorScheme(
                primary='#ff0080',      # Hot pink
                secondary='#00ff80',    # Neon green
                accent='#8000ff',       # Purple
                background='#0a0a0a',   # Near black
                surface='#1a1a1a',      # Dark gray
                text_primary='#ffffff', # White
                text_secondary='#a0a0a0', # Light gray
                success='#00ff40',      # Bright green
                warning='#ffff00',      # Yellow
                error='#ff4040',        # Bright red
                info='#40a0ff'          # Bright blue
            ),
            'kpop_pastel': ColorScheme(
                primary='#a78bfa',      # Pastel purple
                secondary='#fbbf24',    # Pastel yellow
                accent='#34d399',       # Pastel green
                background='#fefefe',   # Off-white
                surface='#f0f8ff',      # Alice blue
                text_primary='#374151', # Dark blue-gray
                text_secondary='#9ca3af', # Light gray
                success='#6ee7b7',      # Light green
                warning='#fcd34d',      # Light yellow
                error='#fca5a5',        # Light red
                info='#93c5fd'          # Light blue
            ),
            'kpop_luxury': ColorScheme(
                primary='#1e1b4b',      # Deep indigo
                secondary='#fbbf24',    # Gold
                accent='#dc2626',       # Rich red
                background='#ffffff',   # Pure white
                surface='#f8fafc',      # Slate white
                text_primary='#0f172a', # Slate black
                text_secondary='#475569', # Slate gray
                success='#059669',      # Emerald
                warning='#d97706',      # Orange
                error='#dc2626',        # Red
                info='#0369a1'          # Blue
            )
        }
    
    def _initialize_typography(self) -> Typography:
        """Initialize typography system"""
        return Typography(
            primary_font='Inter',
            secondary_font='Noto Sans KR',
            korean_font='Noto Sans KR',
            english_font='Inter',
            heading_weights={
                'light': 300,
                'regular': 400,
                'medium': 500,
                'semibold': 600,
                'bold': 700,
                'extrabold': 800
            },
            body_weights={
                'light': 300,
                'regular': 400,
                'medium': 500,
                'semibold': 600
            },
            font_sizes={
                'xs': '0.75rem',    # 12px
                'sm': '0.875rem',   # 14px
                'base': '1rem',     # 16px
                'lg': '1.125rem',   # 18px
                'xl': '1.25rem',    # 20px
                '2xl': '1.5rem',    # 24px
                '3xl': '1.875rem',  # 30px
                '4xl': '2.25rem',   # 36px
                '5xl': '3rem',      # 48px
                '6xl': '3.75rem'    # 60px
            },
            line_heights={
                'tight': 1.25,
                'snug': 1.375,
                'normal': 1.5,
                'relaxed': 1.625,
                'loose': 2.0
            }
        )
    
    def _initialize_brand_assets(self) -> BrandAssets:
        """Initialize brand assets"""
        return BrandAssets(
            logo_primary="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjgwIj48dGV4dCB4PSIxMCIgeT0iNDAiIGZvbnQtZmFtaWx5PSJJbnRlciIgZm9udC1zaXplPSIyNCIgZm9udC13ZWlnaHQ9ImJvbGQiIGZpbGw9IiM2MzY2ZjEiPkstUE9QIEFuYWx5dGljczwvdGV4dD48L3N2Zz4=",
            logo_secondary="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjQwIj48dGV4dCB4PSI1IiB5PSIyNSIgZm9udC1mYW1pbHk9IkludGVyIiBmb250LXNpemU9IjE2IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjZWM0ODk5Ij5LLVBPUDwvdGV4dD48L3N2Zz4=",
            logo_mark="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjE4IiBmaWxsPSIjNjM2NmYxIi8+PHRleHQgeD0iMjAiIHk9IjI2IiBmb250LWZhbWlseT0iSW50ZXIiIGZvbnQtc2l6ZT0iMTQiIGZvbnQtd2VpZ2h0PSJib2xkIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+SzwvdGV4dD48L3N2Zz4=",
            icons={
                'music_note': '🎵',
                'star': '⭐',
                'trending_up': '📈',
                'fire': '🔥',
                'crown': '👑',
                'heart': '💖',
                'diamond': '💎',
                'microphone': '🎤',
                'headphones': '🎧',
                'trophy': '🏆'
            },
            patterns={
                'dots': 'radial-gradient(circle, rgba(99,102,241,0.1) 1px, transparent 1px)',
                'diagonal': 'repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(236,72,153,0.05) 10px, rgba(236,72,153,0.05) 20px)',
                'waves': 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%236366f1" fill-opacity="0.1"%3E%3Cpath d="M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")'
            },
            gradients={
                'primary': 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
                'secondary': 'linear-gradient(135deg, #f59e0b 0%, #ec4899 100%)',
                'success': 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
                'neon': 'linear-gradient(135deg, #ff0080 0%, #8000ff 100%)',
                'sunset': 'linear-gradient(135deg, #ff6b6b 0%, #ffa726 50%, #ffeb3b 100%)',
                'ocean': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'forest': 'linear-gradient(135deg, #134e5e 0%, #71b280 100%)'
            }
        )
    
    def _initialize_tier_colors(self) -> Dict[BrandTier, Dict[str, str]]:
        """Initialize colors for different artist tiers"""
        return {
            BrandTier.LEGENDARY: {
                'primary': '#ffd700',   # Gold
                'secondary': '#ff6b35', # Orange-red
                'accent': '#ffffff',    # White
                'glow': '#ffd70080'     # Gold with transparency
            },
            BrandTier.TOP_TIER: {
                'primary': '#c084fc',   # Purple
                'secondary': '#ec4899', # Pink
                'accent': '#ffffff',    # White
                'glow': '#c084fc80'     # Purple with transparency
            },
            BrandTier.RISING_STAR: {
                'primary': '#10b981',   # Green
                'secondary': '#3b82f6', # Blue
                'accent': '#ffffff',    # White
                'glow': '#10b98180'     # Green with transparency
            },
            BrandTier.ROOKIE: {
                'primary': '#f59e0b',   # Amber
                'secondary': '#6366f1', # Indigo
                'accent': '#ffffff',    # White
                'glow': '#f59e0b80'     # Amber with transparency
            },
            BrandTier.TRAINEE: {
                'primary': '#6b7280',   # Gray
                'secondary': '#9ca3af', # Light gray
                'accent': '#ffffff',    # White
                'glow': '#6b728080'     # Gray with transparency
            }
        }
    
    def _initialize_platform_colors(self) -> Dict[PlatformType, Dict[str, str]]:
        """Initialize colors for different platforms"""
        return {
            PlatformType.YOUTUBE: {
                'primary': '#ff0000',
                'secondary': '#282828',
                'text': '#ffffff'
            },
            PlatformType.INSTAGRAM: {
                'primary': '#e4405f',
                'secondary': '#833ab4',
                'text': '#ffffff'
            },
            PlatformType.TIKTOK: {
                'primary': '#000000',
                'secondary': '#ff0050',
                'text': '#ffffff'
            },
            PlatformType.SPOTIFY: {
                'primary': '#1db954',
                'secondary': '#191414',
                'text': '#ffffff'
            },
            PlatformType.APPLE_MUSIC: {
                'primary': '#fa233b',
                'secondary': '#000000',
                'text': '#ffffff'
            },
            PlatformType.WEIBO: {
                'primary': '#e6162d',
                'secondary': '#ffd700',
                'text': '#ffffff'
            },
            PlatformType.TWITTER: {
                'primary': '#1da1f2',
                'secondary': '#14171a',
                'text': '#ffffff'
            }
        }
    
    def _initialize_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize complete themes combining all elements"""
        themes = {}
        
        for scheme_name, color_scheme in self.color_schemes.items():
            themes[scheme_name] = {
                'colors': color_scheme.__dict__,
                'typography': self.typography.__dict__,
                'assets': self.brand_assets.__dict__,
                'spacing': {
                    'xs': '0.25rem',    # 4px
                    'sm': '0.5rem',     # 8px
                    'md': '1rem',       # 16px
                    'lg': '1.5rem',     # 24px
                    'xl': '2rem',       # 32px
                    '2xl': '3rem',      # 48px
                    '3xl': '4rem'       # 64px
                },
                'border_radius': {
                    'none': '0',
                    'sm': '0.25rem',    # 4px
                    'md': '0.5rem',     # 8px
                    'lg': '0.75rem',    # 12px
                    'xl': '1rem',       # 16px
                    '2xl': '1.5rem',    # 24px
                    'full': '9999px'
                },
                'shadows': {
                    'sm': '0 1px 2px rgba(0, 0, 0, 0.05)',
                    'md': '0 4px 6px rgba(0, 0, 0, 0.1)',
                    'lg': '0 10px 15px rgba(0, 0, 0, 0.1)',
                    'xl': '0 20px 25px rgba(0, 0, 0, 0.1)',
                    'glow': '0 0 20px rgba(99, 102, 241, 0.5)'
                },
                'animations': {
                    'bounce': 'bounce 2s infinite',
                    'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    'spin': 'spin 1s linear infinite',
                    'fade_in': 'fadeIn 0.5s ease-in-out',
                    'slide_up': 'slideUp 0.3s ease-out'
                }
            }
        
        return themes
    
    def get_theme(self, theme_name: str = 'kpop_vibrant') -> Dict[str, Any]:
        """Get complete theme configuration"""
        return self.themes.get(theme_name, self.themes['kpop_vibrant'])
    
    def get_tier_styling(self, tier: BrandTier) -> Dict[str, str]:
        """Get styling for specific artist tier"""
        return self.tier_colors.get(tier, self.tier_colors[BrandTier.ROOKIE])
    
    def get_platform_styling(self, platform: PlatformType) -> Dict[str, str]:
        """Get styling for specific platform"""
        return self.platform_colors.get(platform, {
            'primary': '#6b7280',
            'secondary': '#9ca3af',
            'text': '#ffffff'
        })
    
    def generate_css_variables(self, theme_name: str = 'kpop_vibrant') -> str:
        """Generate CSS custom properties for theme"""
        theme = self.get_theme(theme_name)
        css_vars = []
        
        # Colors
        for key, value in theme['colors'].items():
            css_vars.append(f"  --{key.replace('_', '-')}: {value};")
        
        # Typography
        typography = theme['typography']
        for key, value in typography['font_sizes'].items():
            css_vars.append(f"  --font-size-{key}: {value};")
        
        # Spacing
        for key, value in theme['spacing'].items():
            css_vars.append(f"  --spacing-{key}: {value};")
        
        # Border radius
        for key, value in theme['border_radius'].items():
            css_vars.append(f"  --radius-{key}: {value};")
        
        # Shadows
        for key, value in theme['shadows'].items():
            css_vars.append(f"  --shadow-{key}: {value};")
        
        return ":root {\n" + "\n".join(css_vars) + "\n}"
    
    def generate_scss_variables(self, theme_name: str = 'kpop_vibrant') -> str:
        """Generate SCSS variables for theme"""
        theme = self.get_theme(theme_name)
        scss_vars = []
        
        # Colors
        for key, value in theme['colors'].items():
            scss_vars.append(f"${key.replace('_', '-')}: {value};")
        
        # Typography
        typography = theme['typography']
        scss_vars.append(f"$font-primary: '{typography['primary_font']}', sans-serif;")
        scss_vars.append(f"$font-secondary: '{typography['secondary_font']}', sans-serif;")
        scss_vars.append(f"$font-korean: '{typography['korean_font']}', sans-serif;")
        
        return "\n".join(scss_vars)
    
    def create_component_styles(self, component_type: str, theme_name: str = 'kpop_vibrant') -> Dict[str, str]:
        """Create component-specific styles"""
        theme = self.get_theme(theme_name)
        
        component_styles = {
            'button': {
                'primary': f"""
                    background: {theme['colors']['primary']};
                    color: white;
                    border: none;
                    border-radius: {theme['border_radius']['md']};
                    padding: {theme['spacing']['sm']} {theme['spacing']['md']};
                    font-family: '{theme['typography']['primary_font']}', sans-serif;
                    font-weight: {theme['typography']['body_weights']['semibold']};
                    box-shadow: {theme['shadows']['md']};
                    transition: all 0.2s ease;
                """,
                'secondary': f"""
                    background: {theme['colors']['secondary']};
                    color: white;
                    border: 2px solid {theme['colors']['secondary']};
                    border-radius: {theme['border_radius']['md']};
                    padding: {theme['spacing']['sm']} {theme['spacing']['md']};
                    font-family: '{theme['typography']['primary_font']}', sans-serif;
                    font-weight: {theme['typography']['body_weights']['medium']};
                """
            },
            'card': {
                'default': f"""
                    background: {theme['colors']['background']};
                    border: 1px solid rgba(0, 0, 0, 0.1);
                    border-radius: {theme['border_radius']['lg']};
                    padding: {theme['spacing']['lg']};
                    box-shadow: {theme['shadows']['md']};
                """,
                'elevated': f"""
                    background: {theme['colors']['surface']};
                    border: none;
                    border-radius: {theme['border_radius']['xl']};
                    padding: {theme['spacing']['xl']};
                    box-shadow: {theme['shadows']['lg']};
                """
            },
            'metric_card': {
                'positive': f"""
                    background: linear-gradient(135deg, {theme['colors']['success']}15, {theme['colors']['success']}05);
                    border-left: 4px solid {theme['colors']['success']};
                    border-radius: {theme['border_radius']['lg']};
                    padding: {theme['spacing']['lg']};
                """,
                'negative': f"""
                    background: linear-gradient(135deg, {theme['colors']['error']}15, {theme['colors']['error']}05);
                    border-left: 4px solid {theme['colors']['error']};
                    border-radius: {theme['border_radius']['lg']};
                    padding: {theme['spacing']['lg']};
                """,
                'neutral': f"""
                    background: linear-gradient(135deg, {theme['colors']['info']}15, {theme['colors']['info']}05);
                    border-left: 4px solid {theme['colors']['info']};
                    border-radius: {theme['border_radius']['lg']};
                    padding: {theme['spacing']['lg']};
                """
            }
        }
        
        return component_styles.get(component_type, {})
    
    def create_responsive_breakpoints(self) -> Dict[str, str]:
        """Create responsive breakpoints"""
        return {
            'mobile': '(max-width: 768px)',
            'tablet': '(min-width: 769px) and (max-width: 1024px)',
            'desktop': '(min-width: 1025px)',
            'large': '(min-width: 1440px)',
            'xlarge': '(min-width: 1920px)'
        }
    
    def generate_brand_guidelines(self, theme_name: str = 'kpop_vibrant') -> Dict[str, Any]:
        """Generate comprehensive brand guidelines"""
        theme = self.get_theme(theme_name)
        
        return {
            'brand_identity': {
                'mission': 'K-POP 산업의 데이터 중심 인사이트 제공',
                'vision': '글로벌 K-POP 생태계의 디지털 트랜스포메이션 선도',
                'values': ['혁신', '정확성', '투명성', '팬 중심']
            },
            'visual_identity': {
                'logo_usage': {
                    'primary': '메인 로고 - 대부분의 상황에서 사용',
                    'secondary': '공간이 제한된 경우 사용',
                    'mark': '아이콘으로 사용, 브랜드 인지도가 높을 때'
                },
                'color_palette': theme['colors'],
                'typography': {
                    'primary_use': '헤딩, 중요한 텍스트',
                    'secondary_use': '본문, 한글 텍스트',
                    'pairing': '영문과 한글의 조화로운 조합'
                }
            },
            'design_principles': {
                'clarity': '명확하고 이해하기 쉬운 정보 전달',
                'vibrancy': 'K-POP의 활기찬 에너지 반영',
                'professionalism': '비즈니스 환경에 적합한 신뢰감',
                'accessibility': '모든 사용자를 위한 접근성 고려'
            },
            'application_guidelines': {
                'charts': '브랜드 색상을 활용한 직관적 데이터 시각화',
                'reports': '전문적이면서도 매력적인 리포트 디자인',
                'dashboard': '정보 밀도와 가독성의 균형',
                'mobile': '반응형 디자인으로 모든 기기에서 일관성'
            }
        }

# Utility functions
def get_brand_system() -> KPOPBrandSystem:
    """Get singleton brand system instance"""
    return KPOPBrandSystem()

def apply_tier_styling(tier: BrandTier, base_style: str) -> str:
    """Apply tier-specific styling to base style"""
    brand_system = get_brand_system()
    tier_colors = brand_system.get_tier_styling(tier)
    
    # Add tier-specific enhancements
    enhanced_style = base_style
    enhanced_style += f" border-left: 4px solid {tier_colors['primary']};"
    enhanced_style += f" box-shadow: 0 0 10px {tier_colors['glow']};"
    
    return enhanced_style

def get_platform_badge_style(platform: PlatformType) -> str:
    """Get platform-specific badge styling"""
    brand_system = get_brand_system()
    platform_colors = brand_system.get_platform_styling(platform)
    
    return f"""
        background: {platform_colors['primary']};
        color: {platform_colors['text']};
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    """