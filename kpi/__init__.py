"""
K-POP Dashboard KPI Package
===========================

This package provides a comprehensive KPI (Key Performance Indicator) calculation engine
for K-POP entertainment analytics. It supports flexible KPI definitions, custom formulas,
platform metrics aggregation, and weighted scoring systems.

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

from .kpi_engine import (
    KPIEngine,
    KPIDefinition,
    KPIResult,
    KPICategory,
    AggregationMethod,
    WeightingScheme,
    PlatformWeight,
    KPICalculationError,
    ValidationError
)

from .mobile_optimizer import (
    MobileOptimizer,
    MobileOptimizationConfig,
    MobileDeviceType,
    NotificationType,
    CacheLevel,
    PushNotificationPayload,
    MobileOptimizationResult,
    create_mobile_optimizer,
    generate_cache_key
)

__version__ = "1.0.0"
__author__ = "Backend Development Team"

__all__ = [
    # KPI Engine classes
    "KPIEngine",
    "KPIDefinition", 
    "KPIResult",
    "KPICategory",
    "AggregationMethod",
    "WeightingScheme",
    "PlatformWeight",
    "KPICalculationError",
    "ValidationError",
    
    # Mobile Optimizer classes
    "MobileOptimizer",
    "MobileOptimizationConfig",
    "MobileDeviceType",
    "NotificationType",
    "CacheLevel",
    "PushNotificationPayload",
    "MobileOptimizationResult",
    "create_mobile_optimizer",
    "generate_cache_key"
]