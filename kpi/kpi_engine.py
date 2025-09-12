import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
KPI Engine Module for K-POP Dashboard
=====================================

A comprehensive KPI calculation engine that supports:
- Custom KPI formulas and calculations
- Platform metrics aggregation (YouTube, Spotify, Twitter, etc.)
- Weighted scoring systems
- Extensible KPI types
- Integration with PostgreSQL database schema
- Real-time and batch calculations

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import statistics
import operator
from collections import defaultdict
import psycopg2.extras

from database_postgresql import (
    get_db_connection, get_kpi_definitions, get_kpi_definition_by_name,
    add_kpi_calculation, bulk_insert_kpi_calculations, update_kpi_calculation_rankings,
    get_latest_kpi_calculations
)
from ..config import Config

# Configure module logger
logger = logging.getLogger(__name__)

# ========================================
# Enums and Type Definitions
# ========================================

class KPICategory(Enum):
    """Categories for KPI classification."""
    ENGAGEMENT = "engagement"
    GROWTH = "growth"
    REACH = "reach"
    INFLUENCE = "influence"
    POPULARITY = "popularity"
    CONSISTENCY = "consistency"
    MOMENTUM = "momentum"
    COMPOSITE = "composite"
    CUSTOM = "custom"

class AggregationMethod(Enum):
    """Methods for aggregating platform metrics."""
    SUM = "sum"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    PERCENTILE_90 = "percentile_90"
    GROWTH_RATE = "growth_rate"
    NORMALIZED_SUM = "normalized_sum"

class WeightingScheme(Enum):
    """Different weighting schemes for KPI calculations."""
    EQUAL = "equal"
    PLATFORM_BASED = "platform_based"
    AUDIENCE_BASED = "audience_based"
    ENGAGEMENT_BASED = "engagement_based"
    CUSTOM = "custom"

# ========================================
# Data Classes
# ========================================

@dataclass
class PlatformWeight:
    """Platform-specific weight configuration."""
    platform: str
    weight: float = 1.0
    metric_weights: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Platform weight must be non-negative")
        if not self.metric_weights:
            # Default metric weights for common platforms
            defaults = {
                'youtube': {'subscribers': 0.4, 'total_views': 0.6},
                'spotify': {'monthly_listeners': 0.7, 'popularity': 0.3},
                'twitter': {'followers': 0.8, 'likes': 0.2},
                'instagram': {'followers': 0.6, 'likes': 0.4},
                'tiktok': {'followers': 0.5, 'likes': 0.5}
            }
            self.metric_weights = defaults.get(self.platform.lower(), {'default': 1.0})

@dataclass
class KPIDefinition:
    """Definition of a KPI with calculation parameters."""
    name: str
    category: KPICategory
    description: str
    formula: Union[str, Callable] = None
    platforms: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    weighting_scheme: WeightingScheme = WeightingScheme.PLATFORM_BASED
    platform_weights: Dict[str, PlatformWeight] = field(default_factory=dict)
    time_window_days: int = 30
    normalization_factor: Optional[float] = None
    min_data_points: int = 1
    is_active: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("KPI name cannot be empty")
        if self.time_window_days <= 0:
            raise ValueError("Time window must be positive")
        
        # Initialize default platform weights if not provided
        if not self.platform_weights and self.platforms:
            for platform in self.platforms:
                self.platform_weights[platform] = PlatformWeight(platform=platform)

@dataclass
class KPIResult:
    """Result of a KPI calculation."""
    kpi_name: str
    artist_id: int
    artist_name: str
    value: float
    normalized_value: Optional[float] = None
    rank: Optional[int] = None
    percentile: Optional[float] = None
    calculation_date: datetime = field(default_factory=datetime.now)
    platform_contributions: Dict[str, float] = field(default_factory=dict)
    metric_contributions: Dict[str, float] = field(default_factory=dict)
    data_quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ========================================
# Custom Exceptions
# ========================================

class KPICalculationError(Exception):
    """Raised when KPI calculation fails."""
    pass

class ValidationError(Exception):
    """Raised when KPI validation fails."""
    pass

class FormulaParsingError(Exception):
    """Raised when formula parsing fails."""
    pass

class SecurityError(Exception):
    """Raised when security validation fails."""
    pass

# ========================================
# Main KPI Engine Class
# ========================================

class KPIEngine:
    """
    Main KPI calculation engine for K-POP dashboard.
    
    Features:
    - Custom KPI definitions with flexible formulas
    - Multi-platform metric aggregation
    - Weighted scoring systems
    - Real-time and batch calculations
    - Data quality validation
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize KPI engine with configuration."""
        self.config = config or {}
        self.kpi_definitions: Dict[str, KPIDefinition] = {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl_minutes = self.config.get('cache_ttl_minutes', 15)
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self.database_enabled = self.config.get('database_enabled', True)
        
        # Initialize pattern for metric references (platform.metric)
        self._metric_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*')
        
        # Initialize safe functions for expression evaluation
        self._allowed_functions = {
            'abs': abs,
            'max': max,
            'min': min,
            'round': round,
            'sum': sum,
            'len': len,
            'pow': pow,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'ceil': math.ceil,
            'floor': math.floor,
            'mean': statistics.mean,
            'median': statistics.median,
            'stdev': statistics.stdev,
            'variance': statistics.variance
        }
        
        # Initialize safe operators
        self._safe_operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '<=': operator.le,
            '>': operator.gt,
            '>=': operator.ge,
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_
        }
        
        # Load KPI definitions from database or use defaults
        if self.database_enabled:
            self._load_kpis_from_database()
        else:
            self._load_default_kpis()
        
        logger.info("KPI Engine initialized successfully")
    
    def _load_kpis_from_database(self):
        """Load KPI definitions from database."""
        try:
            db_definitions = get_kpi_definitions(active_only=True)
            
            for db_def in db_definitions:
                # Convert database definition to KPIDefinition object
                kpi_def = self._convert_db_to_kpi_definition(db_def)
                if kpi_def:
                    self.kpi_definitions[kpi_def.name] = kpi_def
            
            logger.info(f"Loaded {len(self.kpi_definitions)} KPI definitions from database")
            
            # If no definitions found in database, load defaults as fallback
            if not self.kpi_definitions:
                logger.warning("No KPI definitions found in database, loading defaults")
                self._load_default_kpis()
                
        except Exception as e:
            logger.error(f"Failed to load KPI definitions from database: {e}")
            logger.info("Falling back to default KPI definitions")
            self._load_default_kpis()
    
    def _convert_db_to_kpi_definition(self, db_def: Dict[str, Any]) -> Optional[KPIDefinition]:
        """Convert database KPI definition to KPIDefinition object."""
        try:
            # Parse platform weights
            platform_weights = {}
            if db_def.get('platform_weights'):
                weights_data = db_def['platform_weights']
                for platform, weight_info in weights_data.items():
                    if isinstance(weight_info, dict):
                        platform_weights[platform] = PlatformWeight(
                            platform=platform,
                            weight=weight_info.get('weight', 1.0),
                            metric_weights=weight_info.get('metrics', {})
                        )
                    else:
                        # Simple weight value
                        platform_weights[platform] = PlatformWeight(platform=platform, weight=weight_info)
            
            # Parse formula to extract platforms and metrics
            formula = db_def.get('kpi_formula', {})
            platforms = formula.get('platforms', [])
            metrics = formula.get('metrics', [])
            
            # Map aggregation method from string to enum
            aggregation_method = AggregationMethod.WEIGHTED_AVERAGE
            try:
                aggregation_method = AggregationMethod(db_def.get('aggregation_method', 'weighted_average'))
            except ValueError:
                logger.warning(f"Unknown aggregation method: {db_def.get('aggregation_method')}")
            
            # Map weighting scheme from string to enum
            weighting_scheme = WeightingScheme.PLATFORM_BASED
            try:
                weighting_scheme = WeightingScheme(db_def.get('weighting_scheme', 'platform_based'))
            except ValueError:
                logger.warning(f"Unknown weighting scheme: {db_def.get('weighting_scheme')}")
            
            # Map category from string to enum
            category = KPICategory.CUSTOM
            try:
                category = KPICategory(db_def.get('kpi_category', 'custom'))
            except ValueError:
                logger.warning(f"Unknown KPI category: {db_def.get('kpi_category')}")
            
            return KPIDefinition(
                name=db_def['kpi_name'],
                category=category,
                description=db_def.get('kpi_description', ''),
                platforms=platforms,
                metrics=metrics,
                aggregation_method=aggregation_method,
                weighting_scheme=weighting_scheme,
                platform_weights=platform_weights,
                time_window_days=db_def.get('time_window_days', 30),
                normalization_factor=db_def.get('normalization_factor'),
                min_data_points=db_def.get('min_data_points', 1),
                custom_params={
                    'target_value': db_def.get('target_value'),
                    'warning_threshold': db_def.get('warning_threshold'),
                    'critical_threshold': db_def.get('critical_threshold'),
                    'kpi_id': db_def.get('kpi_id')
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to convert database KPI definition: {e}")
            return None
    
    def _load_default_kpis(self):
        """Load default KPI definitions for K-POP analytics."""
        
        # Engagement Rate KPI
        self.add_kpi_definition(KPIDefinition(
            name="engagement_rate",
            category=KPICategory.ENGAGEMENT,
            description="Cross-platform engagement rate based on likes/followers ratio",
            platforms=['youtube', 'spotify', 'twitter', 'instagram'],
            metrics=['followers', 'subscribers', 'likes', 'monthly_listeners'],
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            weighting_scheme=WeightingScheme.PLATFORM_BASED,
            time_window_days=30,
            custom_params={'engagement_threshold': 0.05}
        ))
        
        # Growth Momentum KPI
        self.add_kpi_definition(KPIDefinition(
            name="growth_momentum",
            category=KPICategory.GROWTH,
            description="30-day growth momentum across all platforms",
            platforms=['youtube', 'spotify', 'twitter'],
            metrics=['subscribers', 'followers', 'monthly_listeners'],
            aggregation_method=AggregationMethod.GROWTH_RATE,
            weighting_scheme=WeightingScheme.AUDIENCE_BASED,
            time_window_days=30,
            normalization_factor=100.0
        ))
        
        # Platform Reach KPI
        self.add_kpi_definition(KPIDefinition(
            name="platform_reach",
            category=KPICategory.REACH,
            description="Total reach across all platforms",
            platforms=['youtube', 'spotify', 'twitter', 'instagram'],
            metrics=['subscribers', 'followers', 'monthly_listeners'],
            aggregation_method=AggregationMethod.NORMALIZED_SUM,
            weighting_scheme=WeightingScheme.PLATFORM_BASED,
            time_window_days=7
        ))
        
        # Consistency Score KPI
        self.add_kpi_definition(KPIDefinition(
            name="consistency_score",
            category=KPICategory.CONSISTENCY,
            description="Consistency of performance across platforms",
            platforms=['youtube', 'spotify', 'twitter'],
            metrics=['subscribers', 'followers', 'monthly_listeners'],
            aggregation_method=AggregationMethod.AVERAGE,
            weighting_scheme=WeightingScheme.EQUAL,
            time_window_days=90,
            custom_params={'consistency_window': 7}
        ))
        
        # Composite Influence Score
        self.add_kpi_definition(KPIDefinition(
            name="influence_score",
            category=KPICategory.COMPOSITE,
            description="Comprehensive influence score combining reach, engagement, and growth",
            platforms=['youtube', 'spotify', 'twitter', 'instagram'],
            metrics=['subscribers', 'followers', 'monthly_listeners', 'likes', 'total_views'],
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            weighting_scheme=WeightingScheme.CUSTOM,
            time_window_days=30,
            platform_weights={
                'youtube': PlatformWeight('youtube', 0.4, {'subscribers': 0.3, 'total_views': 0.7}),
                'spotify': PlatformWeight('spotify', 0.3, {'monthly_listeners': 1.0}),
                'twitter': PlatformWeight('twitter', 0.2, {'followers': 0.8, 'likes': 0.2}),
                'instagram': PlatformWeight('instagram', 0.1, {'followers': 0.6, 'likes': 0.4})
            }
        ))
    
    def add_kpi_definition(self, kpi_def: KPIDefinition) -> None:
        """Add or update a KPI definition."""
        try:
            # Validate KPI definition
            self._validate_kpi_definition(kpi_def)
            
            self.kpi_definitions[kpi_def.name] = kpi_def
            logger.info(f"Added KPI definition: {kpi_def.name}")
            
        except Exception as e:
            logger.error(f"Failed to add KPI definition {kpi_def.name}: {e}")
            raise ValidationError(f"Invalid KPI definition: {e}")
    
    def _validate_kpi_definition(self, kpi_def: KPIDefinition) -> None:
        """Validate KPI definition parameters."""
        if not kpi_def.name or not kpi_def.name.strip():
            raise ValidationError("KPI name cannot be empty")
        
        if not kpi_def.platforms:
            raise ValidationError("At least one platform must be specified")
        
        if not kpi_def.metrics:
            raise ValidationError("At least one metric must be specified")
        
        # Validate platform weights sum to reasonable value for weighted schemes
        if kpi_def.weighting_scheme in [WeightingScheme.PLATFORM_BASED, WeightingScheme.CUSTOM]:
            if kpi_def.platform_weights:
                total_weight = sum(pw.weight for pw in kpi_def.platform_weights.values())
                if total_weight <= 0:
                    raise ValidationError("Total platform weights must be positive")
    
    def calculate_kpi(self, kpi_name: str, artist_id: int, 
                     calculation_date: Optional[datetime] = None) -> KPIResult:
        """Calculate a specific KPI for an artist."""
        if kpi_name not in self.kpi_definitions:
            raise KPICalculationError(f"KPI definition not found: {kpi_name}")
        
        kpi_def = self.kpi_definitions[kpi_name]
        if not kpi_def.is_active:
            raise KPICalculationError(f"KPI is inactive: {kpi_name}")
        
        calculation_date = calculation_date or datetime.now()
        
        # Check cache
        cache_key = f"{kpi_name}_{artist_id}_{calculation_date.date()}"
        if self.cache_enabled and cache_key in self._cache:
            cache_time, result = self._cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=self.cache_ttl_minutes):
                logger.debug(f"Returning cached result for {cache_key}")
                return result
        
        try:
            # Get artist information
            artist_info = self._get_artist_info(artist_id)
            if not artist_info:
                raise KPICalculationError(f"Artist not found: {artist_id}")
            
            # Get platform metrics
            metrics_data = self._get_platform_metrics(
                artist_id, kpi_def.platforms, kpi_def.metrics, 
                calculation_date, kpi_def.time_window_days
            )
            
            # Validate minimum data requirements
            if len(metrics_data) < kpi_def.min_data_points:
                raise KPICalculationError(
                    f"Insufficient data points: {len(metrics_data)} < {kpi_def.min_data_points}"
                )
            
            # Calculate KPI value - use JSON formula if available
            if hasattr(kpi_def, 'formula') and isinstance(kpi_def.formula, dict):
                value, platform_contributions, metric_contributions = self._calculate_kpi_with_json_formula(
                    kpi_def, metrics_data
                )
            else:
                # Fall back to original calculation method
                value, platform_contributions, metric_contributions = self._calculate_kpi_value(
                    kpi_def, metrics_data
                )
            
            # Normalize if specified
            normalized_value = None
            if kpi_def.normalization_factor:
                normalized_value = value / kpi_def.normalization_factor
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(metrics_data, kpi_def)
            
            # Evaluate KPI status
            status, achievement_rate = self.evaluate_kpi_status(value, kpi_def.custom_params)
            
            # Create result
            result = KPIResult(
                kpi_name=kpi_name,
                artist_id=artist_id,
                artist_name=artist_info['name'],
                value=value,
                normalized_value=normalized_value,
                calculation_date=calculation_date,
                platform_contributions=platform_contributions,
                metric_contributions=metric_contributions,
                data_quality_score=data_quality_score,
                metadata={
                    'time_window_days': kpi_def.time_window_days,
                    'data_points_count': len(metrics_data),
                    'aggregation_method': kpi_def.aggregation_method.value,
                    'status': status,
                    'target_achievement_rate': achievement_rate
                }
            )
            
            # Store result in database if enabled
            if self.database_enabled:
                try:
                    kpi_id = kpi_def.custom_params.get('kpi_id')
                    if kpi_id:
                        add_kpi_calculation(
                            kpi_id=kpi_id,
                            entity_type='artist',
                            entity_id=artist_id,
                            entity_name=artist_info['name'],
                            calculated_value=value,
                            calculation_date=calculation_date,
                            normalized_value=normalized_value,
                            target_achievement_rate=achievement_rate,
                            status=status,
                            data_quality_score=data_quality_score,
                            platform_contributions=platform_contributions,
                            metric_contributions=metric_contributions
                        )
                except Exception as e:
                    logger.warning(f"Failed to store KPI calculation in database: {e}")
            
            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = (datetime.now(), result)
            
            logger.debug(f"Calculated KPI {kpi_name} for artist {artist_id}: {value}")
            return result
            
        except Exception as e:
            logger.error(f"KPI calculation failed for {kpi_name}, artist {artist_id}: {e}")
            raise KPICalculationError(f"Calculation failed: {e}")
    
    def calculate_batch_kpis(self, kpi_names: List[str], artist_ids: List[int],
                           calculation_date: Optional[datetime] = None) -> List[KPIResult]:
        """Calculate multiple KPIs for multiple artists efficiently."""
        results = []
        calculation_date = calculation_date or datetime.now()
        
        logger.info(f"Starting batch calculation for {len(kpi_names)} KPIs, {len(artist_ids)} artists")
        
        for kpi_name in kpi_names:
            for artist_id in artist_ids:
                try:
                    result = self.calculate_kpi(kpi_name, artist_id, calculation_date)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to calculate {kpi_name} for artist {artist_id}: {e}")
                    continue
        
        logger.info(f"Completed batch calculation: {len(results)} successful results")
        return results
    
    def calculate_rankings(self, kpi_name: str, artist_ids: Optional[List[int]] = None,
                          calculation_date: Optional[datetime] = None) -> List[KPIResult]:
        """Calculate KPI rankings for a group of artists."""
        calculation_date = calculation_date or datetime.now()
        
        # Get artist list if not provided
        if artist_ids is None:
            artist_ids = self._get_all_active_artist_ids()
        
        # Calculate KPIs for all artists
        results = []
        for artist_id in artist_ids:
            try:
                result = self.calculate_kpi(kpi_name, artist_id, calculation_date)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to calculate ranking for artist {artist_id}: {e}")
                continue
        
        # Sort by value (descending)
        results.sort(key=lambda x: x.value, reverse=True)
        
        # Assign ranks and percentiles
        total_count = len(results)
        for i, result in enumerate(results):
            result.rank = i + 1
            result.percentile = ((total_count - i) / total_count) * 100
        
        logger.info(f"Calculated rankings for {kpi_name}: {total_count} artists")
        
        # Update rankings in database if enabled
        if self.database_enabled and results:
            try:
                kpi_def = self.kpi_definitions[kpi_name]
                kpi_id = kpi_def.custom_params.get('kpi_id')
                if kpi_id:
                    update_kpi_calculation_rankings(kpi_id, calculation_date)
            except Exception as e:
                logger.warning(f"Failed to update rankings in database: {e}")
        
        return results
    
    def _calculate_kpi_value(self, kpi_def: KPIDefinition, metrics_data: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Calculate the actual KPI value based on aggregation method."""
        platform_contributions = defaultdict(float)
        metric_contributions = defaultdict(float)
        
        if kpi_def.aggregation_method == AggregationMethod.SUM:
            value = self._calculate_sum(metrics_data, kpi_def, platform_contributions, metric_contributions)
        elif kpi_def.aggregation_method == AggregationMethod.AVERAGE:
            value = self._calculate_average(metrics_data, kpi_def, platform_contributions, metric_contributions)
        elif kpi_def.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            value = self._calculate_weighted_average(metrics_data, kpi_def, platform_contributions, metric_contributions)
        elif kpi_def.aggregation_method == AggregationMethod.GROWTH_RATE:
            value = self._calculate_growth_rate(metrics_data, kpi_def, platform_contributions, metric_contributions)
        elif kpi_def.aggregation_method == AggregationMethod.NORMALIZED_SUM:
            value = self._calculate_normalized_sum(metrics_data, kpi_def, platform_contributions, metric_contributions)
        else:
            raise KPICalculationError(f"Unsupported aggregation method: {kpi_def.aggregation_method}")
        
        return value, dict(platform_contributions), dict(metric_contributions)
    
    def _calculate_weighted_average(self, metrics_data: List[Dict[str, Any]], 
                                   kpi_def: KPIDefinition,
                                   platform_contributions: Dict[str, float],
                                   metric_contributions: Dict[str, float]) -> float:
        """Calculate weighted average based on platform and metric weights."""
        total_weighted_value = 0.0
        total_weight = 0.0
        
        for data_point in metrics_data:
            platform = data_point['platform']
            metric_type = data_point['metric_type']
            value = float(data_point['value'])
            
            # Get platform weight
            platform_weight = 1.0
            if platform in kpi_def.platform_weights:
                platform_weight = kpi_def.platform_weights[platform].weight
                
                # Get metric weight for this platform
                metric_weights = kpi_def.platform_weights[platform].metric_weights
                metric_weight = metric_weights.get(metric_type, 1.0)
                
                final_weight = platform_weight * metric_weight
            else:
                final_weight = platform_weight
            
            weighted_value = value * final_weight
            total_weighted_value += weighted_value
            total_weight += final_weight
            
            # Track contributions
            platform_contributions[platform] += weighted_value
            metric_contributions[metric_type] += weighted_value
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_value / total_weight
    
    def _calculate_growth_rate(self, metrics_data: List[Dict[str, Any]], 
                              kpi_def: KPIDefinition,
                              platform_contributions: Dict[str, float],
                              metric_contributions: Dict[str, float]) -> float:
        """Calculate growth rate across time periods."""
        # Group metrics by platform and metric type
        grouped_data = defaultdict(list)
        
        for data_point in metrics_data:
            key = (data_point['platform'], data_point['metric_type'])
            grouped_data[key].append(data_point)
        
        growth_rates = []
        
        for (platform, metric_type), data_points in grouped_data.items():
            if len(data_points) < 2:
                continue
            
            # Sort by collection time
            data_points.sort(key=lambda x: x['collected_at'])
            
            # Calculate growth rate between first and last point
            initial_value = float(data_points[0]['value'])
            final_value = float(data_points[-1]['value'])
            
            if initial_value > 0:
                growth_rate = ((final_value - initial_value) / initial_value) * 100
                growth_rates.append(growth_rate)
                
                platform_contributions[platform] += growth_rate
                metric_contributions[metric_type] += growth_rate
        
        return statistics.mean(growth_rates) if growth_rates else 0.0
    
    def _calculate_normalized_sum(self, metrics_data: List[Dict[str, Any]], 
                                 kpi_def: KPIDefinition,
                                 platform_contributions: Dict[str, float],
                                 metric_contributions: Dict[str, float]) -> float:
        """Calculate normalized sum across platforms."""
        platform_sums = defaultdict(float)
        
        # Sum by platform
        for data_point in metrics_data:
            platform = data_point['platform']
            value = float(data_point['value'])
            platform_sums[platform] += value
        
        # Normalize each platform sum (using log scale for large numbers)
        normalized_values = []
        for platform, total_value in platform_sums.items():
            if total_value > 0:
                normalized_value = math.log10(total_value + 1)  # +1 to handle zeros
                normalized_values.append(normalized_value)
                platform_contributions[platform] = normalized_value
        
        return sum(normalized_values)
    
    def _calculate_sum(self, metrics_data: List[Dict[str, Any]], 
                      kpi_def: KPIDefinition,
                      platform_contributions: Dict[str, float],
                      metric_contributions: Dict[str, float]) -> float:
        """Calculate simple sum of all metric values."""
        total = 0.0
        for data_point in metrics_data:
            value = float(data_point['value'])
            total += value
            platform_contributions[data_point['platform']] += value
            metric_contributions[data_point['metric_type']] += value
        return total
    
    def _calculate_average(self, metrics_data: List[Dict[str, Any]], 
                          kpi_def: KPIDefinition,
                          platform_contributions: Dict[str, float],
                          metric_contributions: Dict[str, float]) -> float:
        """Calculate simple average of all metric values."""
        if not metrics_data:
            return 0.0
        
        total = self._calculate_sum(metrics_data, kpi_def, platform_contributions, metric_contributions)
        return total / len(metrics_data)
    
    def _calculate_data_quality_score(self, metrics_data: List[Dict[str, Any]], 
                                     kpi_def: KPIDefinition) -> float:
        """Calculate data quality score based on completeness and recency."""
        if not metrics_data:
            return 0.0
        
        # Completeness score (percentage of expected platforms/metrics covered)
        expected_combinations = len(kpi_def.platforms) * len(kpi_def.metrics)
        actual_combinations = len(set((d['platform'], d['metric_type']) for d in metrics_data))
        completeness_score = min(actual_combinations / expected_combinations, 1.0)
        
        # Recency score (based on how recent the data is)
        now = datetime.now()
        recency_scores = []
        for data_point in metrics_data:
            collected_at = data_point['collected_at']
            if isinstance(collected_at, str):
                collected_at = datetime.fromisoformat(collected_at.replace('Z', '+00:00'))
            
            age_hours = (now - collected_at).total_seconds() / 3600
            recency_score = max(0, 1 - (age_hours / (24 * 7)))  # Decay over a week
            recency_scores.append(recency_score)
        
        avg_recency_score = statistics.mean(recency_scores) if recency_scores else 0.0
        
        # Combined score (weighted average)
        return (completeness_score * 0.7) + (avg_recency_score * 0.3)
    
    def _get_artist_info(self, artist_id: int) -> Optional[Dict[str, Any]]:
        """Get basic artist information."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute("""
                    SELECT a.artist_id, a.name, a.name_kr, 
                           g.name as group_name, c.name as company_name
                    FROM artists a
                    LEFT JOIN groups g ON a.group_id = g.group_id
                    LEFT JOIN companies c ON g.company_id = c.company_id
                    WHERE a.artist_id = %s
                """, (artist_id,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Failed to get artist info for {artist_id}: {e}")
            return None
    
    def _get_platform_metrics(self, artist_id: int, platforms: List[str], 
                             metrics: List[str], calculation_date: datetime,
                             time_window_days: int) -> List[Dict[str, Any]]:
        """Get platform metrics for KPI calculation."""
        try:
            start_date = calculation_date - timedelta(days=time_window_days)
            
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Build query with platform and metric filters
                platform_placeholders = ','.join(['%s'] * len(platforms))
                metric_placeholders = ','.join(['%s'] * len(metrics))
                
                cursor.execute(f"""
                    SELECT pm.platform, pm.metric_type, pm.value, pm.collected_at,
                           aa.account_identifier
                    FROM platform_metrics pm
                    JOIN artist_accounts aa ON pm.account_id = aa.account_id
                    WHERE aa.artist_id = %s
                    AND pm.platform IN ({platform_placeholders})
                    AND pm.metric_type IN ({metric_placeholders})
                    AND pm.collected_at >= %s
                    AND pm.collected_at <= %s
                    AND aa.is_active = true
                    ORDER BY pm.collected_at DESC
                """, [artist_id] + platforms + metrics + [start_date, calculation_date])
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Failed to get platform metrics: {e}")
            return []
    
    def _get_all_active_artist_ids(self) -> List[int]:
        """Get all active artist IDs."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT a.artist_id 
                    FROM artists a
                    JOIN artist_accounts aa ON a.artist_id = aa.artist_id
                    WHERE aa.is_active = true
                """)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get active artist IDs: {e}")
            return []
    
    def get_kpi_definitions(self) -> Dict[str, KPIDefinition]:
        """Get all KPI definitions."""
        return self.kpi_definitions.copy()
    
    def remove_kpi_definition(self, kpi_name: str) -> bool:
        """Remove a KPI definition."""
        if kpi_name in self.kpi_definitions:
            del self.kpi_definitions[kpi_name]
            logger.info(f"Removed KPI definition: {kpi_name}")
            return True
        return False
    
    def clear_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()
        logger.info("KPI calculation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self._cache),
            'cache_ttl_minutes': self.cache_ttl_minutes
        }
    
    # ========================================
    # Database Integration Methods
    # ========================================
    
    def reload_kpis_from_database(self) -> int:
        """Reload KPI definitions from database and return count loaded."""
        if not self.database_enabled:
            logger.warning("Database integration is disabled")
            return 0
        
        old_count = len(self.kpi_definitions)
        self.kpi_definitions.clear()
        self._load_kpis_from_database()
        new_count = len(self.kpi_definitions)
        
        logger.info(f"Reloaded KPI definitions: {old_count} -> {new_count}")
        return new_count
    
    def calculate_batch_kpis_with_storage(self, kpi_names: List[str], artist_ids: List[int],
                                         calculation_date: Optional[datetime] = None,
                                         update_rankings: bool = True) -> Dict[str, List[KPIResult]]:
        """Calculate multiple KPIs for multiple artists and store in database efficiently."""
        if not self.database_enabled:
            logger.warning("Database storage is disabled, using regular batch calculation")
            return {'results': self.calculate_batch_kpis(kpi_names, artist_ids, calculation_date)}
        
        calculation_date = calculation_date or datetime.now()
        results_by_kpi = {}
        calculations_to_store = []
        
        logger.info(f"Starting batch calculation with storage for {len(kpi_names)} KPIs, {len(artist_ids)} artists")
        
        for kpi_name in kpi_names:
            if kpi_name not in self.kpi_definitions:
                logger.warning(f"KPI definition not found: {kpi_name}")
                continue
            
            kpi_results = []
            kpi_def = self.kpi_definitions[kpi_name]
            kpi_id = kpi_def.custom_params.get('kpi_id')
            
            if not kpi_id:
                logger.warning(f"No database KPI ID found for {kpi_name}, skipping storage")
                continue
            
            for artist_id in artist_ids:
                try:
                    result = self.calculate_kpi(kpi_name, artist_id, calculation_date)
                    kpi_results.append(result)
                    
                    # Prepare calculation for bulk storage
                    calculations_to_store.append({
                        'kpi_id': kpi_id,
                        'entity_type': 'artist',
                        'entity_id': artist_id,
                        'entity_name': result.artist_name,
                        'calculated_value': result.value,
                        'calculation_date': calculation_date,
                        'normalized_value': result.normalized_value,
                        'target_achievement_rate': result.metadata.get('target_achievement_rate'),
                        'status': result.metadata.get('status', 'normal'),
                        'data_quality_score': result.data_quality_score,
                        'platform_contributions': result.platform_contributions,
                        'metric_contributions': result.metric_contributions
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate {kpi_name} for artist {artist_id}: {e}")
                    continue
            
            results_by_kpi[kpi_name] = kpi_results
        
        # Bulk store calculations
        if calculations_to_store:
            try:
                stored_count = bulk_insert_kpi_calculations(calculations_to_store)
                logger.info(f"Bulk stored {stored_count} KPI calculations")
            except Exception as e:
                logger.error(f"Failed to bulk store KPI calculations: {e}")
        
        # Update rankings for each KPI
        if update_rankings:
            for kpi_name in kpi_names:
                if kpi_name in self.kpi_definitions:
                    try:
                        kpi_def = self.kpi_definitions[kpi_name]
                        kpi_id = kpi_def.custom_params.get('kpi_id')
                        if kpi_id:
                            update_kpi_calculation_rankings(kpi_id, calculation_date)
                    except Exception as e:
                        logger.warning(f"Failed to update rankings for {kpi_name}: {e}")
        
        logger.info(f"Completed batch calculation with storage: {sum(len(results) for results in results_by_kpi.values())} total results")
        return results_by_kpi
    
    def get_kpi_calculations_from_database(self, kpi_name: Optional[str] = None, 
                                          artist_id: Optional[int] = None,
                                          days_back: int = 30) -> List[Dict[str, Any]]:
        """Get KPI calculations from database."""
        if not self.database_enabled:
            logger.warning("Database integration is disabled")
            return []
        
        try:
            from database_postgresql import get_kpi_calculations
            
            kpi_id = None
            if kpi_name and kpi_name in self.kpi_definitions:
                kpi_id = self.kpi_definitions[kpi_name].custom_params.get('kpi_id')
            
            return get_kpi_calculations(
                kpi_id=kpi_id,
                entity_type='artist',
                entity_id=artist_id,
                days_back=days_back
            )
            
        except Exception as e:
            logger.error(f"Failed to get KPI calculations from database: {e}")
            return []
    
    def get_latest_kpi_values(self, artist_id: Optional[int] = None) -> Dict[str, Any]:
        """Get latest KPI values for artist(s) from database."""
        if not self.database_enabled:
            logger.warning("Database integration is disabled")
            return {}
        
        try:
            latest_calculations = get_latest_kpi_calculations(
                entity_type='artist',
                entity_id=artist_id
            )
            
            # Organize by artist and KPI
            results = {}
            for calc in latest_calculations:
                artist_key = f"artist_{calc['entity_id']}"
                if artist_key not in results:
                    results[artist_key] = {
                        'artist_id': calc['entity_id'],
                        'artist_name': calc['entity_name'],
                        'kpis': {}
                    }
                
                results[artist_key]['kpis'][calc['kpi_name']] = {
                    'value': calc['calculated_value'],
                    'normalized_value': calc['normalized_value'],
                    'status': calc['status'],
                    'achievement_rate': calc['target_achievement_rate'],
                    'rank': calc['rank_position'],
                    'percentile': calc['percentile'],
                    'calculation_date': calc['calculation_date'],
                    'data_quality': calc['data_quality_score']
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get latest KPI values from database: {e}")
            return {}
    
    # ========================================
    # Enhanced KPI Methods
    # ========================================
    
    def define_kpi(self, name: str, description: str, formula: Union[str, Dict[str, Any]], 
                   category: Union[str, KPICategory] = KPICategory.CUSTOM,
                   target_value: Optional[float] = None,
                   warning_threshold: float = 80.0,
                   critical_threshold: float = 50.0,
                   time_window_days: int = 30,
                   platforms: Optional[List[str]] = None,
                   metrics: Optional[List[str]] = None,
                   platform_weights: Optional[Dict[str, Union[float, Dict]]] = None,
                   created_by: str = 'system') -> bool:
        """Create or update a KPI definition with JSON formula parsing.
        
        Args:
            name: Unique KPI name
            description: Human-readable description
            formula: JSON formula object or string expression
            category: KPI category (engagement, growth, etc.)
            target_value: Target value for achievement calculations
            warning_threshold: Warning threshold percentage (default 80%)
            critical_threshold: Critical threshold percentage (default 50%)
            time_window_days: Time window for calculations (default 30 days)
            platforms: List of platforms (auto-extracted from formula if None)
            metrics: List of metrics (auto-extracted from formula if None)
            platform_weights: Platform weight configuration
            created_by: User who created the KPI
        
        Returns:
            bool: True if successfully defined, False otherwise
        """
        try:
            # Parse and validate formula
            parsed_formula = self._parse_and_validate_formula(formula)
            
            # Auto-extract platforms and metrics if not provided
            if platforms is None:
                platforms = parsed_formula.get('platforms', [])
            if metrics is None:
                metrics = parsed_formula.get('metrics', [])
            
            # Convert category to enum if string
            if isinstance(category, str):
                try:
                    category = KPICategory(category)
                except ValueError:
                    category = KPICategory.CUSTOM
            
            # Process platform weights
            processed_weights = {}
            if platform_weights:
                for platform, weight_config in platform_weights.items():
                    if isinstance(weight_config, (int, float)):
                        processed_weights[platform] = PlatformWeight(platform=platform, weight=float(weight_config))
                    elif isinstance(weight_config, dict):
                        processed_weights[platform] = PlatformWeight(
                            platform=platform,
                            weight=weight_config.get('weight', 1.0),
                            metric_weights=weight_config.get('metrics', {})
                        )
            
            # Determine aggregation method from formula
            aggregation_method = AggregationMethod.WEIGHTED_AVERAGE
            if parsed_formula.get('aggregation'):
                try:
                    aggregation_method = AggregationMethod(parsed_formula['aggregation'])
                except ValueError:
                    logger.warning(f"Unknown aggregation method in formula: {parsed_formula.get('aggregation')}")
            
            # Create KPI definition
            kpi_def = KPIDefinition(
                name=name,
                category=category,
                description=description,
                formula=parsed_formula,  # Store the parsed JSON formula
                platforms=platforms,
                metrics=metrics,
                aggregation_method=aggregation_method,
                weighting_scheme=WeightingScheme.CUSTOM if platform_weights else WeightingScheme.PLATFORM_BASED,
                platform_weights=processed_weights,
                time_window_days=time_window_days,
                custom_params={
                    'target_value': target_value,
                    'warning_threshold': warning_threshold,
                    'critical_threshold': critical_threshold,
                    'json_formula': parsed_formula
                }
            )
            
            # Validate and add definition
            self._validate_kpi_definition(kpi_def)
            self.kpi_definitions[name] = kpi_def
            
            # Sync to database if enabled
            if self.database_enabled:
                kpi_id = self.sync_kpi_definition_to_database(kpi_def, created_by)
                if kpi_id:
                    kpi_def.custom_params['kpi_id'] = kpi_id
            
            logger.info(f"Successfully defined KPI: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to define KPI '{name}': {e}")
            return False
    
    def evaluate_kpi_status(self, calculated_value: float, kpi_params: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        """Evaluate KPI status based on thresholds.
        
        Args:
            calculated_value: The calculated KPI value
            kpi_params: KPI parameters containing thresholds
        
        Returns:
            Tuple of (status, achievement_rate)
            status: 'normal', 'warning', 'critical'
            achievement_rate: Percentage achievement rate if target is defined
        """
        target_value = kpi_params.get('target_value')
        warning_threshold = kpi_params.get('warning_threshold', 80.0)
        critical_threshold = kpi_params.get('critical_threshold', 50.0)
        
        achievement_rate = None
        status = 'normal'
        
        if target_value and target_value > 0:
            achievement_rate = (calculated_value / target_value) * 100
            
            if achievement_rate < critical_threshold:
                status = 'critical'
            elif achievement_rate < warning_threshold:
                status = 'warning'
            else:
                status = 'normal'
        
        return status, achievement_rate
    
    def batch_calculate_all_kpis(self, artist_ids: Optional[List[int]] = None,
                                calculation_date: Optional[datetime] = None,
                                update_rankings: bool = True,
                                chunk_size: int = 100) -> Dict[str, Any]:
        """Bulk processing for all active KPIs.
        
        Args:
            artist_ids: List of artist IDs to process (all active if None)
            calculation_date: Date for calculations (current time if None)
            update_rankings: Whether to update rankings after calculations
            chunk_size: Number of records to process in each batch
        
        Returns:
            Dict containing summary statistics and results
        """
        calculation_date = calculation_date or datetime.now()
        
        # Get artist IDs if not provided
        if artist_ids is None:
            artist_ids = self._get_all_active_artist_ids()
        
        # Get all active KPI names
        active_kpis = [name for name, kpi_def in self.kpi_definitions.items() if kpi_def.is_active]
        
        logger.info(f"Starting batch calculation for {len(active_kpis)} KPIs, {len(artist_ids)} artists")
        
        # Calculate in batches for efficiency
        total_calculated = 0
        total_failed = 0
        results_by_kpi = {}
        
        # Process in chunks to manage memory
        for i in range(0, len(artist_ids), chunk_size):
            chunk_artist_ids = artist_ids[i:i + chunk_size]
            
            chunk_results = self.calculate_batch_kpis_with_storage(
                kpi_names=active_kpis,
                artist_ids=chunk_artist_ids,
                calculation_date=calculation_date,
                update_rankings=False  # Update rankings at the end
            )
            
            # Merge results
            for kpi_name, results in chunk_results.items():
                if kpi_name not in results_by_kpi:
                    results_by_kpi[kpi_name] = []
                results_by_kpi[kpi_name].extend(results)
                total_calculated += len(results)
        
        # Update rankings for all KPIs if requested
        if update_rankings and self.database_enabled:
            for kpi_name in active_kpis:
                if kpi_name in self.kpi_definitions:
                    try:
                        kpi_def = self.kpi_definitions[kpi_name]
                        kpi_id = kpi_def.custom_params.get('kpi_id')
                        if kpi_id:
                            update_kpi_calculation_rankings(kpi_id, calculation_date)
                    except Exception as e:
                        logger.warning(f"Failed to update rankings for {kpi_name}: {e}")
                        total_failed += 1
        
        # Calculate summary statistics
        summary = {
            'total_kpis': len(active_kpis),
            'total_artists': len(artist_ids),
            'total_calculated': total_calculated,
            'total_failed': total_failed,
            'calculation_date': calculation_date,
            'success_rate': (total_calculated / (total_calculated + total_failed)) * 100 if (total_calculated + total_failed) > 0 else 0,
            'kpi_results_count': {kpi_name: len(results) for kpi_name, results in results_by_kpi.items()},
            'rankings_updated': update_rankings
        }
        
        logger.info(f"Batch calculation completed: {total_calculated} successful, {total_failed} failed")
        
        return {
            'summary': summary,
            'results_by_kpi': results_by_kpi
        }
    
    # JSON Formula Engine Helper Methods
    def _parse_and_validate_formula(self, formula: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse and validate JSON formula structure."""
        try:
            if isinstance(formula, str):
                platforms, metrics = self._extract_platforms_metrics_from_expression(formula)
                parsed_formula = {
                    'type': 'expression',
                    'expression': formula,
                    'platforms': platforms,
                    'metrics': metrics
                }
            else:
                parsed_formula = formula.copy()
            
            if 'type' not in parsed_formula:
                parsed_formula['type'] = 'formula'
            
            self._validate_formula_security(parsed_formula)
            
            if parsed_formula['type'] == 'expression':
                self._validate_expression_formula(parsed_formula)
            elif parsed_formula['type'] == 'formula':
                self._validate_json_formula(parsed_formula)
            else:
                raise FormulaParsingError(f"Unsupported formula type: {parsed_formula['type']}")
            
            return parsed_formula
        except Exception as e:
            if isinstance(e, (FormulaParsingError, SecurityError)):
                raise
            raise FormulaParsingError(f"Formula parsing failed: {e}")
    
    def _extract_platforms_metrics_from_expression(self, expression: str) -> Tuple[List[str], List[str]]:
        platforms = set()
        metrics = set()
        metric_refs = self._metric_pattern.findall(expression)
        for ref in metric_refs:
            platform, metric = ref.split('.')
            platforms.add(platform)
            metrics.add(metric)
        return list(platforms), list(metrics)
    
    def _validate_formula_security(self, formula: Dict[str, Any]) -> None:
        dangerous_keywords = [
            'import', 'exec', 'eval', 'compile', '__', 'globals', 'locals',
            'open', 'file', 'input', 'raw_input', 'reload', 'exit', 'quit'
        ]
        
        formula_str = json.dumps(formula).lower()
        for keyword in dangerous_keywords:
            if keyword in formula_str:
                raise SecurityError(f"Dangerous keyword detected: {keyword}")
        
        if 'expression' in formula:
            expression = formula['expression']
            for keyword in dangerous_keywords:
                if keyword in expression.lower():
                    raise SecurityError(f"Dangerous keyword in expression: {keyword}")
    
    def _validate_expression_formula(self, formula: Dict[str, Any]) -> None:
        if 'expression' not in formula:
            raise FormulaParsingError("Expression formula must contain 'expression' field")
        
        expression = formula['expression']
        if not isinstance(expression, str) or not expression.strip():
            raise FormulaParsingError("Expression must be a non-empty string")
        
        metric_refs = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*', expression)
        if not metric_refs:
            raise FormulaParsingError("Expression must contain at least one metric reference (platform.metric)")
    
    def _validate_json_formula(self, formula: Dict[str, Any]) -> None:
        required_fields = ['platforms', 'metrics']
        for field in required_fields:
            if field not in formula:
                raise FormulaParsingError(f"JSON formula must contain '{field}' field")
        
        if not isinstance(formula['platforms'], list) or not formula['platforms']:
            raise FormulaParsingError("Platforms must be a non-empty list")
        
        if not isinstance(formula['metrics'], list) or not formula['metrics']:
            raise FormulaParsingError("Metrics must be a non-empty list")
    
    def _calculate_kpi_with_json_formula(self, kpi_def: KPIDefinition, 
                                        metrics_data: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Calculate KPI using JSON formula definition."""
        formula = kpi_def.formula
        platform_contributions = defaultdict(float)
        metric_contributions = defaultdict(float)
        
        if formula['type'] == 'expression':
            return self._evaluate_expression_formula(formula, metrics_data, platform_contributions, metric_contributions)
        elif formula['type'] == 'formula':
            return self._evaluate_json_formula(formula, metrics_data, platform_contributions, metric_contributions)
        else:
            raise KPICalculationError(f"Unsupported formula type: {formula['type']}")
    
    def _evaluate_expression_formula(self, formula: Dict[str, Any], metrics_data: List[Dict[str, Any]],
                                   platform_contributions: Dict[str, float], 
                                   metric_contributions: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Evaluate expression-based formula safely."""
        expression = formula['expression']
        
        # Create metric value mapping
        metric_values = self._create_metric_value_mapping(metrics_data)
        
        # Replace metric references with actual values
        safe_expression = self._substitute_metric_references(expression, metric_values, 
                                                            platform_contributions, metric_contributions)
        
        # Evaluate the expression safely
        try:
            result = self._safe_eval(safe_expression)
            return float(result), dict(platform_contributions), dict(metric_contributions)
        except Exception as e:
            raise KPICalculationError(f"Expression evaluation failed: {e}")
    
    def _evaluate_json_formula(self, formula: Dict[str, Any], metrics_data: List[Dict[str, Any]],
                              platform_contributions: Dict[str, float], 
                              metric_contributions: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Evaluate JSON-structured formula."""
        platforms = formula['platforms']
        metrics = formula['metrics']
        aggregation = formula.get('aggregation', 'weighted_average')
        weights = formula.get('weights', {})
        
        # Filter and organize data
        organized_data = self._organize_metrics_data(metrics_data, platforms, metrics)
        
        # Apply aggregation method
        if aggregation == 'weighted_average':
            return self._apply_weighted_average_aggregation(organized_data, weights, platform_contributions, metric_contributions)
        elif aggregation == 'sum':
            return self._apply_sum_aggregation(organized_data, platform_contributions, metric_contributions)
        elif aggregation == 'average':
            return self._apply_average_aggregation(organized_data, platform_contributions, metric_contributions)
        else:
            # Fall back to weighted average
            return self._apply_weighted_average_aggregation(organized_data, weights, platform_contributions, metric_contributions)
    
    def _create_metric_value_mapping(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Create mapping from metric references to their values."""
        metric_values = {}
        
        # Group by platform.metric and use latest value
        for data_point in metrics_data:
            platform = data_point['platform']
            metric_type = data_point['metric_type']
            value = float(data_point['value'])
            
            key = f"{platform}.{metric_type}"
            if key not in metric_values or data_point['collected_at'] > metric_values[key + '_collected_at']:
                metric_values[key] = value
                metric_values[key + '_collected_at'] = data_point['collected_at']
        
        # Remove timestamp keys
        return {k: v for k, v in metric_values.items() if not k.endswith('_collected_at')}
    
    def _substitute_metric_references(self, expression: str, metric_values: Dict[str, float],
                                    platform_contributions: Dict[str, float], 
                                    metric_contributions: Dict[str, float]) -> str:
        """Replace metric references with values and track contributions."""
        def replace_metric(match):
            metric_ref = match.group(0)
            if metric_ref in metric_values:
                value = metric_values[metric_ref]
                # Track contributions
                try:
                    platform, metric = metric_ref.split('.')
                    platform_contributions[platform] += value
                    metric_contributions[metric] += value
                except ValueError:
                    logger.warning(f"Invalid metric reference format: {metric_ref}")
                    return '0'
                return str(value)
            else:
                # Return 0 for missing metrics with warning
                logger.warning(f"Metric reference not found: {metric_ref}")
                return '0'
        
        # Replace all metric references using the compiled pattern
        substituted = re.sub(self._metric_pattern, replace_metric, expression)
        return substituted
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expression without using eval()."""
        # Security validation
        self._validate_expression_security(expression)
        
        try:
            # Use a simple mathematical expression parser instead of eval()
            result = self._parse_and_evaluate_expression(expression.strip())
            return float(result)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")
    
    def _validate_expression_security(self, expression: str) -> None:
        """Validate that the expression is safe for evaluation."""
        # Check for dangerous keywords
        dangerous_keywords = [
            'import', 'exec', 'eval', 'compile', '__', 'globals', 'locals',
            'open', 'file', 'input', 'raw_input', 'reload', 'exit', 'quit',
            'getattr', 'setattr', 'delattr', 'hasattr', 'vars', 'dir'
        ]
        
        expression_lower = expression.lower()
        for keyword in dangerous_keywords:
            if keyword in expression_lower:
                raise SecurityError(f"Dangerous keyword detected: {keyword}")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'__[a-zA-Z_][a-zA-Z0-9_]*__',  # Dunder methods
            r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(',  # Function calls (except allowed ones)
            r'\[\s*[\'"][^\'"]*[\'"]\s*\]',  # Dictionary/list access with strings
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, expression):
                # Check if it's an allowed function
                if pattern == r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(':
                    matches = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', expression)
                    for match in matches:
                        if match not in self._allowed_functions:
                            raise SecurityError(f"Unauthorized function call: {match}")
                else:
                    raise SecurityError(f"Suspicious pattern detected in expression")
    
    def _parse_and_evaluate_expression(self, expression: str) -> float:
        """Parse and evaluate mathematical expression safely."""
        # Remove whitespace
        expression = re.sub(r'\s+', '', expression)
        
        if not expression:
            return 0.0
        
        # Handle simple number
        try:
            return float(expression)
        except ValueError:
            pass
        
        # Handle parentheses first (recursive)
        while '(' in expression:
            # Find innermost parentheses
            start = -1
            for i, char in enumerate(expression):
                if char == '(':
                    start = i
                elif char == ')':
                    if start == -1:
                        raise ValueError("Mismatched parentheses")
                    
                    # Evaluate the expression inside parentheses
                    inner_expr = expression[start + 1:i]
                    inner_result = self._parse_and_evaluate_expression(inner_expr)
                    
                    # Replace the parenthesized expression with its result
                    expression = expression[:start] + str(inner_result) + expression[i + 1:]
                    break
            else:
                if start != -1:
                    raise ValueError("Mismatched parentheses")
        
        # Handle function calls
        expression = self._evaluate_functions_in_expression(expression)
        
        # Handle operators with precedence
        return self._evaluate_operators(expression)
    
    def _evaluate_functions_in_expression(self, expression: str) -> str:
        """Evaluate function calls in the expression."""
        # Find function calls pattern: func(args)
        func_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)')
        
        while True:
            match = func_pattern.search(expression)
            if not match:
                break
            
            func_name = match.group(1)
            args_str = match.group(2)
            
            if func_name not in self._allowed_functions:
                raise SecurityError(f"Unauthorized function: {func_name}")
            
            # Parse arguments
            if args_str.strip():
                # Split by comma, but respect nested parentheses and functions
                args = self._parse_function_arguments(args_str)
                # Evaluate each argument
                evaluated_args = [self._parse_and_evaluate_expression(arg) for arg in args]
            else:
                evaluated_args = []
            
            # Call the function
            try:
                func = self._allowed_functions[func_name]
                result = func(*evaluated_args)
                
                # Replace function call with result
                expression = expression[:match.start()] + str(result) + expression[match.end():]
            except Exception as e:
                raise ValueError(f"Function {func_name} evaluation failed: {e}")
        
        return expression
    
    def _parse_function_arguments(self, args_str: str) -> List[str]:
        """Parse function arguments, respecting nested parentheses."""
        if not args_str.strip():
            return []
        
        args = []
        current_arg = ""
        paren_count = 0
        
        for char in args_str:
            if char == '(' :
                paren_count += 1
                current_arg += char
            elif char == ')':
                paren_count -= 1
                current_arg += char
            elif char == ',' and paren_count == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def _evaluate_operators(self, expression: str) -> float:
        """Evaluate operators with proper precedence."""
        if not expression:
            return 0.0
        
        # Try to parse as simple number first
        try:
            return float(expression)
        except ValueError:
            pass
        
        # Handle unary minus
        if expression.startswith('-'):
            return -self._evaluate_operators(expression[1:])
        
        if expression.startswith('+'):
            return self._evaluate_operators(expression[1:])
        
        # Handle operators by precedence (lowest to highest)
        # Level 1: or
        for op in ['or']:
            result = self._try_split_by_operator(expression, op)
            if result is not None:
                return result
        
        # Level 2: and
        for op in ['and']:
            result = self._try_split_by_operator(expression, op)
            if result is not None:
                return result
        
        # Level 3: comparison operators
        for op in ['==', '!=', '<=', '>=', '<', '>']:
            result = self._try_split_by_operator(expression, op)
            if result is not None:
                return result
        
        # Level 4: addition and subtraction
        for op in ['+', '-']:
            result = self._try_split_by_operator(expression, op, right_to_left=True)
            if result is not None:
                return result
        
        # Level 5: multiplication, division, modulo
        for op in ['*', '/', '//', '%']:
            result = self._try_split_by_operator(expression, op, right_to_left=True)
            if result is not None:
                return result
        
        # Level 6: exponentiation
        for op in ['**']:
            result = self._try_split_by_operator(expression, op)
            if result is not None:
                return result
        
        # If we get here, try to parse as a simple number one more time
        try:
            return float(expression)
        except ValueError:
            raise ValueError(f"Unable to evaluate expression: {expression}")
    
    def _try_split_by_operator(self, expression: str, op: str, right_to_left: bool = False) -> Optional[float]:
        """Try to split expression by operator and evaluate parts."""
        # Find the operator (rightmost for right-to-left associativity)
        if right_to_left:
            pos = expression.rfind(op)
        else:
            pos = expression.find(op)
        
        if pos == -1:
            return None
        
        # Skip if operator is part of another operator (e.g., ** contains *)
        if pos > 0 and expression[pos - 1:pos + len(op) + 1] != op:
            # Check if this is part of a longer operator
            longer_ops = ['**', '//', '==', '!=', '<=', '>=']
            for longer_op in longer_ops:
                if longer_op != op and op in longer_op:
                    if (pos > 0 and expression[pos - 1] in longer_op) or \
                       (pos + len(op) < len(expression) and expression[pos + len(op)] in longer_op):
                        return None
        
        left_expr = expression[:pos]
        right_expr = expression[pos + len(op):]
        
        if not left_expr or not right_expr:
            return None
        
        try:
            left_val = self._evaluate_operators(left_expr)
            right_val = self._evaluate_operators(right_expr)
            
            if op not in self._safe_operators:
                raise ValueError(f"Unsupported operator: {op}")
            
            # Handle special cases
            if op == '/' and right_val == 0:
                raise ValueError("Division by zero")
            
            if op in ['and', 'or']:
                # Convert to boolean context
                if op == 'and':
                    return float(bool(left_val) and bool(right_val))
                else:  # or
                    return float(bool(left_val) or bool(right_val))
            
            # Apply operator
            operator_func = self._safe_operators[op]
            result = operator_func(left_val, right_val)
            
            # Convert boolean results to float
            if isinstance(result, bool):
                return float(result)
            
            return float(result)
            
        except Exception as e:
            raise ValueError(f"Operator {op} evaluation failed: {e}")
    
    def _organize_metrics_data(self, metrics_data: List[Dict[str, Any]], 
                              platforms: List[str], metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Organize metrics data by platform and metric type."""
        organized = defaultdict(lambda: defaultdict(float))
        
        for data_point in metrics_data:
            platform = data_point['platform']
            metric_type = data_point['metric_type']
            
            if platform in platforms and metric_type in metrics:
                value = float(data_point['value'])
                # Use latest value for each platform-metric combination
                current_time = data_point['collected_at']
                key = f"{platform}_{metric_type}_time"
                
                if key not in organized or current_time > organized[key]:
                    organized[platform][metric_type] = value
                    organized[key] = current_time
        
        # Remove timestamp keys
        return {k: v for k, v in organized.items() if not k.endswith('_time')}
    
    def _apply_weighted_average_aggregation(self, organized_data: Dict[str, Dict[str, float]], 
                                          weights: Dict[str, float],
                                          platform_contributions: Dict[str, float], 
                                          metric_contributions: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Apply weighted average aggregation."""
        total_weighted_value = 0.0
        total_weight = 0.0
        
        for platform, metrics in organized_data.items():
            platform_weight = weights.get(platform, 1.0)
            
            for metric_type, value in metrics.items():
                metric_weight = weights.get(f"{platform}.{metric_type}", 1.0)
                final_weight = platform_weight * metric_weight
                
                weighted_value = value * final_weight
                total_weighted_value += weighted_value
                total_weight += final_weight
                
                platform_contributions[platform] += weighted_value
                metric_contributions[metric_type] += weighted_value
        
        if total_weight == 0:
            return 0.0, dict(platform_contributions), dict(metric_contributions)
        
        return total_weighted_value / total_weight, dict(platform_contributions), dict(metric_contributions)
    
    def _apply_sum_aggregation(self, organized_data: Dict[str, Dict[str, float]],
                              platform_contributions: Dict[str, float], 
                              metric_contributions: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Apply sum aggregation."""
        total = 0.0
        
        for platform, metrics in organized_data.items():
            for metric_type, value in metrics.items():
                total += value
                platform_contributions[platform] += value
                metric_contributions[metric_type] += value
        
        return total, dict(platform_contributions), dict(metric_contributions)
    
    def _apply_average_aggregation(self, organized_data: Dict[str, Dict[str, float]],
                                  platform_contributions: Dict[str, float], 
                                  metric_contributions: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Apply average aggregation."""
        total, platform_contributions, metric_contributions = self._apply_sum_aggregation(
            organized_data, platform_contributions, metric_contributions
        )
        
        count = sum(len(metrics) for metrics in organized_data.values())
        if count == 0:
            return 0.0, dict(platform_contributions), dict(metric_contributions)
        
        return total / count, dict(platform_contributions), dict(metric_contributions)
    
    def sync_kpi_definition_to_database(self, kpi_def: KPIDefinition, created_by: str = 'system') -> Optional[int]:
        """Sync a KPI definition to the database and return the kpi_id."""
        if not self.database_enabled:
            logger.warning("Database integration is disabled")
            return None
        
        try:
            from database_postgresql import add_kpi_definition
            
            # Convert KPIDefinition to database format
            formula = {
                'platforms': kpi_def.platforms,
                'metrics': kpi_def.metrics,
                'method': kpi_def.aggregation_method.value
            }
            
            # Convert platform weights to database format
            platform_weights = {}
            for platform, weight_obj in kpi_def.platform_weights.items():
                platform_weights[platform] = {
                    'weight': weight_obj.weight,
                    'metrics': weight_obj.metric_weights
                }
            
            kpi_id = add_kpi_definition(
                kpi_name=kpi_def.name,
                kpi_formula=formula,
                kpi_description=kpi_def.description,
                kpi_category=kpi_def.category.value,
                aggregation_method=kpi_def.aggregation_method.value,
                weighting_scheme=kpi_def.weighting_scheme.value,
                platform_weights=platform_weights,
                time_window_days=kpi_def.time_window_days,
                target_value=kpi_def.custom_params.get('target_value'),
                warning_threshold=kpi_def.custom_params.get('warning_threshold', 80.0),
                critical_threshold=kpi_def.custom_params.get('critical_threshold', 50.0),
                normalization_factor=kpi_def.normalization_factor,
                min_data_points=kpi_def.min_data_points,
                created_by=created_by
            )
            
            if kpi_id:
                # Update the KPI definition with database ID
                kpi_def.custom_params['kpi_id'] = kpi_id
                logger.info(f"Synced KPI definition '{kpi_def.name}' to database with ID {kpi_id}")
            
            return kpi_id
            
        except Exception as e:
            logger.error(f"Failed to sync KPI definition to database: {e}")
            return None