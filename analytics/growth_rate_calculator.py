"""
Growth Rate Calculator Module
============================

Backend module for K-Pop artist growth rate calculations with multiple algorithms
and statistical methods for performance analysis.

Features:
- Multiple calculation methods (simple, rolling average, weighted, exponential)
- Time-series growth rate calculations (daily, weekly, monthly, quarterly, YoY)
- Statistical validation and outlier detection
- Data quality scoring and confidence intervals
- Batch processing for multiple accounts/platforms

Author: Backend Development Team
Date: 2025-09-08
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import math
import statistics
from dataclasses import dataclass
from enum import Enum

# Configure module logger
logger = logging.getLogger(__name__)

class CalculationMethod(Enum):
    """Enumeration of available calculation methods."""
    SIMPLE = "simple"
    ROLLING_AVERAGE = "rolling_average"
    WEIGHTED = "weighted"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"

class GrowthPeriod(Enum):
    """Enumeration of growth calculation periods."""
    DAILY = 1
    WEEKLY = 7
    MONTHLY = 30
    QUARTERLY = 90
    YEARLY = 365

@dataclass
class MetricDataPoint:
    """Data structure for a single metric measurement."""
    value: int
    timestamp: datetime
    platform: str
    metric_type: str
    quality_score: float = 1.0

@dataclass
class GrowthRateResult:
    """Result structure for growth rate calculations."""
    growth_rate: float
    calculation_method: str
    period_days: int
    data_quality_score: float
    confidence_interval: Optional[Tuple[float, float]] = None
    is_significant: bool = False
    outlier_detected: bool = False
    sample_size: int = 0

class GrowthRateCalculator:
    """
    Main class for calculating growth rates with various algorithms and statistical validations.
    
    This class provides comprehensive growth rate calculation capabilities with:
    - Multiple calculation methods
    - Statistical validation
    - Data quality assessment
    - Outlier detection
    - Confidence interval calculation
    """
    
    def __init__(self, 
                 min_data_points: int = 2,
                 outlier_threshold: float = 3.0,
                 confidence_level: float = 0.95):
        """
        Initialize the growth rate calculator.
        
        Args:
            min_data_points: Minimum number of data points required for calculation
            outlier_threshold: Z-score threshold for outlier detection
            confidence_level: Confidence level for interval calculations (0.0-1.0)
        """
        self.min_data_points = max(2, min_data_points)
        self.outlier_threshold = outlier_threshold
        self.confidence_level = confidence_level
        
        logger.info(f"GrowthRateCalculator initialized: min_points={min_data_points}, "
                   f"outlier_threshold={outlier_threshold}, confidence={confidence_level}")
    
    def calculate_growth_rate(self,
                             data_points: List[MetricDataPoint],
                             method: CalculationMethod = CalculationMethod.ROLLING_AVERAGE,
                             period: GrowthPeriod = GrowthPeriod.DAILY) -> Optional[GrowthRateResult]:
        """
        Calculate growth rate using specified method and period.
        
        Args:
            data_points: List of metric data points sorted by timestamp (newest first)
            method: Calculation method to use
            period: Growth period for calculation
            
        Returns:
            GrowthRateResult object or None if calculation not possible
        """
        if not self._validate_data_points(data_points):
            logger.warning("Insufficient or invalid data points for growth calculation")
            return None
        
        try:
            # Sort data points by timestamp (newest first)
            sorted_points = sorted(data_points, key=lambda x: x.timestamp, reverse=True)
            
            # Detect outliers
            outliers = self._detect_outliers(sorted_points)
            filtered_points = [p for i, p in enumerate(sorted_points) if i not in outliers]
            
            if len(filtered_points) < self.min_data_points:
                logger.warning("Too many outliers detected, insufficient clean data")
                return None
            
            # Calculate growth rate based on method
            if method == CalculationMethod.SIMPLE:
                growth_rate = self._calculate_simple_growth(filtered_points, period)
            elif method == CalculationMethod.ROLLING_AVERAGE:
                growth_rate = self._calculate_rolling_average_growth(filtered_points, period)
            elif method == CalculationMethod.WEIGHTED:
                growth_rate = self._calculate_weighted_growth(filtered_points, period)
            elif method == CalculationMethod.EXPONENTIAL_SMOOTHING:
                growth_rate = self._calculate_exponential_smoothing_growth(filtered_points, period)
            else:
                raise ValueError(f"Unknown calculation method: {method}")
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(filtered_points, outliers)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(filtered_points, growth_rate)
            
            # Determine statistical significance
            is_significant = self._is_statistically_significant(filtered_points, growth_rate)
            
            result = GrowthRateResult(
                growth_rate=growth_rate,
                calculation_method=method.value,
                period_days=period.value,
                data_quality_score=quality_score,
                confidence_interval=confidence_interval,
                is_significant=is_significant,
                outlier_detected=len(outliers) > 0,
                sample_size=len(filtered_points)
            )
            
            logger.info(f"Growth rate calculated: {growth_rate:.2f}% using {method.value} method")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating growth rate: {e}")
            return None
    
    def calculate_multiple_periods(self,
                                  data_points: List[MetricDataPoint],
                                  method: CalculationMethod = CalculationMethod.ROLLING_AVERAGE) -> Dict[str, Optional[GrowthRateResult]]:
        """
        Calculate growth rates for multiple time periods.
        
        Args:
            data_points: List of metric data points
            method: Calculation method to use
            
        Returns:
            Dictionary with period names as keys and results as values
        """
        results = {}
        
        for period in GrowthPeriod:
            result = self.calculate_growth_rate(data_points, method, period)
            results[f"{period.name.lower()}_growth"] = result
        
        return results
    
    # =============================================================================
    # SPECIALIZED CALCULATION METHODS (Backend Implementation)
    # =============================================================================
    
    def calculate_daily_growth(self,
                              data_points: List[MetricDataPoint],
                              use_average: bool = False,
                              exclude_weekends: bool = False) -> Optional[GrowthRateResult]:
        """
        Calculate daily growth rate with advanced options and statistical validation.
        
        Args:
            data_points: List of metric data points (newest first)
            use_average: If True, use average of multiple daily comparisons
            exclude_weekends: If True, skip weekend data points for business metrics
            
        Returns:
            GrowthRateResult with daily growth analysis or None if calculation fails
        """
        if not self._validate_data_points(data_points):
            logger.warning("Invalid data points for daily growth calculation")
            return None
        
        try:
            # Sort and filter data points
            sorted_points = sorted(data_points, key=lambda x: x.timestamp, reverse=True)
            
            # Filter weekends if requested
            if exclude_weekends:
                filtered_points = []
                for point in sorted_points:
                    # Skip Saturday (5) and Sunday (6) 
                    if point.timestamp.weekday() not in [5, 6]:
                        filtered_points.append(point)
                sorted_points = filtered_points
            
            if len(sorted_points) < 2:
                logger.warning("Insufficient data points after filtering for daily growth")
                return None
            
            # Detect and remove outliers
            outliers = self._detect_outliers(sorted_points)
            clean_points = [p for i, p in enumerate(sorted_points) if i not in outliers]
            
            if len(clean_points) < 2:
                logger.warning("Insufficient clean data points for daily growth calculation")
                return None
            
            if use_average and len(clean_points) >= 5:
                # Calculate average daily growth from multiple comparisons
                daily_growth_rates = []
                
                for i in range(min(7, len(clean_points) - 1)):  # Max 7 daily comparisons
                    current = clean_points[i]
                    previous = clean_points[i + 1]
                    
                    # Calculate days between measurements
                    days_diff = (current.timestamp - previous.timestamp).days
                    if days_diff == 0:
                        continue  # Skip same-day measurements
                    
                    if previous.value > 0:
                        # Normalize to daily growth rate
                        period_growth = ((current.value - previous.value) / previous.value) * 100
                        daily_growth = period_growth / days_diff
                        daily_growth_rates.append(daily_growth)
                
                if not daily_growth_rates:
                    return None
                
                # Use median to reduce impact of extreme values
                growth_rate = statistics.median(daily_growth_rates)
                calculation_method = "daily_average_median"
                
                logger.debug(f"Calculated average daily growth from {len(daily_growth_rates)} comparisons")
                
            else:
                # Simple daily growth between most recent two points
                current = clean_points[0]
                previous = clean_points[1]
                
                # Find the actual previous day's data point
                target_date = current.timestamp - timedelta(days=1)
                previous = self._find_closest_data_point(clean_points[1:], target_date)
                
                if previous is None or previous.value == 0:
                    logger.warning("No suitable previous data point found for daily growth")
                    return None
                
                # Calculate days between measurements for normalization
                days_diff = max(1, (current.timestamp - previous.timestamp).days)
                period_growth = ((current.value - previous.value) / previous.value) * 100
                
                # Normalize to daily rate
                growth_rate = period_growth / days_diff
                calculation_method = "daily_simple"
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(clean_points, outliers)
            
            # Calculate confidence interval for daily growth
            confidence_interval = None
            if use_average and len(clean_points) >= 5:
                try:
                    values = [p.value for p in clean_points[:7]]  # Recent week
                    daily_changes = []
                    for i in range(1, len(values)):
                        if values[i] > 0:
                            daily_change = ((values[i-1] - values[i]) / values[i]) * 100
                            daily_changes.append(daily_change)
                    
                    if len(daily_changes) >= 3:
                        std_err = statistics.stdev(daily_changes) / math.sqrt(len(daily_changes))
                        margin = 1.96 * std_err  # 95% confidence
                        confidence_interval = (
                            round(growth_rate - margin, 2),
                            round(growth_rate + margin, 2)
                        )
                except (statistics.StatisticsError, ZeroDivisionError):
                    pass
            
            # Determine statistical significance
            is_significant = abs(growth_rate) >= 1.0 and quality_score >= 0.7
            
            result = GrowthRateResult(
                growth_rate=round(growth_rate, 4),
                calculation_method=calculation_method,
                period_days=1,
                data_quality_score=quality_score,
                confidence_interval=confidence_interval,
                is_significant=is_significant,
                outlier_detected=len(outliers) > 0,
                sample_size=len(clean_points)
            )
            
            logger.info(f"Daily growth calculated: {growth_rate:.2f}% using {calculation_method}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating daily growth: {e}")
            return None
    
    def calculate_rolling_growth(self,
                                data_points: List[MetricDataPoint],
                                window_days: int = 7,
                                step_days: int = 1,
                                method: str = 'mean') -> Optional[GrowthRateResult]:
        """
        Calculate rolling growth rate with configurable window and statistical methods.
        
        Args:
            data_points: List of metric data points (newest first)
            window_days: Size of the rolling window in days
            step_days: Step size for rolling calculation (default: 1 day)
            method: Statistical method ('mean', 'median', 'weighted')
            
        Returns:
            GrowthRateResult with rolling growth analysis or None if calculation fails
        """
        if not self._validate_data_points(data_points):
            logger.warning("Invalid data points for rolling growth calculation")
            return None
        
        if window_days < 2:
            raise ValueError("Window size must be at least 2 days")
        
        if method not in ['mean', 'median', 'weighted']:
            raise ValueError("Method must be 'mean', 'median', or 'weighted'")
        
        try:
            # Sort and clean data
            sorted_points = sorted(data_points, key=lambda x: x.timestamp, reverse=True)
            outliers = self._detect_outliers(sorted_points)
            clean_points = [p for i, p in enumerate(sorted_points) if i not in outliers]
            
            if len(clean_points) < window_days * 2:
                logger.warning(f"Insufficient data for {window_days}-day rolling window")
                return None
            
            # Group data points by date for daily aggregation
            daily_data = {}
            for point in clean_points:
                date_key = point.timestamp.date()
                if date_key not in daily_data:
                    daily_data[date_key] = []
                daily_data[date_key].append(point)
            
            # Calculate daily averages
            daily_averages = {}
            for date_key, points in daily_data.items():
                if method == 'weighted':
                    total_weighted = sum(p.value * p.quality_score for p in points)
                    total_weight = sum(p.quality_score for p in points)
                    daily_averages[date_key] = total_weighted / total_weight if total_weight > 0 else 0
                elif method == 'median':
                    daily_averages[date_key] = statistics.median([p.value for p in points])
                else:  # mean
                    daily_averages[date_key] = statistics.mean([p.value for p in points])
            
            # Sort dates and calculate rolling windows
            sorted_dates = sorted(daily_averages.keys(), reverse=True)
            
            if len(sorted_dates) < window_days * 2:
                logger.warning("Insufficient daily data for rolling calculation")
                return None
            
            # Calculate rolling growth rates
            rolling_growth_rates = []
            
            for i in range(0, len(sorted_dates) - window_days, step_days):
                if i + window_days * 2 > len(sorted_dates):
                    break
                
                # Current window (most recent)
                current_window = sorted_dates[i:i + window_days]
                current_values = [daily_averages[date] for date in current_window]
                
                # Previous window
                previous_window = sorted_dates[i + window_days:i + window_days * 2]
                previous_values = [daily_averages[date] for date in previous_window]
                
                # Calculate window averages
                if method == 'median':
                    current_avg = statistics.median(current_values)
                    previous_avg = statistics.median(previous_values)
                else:  # mean or weighted (already processed)
                    current_avg = statistics.mean(current_values)
                    previous_avg = statistics.mean(previous_values)
                
                if previous_avg > 0:
                    window_growth = ((current_avg - previous_avg) / previous_avg) * 100
                    rolling_growth_rates.append(window_growth)
            
            if not rolling_growth_rates:
                logger.warning("Could not calculate any rolling growth rates")
                return None
            
            # Final growth rate calculation
            if method == 'median':
                final_growth_rate = statistics.median(rolling_growth_rates)
            else:
                final_growth_rate = statistics.mean(rolling_growth_rates)
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(clean_points, outliers)
            
            # Enhance quality score based on rolling window stability
            if len(rolling_growth_rates) >= 3:
                try:
                    cv = statistics.stdev(rolling_growth_rates) / abs(statistics.mean(rolling_growth_rates))
                    stability_bonus = max(0, 0.1 - cv / 10)  # Reward stability
                    quality_score = min(1.0, quality_score + stability_bonus)
                except (statistics.StatisticsError, ZeroDivisionError):
                    pass
            
            # Calculate confidence interval
            confidence_interval = None
            if len(rolling_growth_rates) >= 3:
                try:
                    std_err = statistics.stdev(rolling_growth_rates) / math.sqrt(len(rolling_growth_rates))
                    margin = 1.96 * std_err
                    confidence_interval = (
                        round(final_growth_rate - margin, 2),
                        round(final_growth_rate + margin, 2)
                    )
                except statistics.StatisticsError:
                    pass
            
            # Statistical significance based on consistency and magnitude
            is_significant = (
                abs(final_growth_rate) >= 2.0 and  # Meaningful magnitude
                quality_score >= 0.7 and           # Good data quality
                len(rolling_growth_rates) >= 3     # Sufficient samples
            )
            
            result = GrowthRateResult(
                growth_rate=round(final_growth_rate, 4),
                calculation_method=f"rolling_{window_days}d_{method}",
                period_days=window_days,
                data_quality_score=round(quality_score, 3),
                confidence_interval=confidence_interval,
                is_significant=is_significant,
                outlier_detected=len(outliers) > 0,
                sample_size=len(rolling_growth_rates)
            )
            
            logger.info(f"Rolling growth ({window_days}d {method}): {final_growth_rate:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating rolling growth: {e}")
            return None
    
    def calculate_yoy_growth(self,
                            data_points: List[MetricDataPoint],
                            seasonal_adjustment: bool = False,
                            use_quarters: bool = False) -> Optional[GrowthRateResult]:
        """
        Calculate year-over-year growth rate with seasonal adjustment options.
        
        Args:
            data_points: List of metric data points (newest first)
            seasonal_adjustment: Apply seasonal adjustment for cyclical patterns
            use_quarters: Use quarterly comparison instead of exact year
            
        Returns:
            GrowthRateResult with YoY growth analysis or None if calculation fails
        """
        if not self._validate_data_points(data_points):
            logger.warning("Invalid data points for YoY growth calculation")
            return None
        
        try:
            # Sort and clean data
            sorted_points = sorted(data_points, key=lambda x: x.timestamp, reverse=True)
            outliers = self._detect_outliers(sorted_points)
            clean_points = [p for i, p in enumerate(sorted_points) if i not in outliers]
            
            # Need at least 1 year of data
            min_days = 90 if use_quarters else 365  # 3 months or 1 year
            oldest_point = min(clean_points, key=lambda x: x.timestamp)
            data_span_days = (clean_points[0].timestamp - oldest_point.timestamp).days
            
            if data_span_days < min_days:
                logger.warning(f"Insufficient historical data for YoY calculation: {data_span_days} < {min_days} days")
                return None
            
            current_point = clean_points[0]
            
            # Find comparison point (1 year ago or 1 quarter ago)
            if use_quarters:
                target_date = current_point.timestamp - timedelta(days=90)
                calculation_method = "quarterly_yoy"
            else:
                target_date = current_point.timestamp - timedelta(days=365)
                calculation_method = "annual_yoy"
            
            # Find the closest historical point
            comparison_point = self._find_closest_data_point(clean_points[1:], target_date)
            
            if comparison_point is None or comparison_point.value == 0:
                logger.warning("No suitable historical comparison point found")
                return None
            
            # Basic YoY calculation
            basic_growth = ((current_point.value - comparison_point.value) / comparison_point.value) * 100
            
            # Apply seasonal adjustment if requested
            if seasonal_adjustment and not use_quarters:
                adjusted_growth = self._apply_seasonal_adjustment(
                    clean_points, current_point, comparison_point, basic_growth
                )
                growth_rate = adjusted_growth
                calculation_method += "_seasonal_adjusted"
            else:
                growth_rate = basic_growth
            
            # Annualize the growth rate if using quarters
            if use_quarters:
                # Convert quarterly growth to annualized rate
                actual_days = (current_point.timestamp - comparison_point.timestamp).days
                annualization_factor = 365.0 / actual_days
                growth_rate = ((1 + growth_rate / 100) ** annualization_factor - 1) * 100
            
            # Calculate data quality score with historical depth bonus
            base_quality = self._calculate_data_quality_score(clean_points, outliers)
            
            # Bonus for longer historical data
            years_of_data = data_span_days / 365.0
            history_bonus = min(0.15, years_of_data / 10)  # Up to 15% bonus for 10+ years
            quality_score = min(1.0, base_quality + history_bonus)
            
            # Calculate confidence interval using historical volatility
            confidence_interval = None
            if len(clean_points) >= 8:  # Need quarterly data minimum
                try:
                    # Calculate historical quarterly/yearly changes
                    historical_changes = []
                    period_days = 90 if use_quarters else 365
                    
                    for i in range(0, len(clean_points) - 1):
                        current = clean_points[i]
                        target_historical = current.timestamp - timedelta(days=period_days)
                        historical = self._find_closest_data_point(clean_points[i+1:], target_historical)
                        
                        if historical and historical.value > 0:
                            change = ((current.value - historical.value) / historical.value) * 100
                            historical_changes.append(change)
                            
                        if len(historical_changes) >= 4:  # Stop after 4 comparisons
                            break
                    
                    if len(historical_changes) >= 3:
                        std_dev = statistics.stdev(historical_changes)
                        margin = 1.96 * (std_dev / math.sqrt(len(historical_changes)))
                        confidence_interval = (
                            round(growth_rate - margin, 2),
                            round(growth_rate + margin, 2)
                        )
                except statistics.StatisticsError:
                    pass
            
            # YoY growth is significant if it's substantial and consistent
            is_significant = (
                abs(growth_rate) >= 5.0 and      # At least 5% YoY change
                quality_score >= 0.6 and         # Reasonable data quality
                data_span_days >= min_days * 1.2  # Some buffer beyond minimum
            )
            
            result = GrowthRateResult(
                growth_rate=round(growth_rate, 4),
                calculation_method=calculation_method,
                period_days=365 if not use_quarters else 90,
                data_quality_score=round(quality_score, 3),
                confidence_interval=confidence_interval,
                is_significant=is_significant,
                outlier_detected=len(outliers) > 0,
                sample_size=len(clean_points)
            )
            
            logger.info(f"YoY growth calculated: {growth_rate:.2f}% using {calculation_method}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating YoY growth: {e}")
            return None
    
    def batch_calculate_growth_rates(self,
                                    batch_data: Dict[str, List[MetricDataPoint]],
                                    calculation_types: List[str] = None,
                                    parallel_processing: bool = True,
                                    progress_callback: Optional[callable] = None) -> Dict[str, Dict[str, Optional[GrowthRateResult]]]:
        """
        Calculate growth rates for multiple accounts/platforms in batch with optimized processing.
        
        Args:
            batch_data: Dictionary with keys as identifiers and values as data points
            calculation_types: List of calculation types ['daily', 'rolling', 'yoy', 'all']
            parallel_processing: Enable parallel processing for performance (future enhancement)
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            Dictionary with nested results: {identifier: {calc_type: result}}
        """
        if not batch_data:
            logger.warning("No data provided for batch calculation")
            return {}
        
        if calculation_types is None:
            calculation_types = ['daily', 'rolling', 'yoy']
        
        if 'all' in calculation_types:
            calculation_types = ['daily', 'rolling', 'yoy']
        
        logger.info(f"Starting batch calculation for {len(batch_data)} accounts, "
                   f"types: {calculation_types}")
        
        batch_results = {}
        processed_count = 0
        total_count = len(batch_data)
        
        # Batch processing statistics
        successful_calculations = 0
        failed_calculations = 0
        calculation_times = []
        
        for identifier, data_points in batch_data.items():
            start_time = datetime.now()
            account_results = {}
            
            try:
                # Validate data points for this account
                if not self._validate_data_points(data_points):
                    logger.warning(f"Invalid data points for account {identifier}")
                    account_results = {calc_type: None for calc_type in calculation_types}
                    failed_calculations += len(calculation_types)
                else:
                    # Calculate each requested growth type
                    for calc_type in calculation_types:
                        try:
                            if calc_type == 'daily':
                                result = self.calculate_daily_growth(
                                    data_points=data_points,
                                    use_average=True,
                                    exclude_weekends=False
                                )
                            elif calc_type == 'rolling':
                                result = self.calculate_rolling_growth(
                                    data_points=data_points,
                                    window_days=7,
                                    method='mean'
                                )
                            elif calc_type == 'yoy':
                                result = self.calculate_yoy_growth(
                                    data_points=data_points,
                                    seasonal_adjustment=True,
                                    use_quarters=False
                                )
                            else:
                                logger.warning(f"Unknown calculation type: {calc_type}")
                                result = None
                            
                            account_results[calc_type] = result
                            
                            if result is not None:
                                successful_calculations += 1
                                logger.debug(f"Account {identifier} {calc_type}: {result.growth_rate:.2f}%")
                            else:
                                failed_calculations += 1
                                
                        except Exception as e:
                            logger.error(f"Error calculating {calc_type} for {identifier}: {e}")
                            account_results[calc_type] = None
                            failed_calculations += 1
                
                batch_results[identifier] = account_results
                
                # Track processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                calculation_times.append(processing_time)
                
                processed_count += 1
                
                # Progress callback
                if progress_callback:
                    progress_percentage = (processed_count / total_count) * 100
                    progress_callback(processed_count, total_count, progress_percentage)
                
                # Log progress every 10% or 100 accounts
                if processed_count % max(1, total_count // 10) == 0 or processed_count % 100 == 0:
                    logger.info(f"Batch progress: {processed_count}/{total_count} accounts processed "
                              f"({processed_count/total_count*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error processing account {identifier}: {e}")
                account_results = {calc_type: None for calc_type in calculation_types}
                batch_results[identifier] = account_results
                failed_calculations += len(calculation_types)
        
        # Final batch statistics
        total_calculations = successful_calculations + failed_calculations
        success_rate = (successful_calculations / total_calculations) * 100 if total_calculations > 0 else 0
        avg_processing_time = statistics.mean(calculation_times) if calculation_times else 0
        
        logger.info(f"Batch calculation completed:")
        logger.info(f"  Accounts processed: {processed_count}/{total_count}")
        logger.info(f"  Successful calculations: {successful_calculations}")
        logger.info(f"  Failed calculations: {failed_calculations}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Average processing time: {avg_processing_time:.3f}s per account")
        
        # Add batch metadata to results
        batch_results['_batch_metadata'] = {
            'total_accounts': total_count,
            'processed_accounts': processed_count,
            'successful_calculations': successful_calculations,
            'failed_calculations': failed_calculations,
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'total_processing_time': sum(calculation_times),
            'calculation_types': calculation_types,
            'timestamp': datetime.now()
        }
        
        return batch_results
    
    # =============================================================================
    # HELPER METHODS FOR SPECIALIZED CALCULATIONS
    # =============================================================================
    
    def _apply_seasonal_adjustment(self,
                                  data_points: List[MetricDataPoint],
                                  current_point: MetricDataPoint,
                                  comparison_point: MetricDataPoint,
                                  basic_growth: float) -> float:
        """Apply seasonal adjustment to YoY growth calculation."""
        try:
            # Simple seasonal adjustment based on monthly patterns
            current_month = current_point.timestamp.month
            comparison_month = comparison_point.timestamp.month
            
            # Group data by month to find seasonal patterns
            monthly_data = {}
            for point in data_points[-365:]:  # Last year of data
                month_key = point.timestamp.month
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                monthly_data[month_key].append(point.value)
            
            # Calculate monthly averages
            monthly_averages = {}
            for month, values in monthly_data.items():
                if len(values) >= 2:  # Need at least 2 data points per month
                    monthly_averages[month] = statistics.mean(values)
            
            if len(monthly_averages) >= 6:  # Need at least 6 months of data
                overall_average = statistics.mean(monthly_averages.values())
                
                # Calculate seasonal factors
                current_seasonal_factor = monthly_averages.get(current_month, overall_average) / overall_average
                comparison_seasonal_factor = monthly_averages.get(comparison_month, overall_average) / overall_average
                
                # Apply seasonal adjustment
                seasonal_adjustment = current_seasonal_factor / comparison_seasonal_factor
                adjusted_growth = basic_growth / seasonal_adjustment
                
                logger.debug(f"Applied seasonal adjustment: {basic_growth:.2f}% → {adjusted_growth:.2f}%")
                return adjusted_growth
            
            # Return basic growth if insufficient seasonal data
            return basic_growth
            
        except Exception as e:
            logger.warning(f"Seasonal adjustment failed, using basic growth: {e}")
            return basic_growth
    
    def _validate_data_points(self, data_points: List[MetricDataPoint]) -> bool:
        """Validate data points for calculation requirements."""
        if not data_points or len(data_points) < self.min_data_points:
            return False
        
        # Check for non-negative values
        if any(point.value < 0 for point in data_points):
            logger.warning("Negative values detected in data points")
            return False
        
        # Check for valid timestamps
        if any(not isinstance(point.timestamp, datetime) for point in data_points):
            logger.warning("Invalid timestamp types in data points")
            return False
        
        return True
    
    def _detect_outliers(self, data_points: List[MetricDataPoint]) -> List[int]:
        """Detect outliers using Z-score method."""
        if len(data_points) < 3:
            return []
        
        values = [point.value for point in data_points]
        mean_val = statistics.mean(values)
        
        try:
            std_val = statistics.stdev(values)
            if std_val == 0:
                return []
        except statistics.StatisticsError:
            return []
        
        outliers = []
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_val)
            if z_score > self.outlier_threshold:
                outliers.append(i)
                logger.debug(f"Outlier detected at index {i}: value={value}, z_score={z_score:.2f}")
        
        return outliers
    
    def _calculate_simple_growth(self, 
                                data_points: List[MetricDataPoint], 
                                period: GrowthPeriod) -> float:
        """Calculate simple growth rate between current and previous period."""
        current = data_points[0]
        
        # Find appropriate comparison point based on period
        target_date = current.timestamp - timedelta(days=period.value)
        previous = self._find_closest_data_point(data_points[1:], target_date)
        
        if previous is None or previous.value == 0:
            return 0.0
        
        growth_rate = ((current.value - previous.value) / previous.value) * 100
        return round(growth_rate, 4)
    
    def _calculate_rolling_average_growth(self,
                                         data_points: List[MetricDataPoint],
                                         period: GrowthPeriod) -> float:
        """Calculate growth rate using rolling average."""
        if len(data_points) < period.value:
            # Fall back to simple calculation if insufficient data
            return self._calculate_simple_growth(data_points, period)
        
        # Calculate rolling averages
        window_size = min(period.value, len(data_points) // 2)
        
        current_avg = statistics.mean([p.value for p in data_points[:window_size]])
        previous_avg = statistics.mean([p.value for p in data_points[window_size:window_size*2]])
        
        if previous_avg == 0:
            return 0.0
        
        growth_rate = ((current_avg - previous_avg) / previous_avg) * 100
        return round(growth_rate, 4)
    
    def _calculate_weighted_growth(self,
                                  data_points: List[MetricDataPoint],
                                  period: GrowthPeriod) -> float:
        """Calculate weighted growth rate based on data quality scores."""
        current = data_points[0]
        target_date = current.timestamp - timedelta(days=period.value)
        previous = self._find_closest_data_point(data_points[1:], target_date)
        
        if previous is None or previous.value == 0:
            return 0.0
        
        # Weight the calculation by data quality scores
        quality_weight = (current.quality_score + previous.quality_score) / 2
        
        base_growth = ((current.value - previous.value) / previous.value) * 100
        weighted_growth = base_growth * quality_weight
        
        return round(weighted_growth, 4)
    
    def _calculate_exponential_smoothing_growth(self,
                                              data_points: List[MetricDataPoint],
                                              period: GrowthPeriod) -> float:
        """Calculate growth rate using exponential smoothing."""
        if len(data_points) < 3:
            return self._calculate_simple_growth(data_points, period)
        
        # Exponential smoothing with alpha = 0.3
        alpha = 0.3
        values = [p.value for p in reversed(data_points)]  # Oldest first for smoothing
        
        smoothed = [values[0]]
        for value in values[1:]:
            smoothed_value = alpha * value + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)
        
        # Calculate growth rate using smoothed values
        current_smooth = smoothed[-1]
        target_index = max(0, len(smoothed) - period.value - 1)
        previous_smooth = smoothed[target_index]
        
        if previous_smooth == 0:
            return 0.0
        
        growth_rate = ((current_smooth - previous_smooth) / previous_smooth) * 100
        return round(growth_rate, 4)
    
    def _find_closest_data_point(self,
                                data_points: List[MetricDataPoint],
                                target_date: datetime) -> Optional[MetricDataPoint]:
        """Find the data point closest to the target date."""
        if not data_points:
            return None
        
        closest = min(data_points, 
                     key=lambda p: abs((p.timestamp - target_date).total_seconds()))
        
        # Only return if within reasonable range (within 2x the period)
        time_diff = abs((closest.timestamp - target_date).days)
        if time_diff <= 60:  # Within 2 months
            return closest
        
        return None
    
    def _calculate_data_quality_score(self,
                                     data_points: List[MetricDataPoint],
                                     outliers: List[int]) -> float:
        """Calculate overall data quality score."""
        if not data_points:
            return 0.0
        
        # Base quality from individual data point scores
        base_quality = statistics.mean([p.quality_score for p in data_points])
        
        # Penalize for outliers
        outlier_penalty = len(outliers) / len(data_points) * 0.2
        
        # Bonus for larger sample size (up to 10% bonus)
        size_bonus = min(0.1, len(data_points) / 100)
        
        # Check for data consistency (low variance is better)
        values = [p.value for p in data_points]
        if len(values) > 1:
            try:
                cv = statistics.stdev(values) / statistics.mean(values)  # Coefficient of variation
                consistency_bonus = max(0, 0.1 - cv / 10)  # Lower CV gets bonus
            except (statistics.StatisticsError, ZeroDivisionError):
                consistency_bonus = 0
        else:
            consistency_bonus = 0
        
        quality_score = base_quality - outlier_penalty + size_bonus + consistency_bonus
        return round(min(1.0, max(0.0, quality_score)), 3)
    
    def _calculate_confidence_interval(self,
                                      data_points: List[MetricDataPoint],
                                      growth_rate: float) -> Optional[Tuple[float, float]]:
        """Calculate confidence interval for the growth rate."""
        if len(data_points) < 3:
            return None
        
        try:
            values = [p.value for p in data_points]
            std_error = statistics.stdev(values) / math.sqrt(len(values))
            
            # Use t-distribution approximation (simplified)
            t_value = 1.96 if self.confidence_level >= 0.95 else 1.645
            
            margin_of_error = t_value * (std_error / statistics.mean(values)) * 100
            
            lower_bound = growth_rate - margin_of_error
            upper_bound = growth_rate + margin_of_error
            
            return (round(lower_bound, 2), round(upper_bound, 2))
            
        except (statistics.StatisticsError, ZeroDivisionError):
            return None
    
    def _is_statistically_significant(self,
                                     data_points: List[MetricDataPoint],
                                     growth_rate: float) -> bool:
        """Determine if the growth rate is statistically significant."""
        if len(data_points) < 3 or abs(growth_rate) < 1.0:
            return False
        
        confidence_interval = self._calculate_confidence_interval(data_points, growth_rate)
        if confidence_interval is None:
            return False
        
        # Significant if confidence interval doesn't include zero
        lower, upper = confidence_interval
        return not (lower <= 0 <= upper)

# Standalone utility functions for direct use

def calculate_daily_growth_rate(current_value: int, previous_value: int) -> float:
    """Calculate simple daily growth rate."""
    if previous_value == 0:
        return 0.0
    return round(((current_value - previous_value) / previous_value) * 100, 2)

def calculate_weekly_growth_rate(data_points: List[MetricDataPoint]) -> Optional[float]:
    """Calculate weekly growth rate from data points."""
    calculator = GrowthRateCalculator()
    result = calculator.calculate_growth_rate(data_points, 
                                            CalculationMethod.ROLLING_AVERAGE,
                                            GrowthPeriod.WEEKLY)
    return result.growth_rate if result else None

def calculate_monthly_growth_rate(data_points: List[MetricDataPoint]) -> Optional[float]:
    """Calculate monthly growth rate from data points."""
    calculator = GrowthRateCalculator()
    result = calculator.calculate_growth_rate(data_points,
                                            CalculationMethod.ROLLING_AVERAGE, 
                                            GrowthPeriod.MONTHLY)
    return result.growth_rate if result else None

def calculate_rolling_average_growth(values: List[int], window_size: int = 7) -> float:
    """Calculate rolling average growth rate."""
    if len(values) < window_size * 2:
        return 0.0
    
    current_avg = sum(values[:window_size]) / window_size
    previous_avg = sum(values[window_size:window_size*2]) / window_size
    
    if previous_avg == 0:
        return 0.0
    
    return round(((current_avg - previous_avg) / previous_avg) * 100, 2)

def calculate_compound_growth_rate(initial_value: int, final_value: int, periods: int) -> float:
    """Calculate compound annual growth rate (CAGR)."""
    if initial_value <= 0 or final_value <= 0 or periods <= 0:
        return 0.0
    
    cagr = (pow(final_value / initial_value, 1 / periods) - 1) * 100
    return round(cagr, 2)

# Module-level constants for validation
MIN_VALID_VALUE = 0
MAX_GROWTH_RATE = 10000.0  # 10,000% maximum growth rate (sanity check)
MIN_DATA_POINTS_REQUIRED = 2

# Export main components
__all__ = [
    'GrowthRateCalculator',
    'MetricDataPoint', 
    'GrowthRateResult',
    'CalculationMethod',
    'GrowthPeriod',
    'calculate_daily_growth_rate',
    'calculate_weekly_growth_rate', 
    'calculate_monthly_growth_rate',
    'calculate_rolling_average_growth',
    'calculate_compound_growth_rate'
]