"""
K-Pop Dashboard Analytics - Predictive Modeling System
=====================================================

Advanced predictive modeling framework for forecasting K-Pop artist performance,
growth trends, and engagement patterns using machine learning and statistical models.

This module provides:
- Time series forecasting for artist metrics (subscribers, views, streams)
- Growth trajectory prediction with confidence intervals
- Anomaly detection in performance patterns
- Multi-platform trend correlation analysis
- Seasonal pattern recognition and modeling
- Performance milestone prediction

Author: Backend Development Team
Version: 1.0.0
Date: 2025-09-08
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from collections import defaultdict

# Core ML and Statistical Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
from scipy.optimize import minimize

# Plotting and Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Import from existing analytics modules
from .growth_rate_calculator import MetricDataPoint, GrowthRateCalculator
from .ranking_system import ArtistMetrics
from .alert_system import AlertEngine, AlertType, AlertSeverity

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of prediction models available."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    ENSEMBLE = "ensemble"


class PredictionHorizon(Enum):
    """Time horizons for predictions."""
    SHORT_TERM = "short_term"    # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"      # 1-3 months


class ConfidenceLevel(Enum):
    """Confidence levels for prediction intervals."""
    LOW = 0.68      # ~1 sigma
    MEDIUM = 0.90   # ~1.6 sigma
    HIGH = 0.95     # ~2 sigma
    VERY_HIGH = 0.99  # ~2.6 sigma


@dataclass
class PredictionResult:
    """Result structure for prediction analysis."""
    artist_id: int
    artist_name: str
    platform: str
    metric_type: str
    model_type: ModelType
    
    # Prediction data
    prediction_dates: List[datetime]
    predicted_values: List[float]
    confidence_intervals: Dict[str, List[Tuple[float, float]]]  # confidence_level -> [(lower, upper)]
    
    # Model performance
    model_accuracy: float
    rmse: float
    mae: float
    r2_score: float
    
    # Metadata
    training_data_points: int
    prediction_horizon: PredictionHorizon
    model_parameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class SeasonalityAnalysis:
    """Analysis of seasonal patterns in artist metrics."""
    artist_id: int
    platform: str
    metric_type: str
    
    # Seasonal components
    trend: List[float]
    seasonal: List[float]
    residual: List[float]
    
    # Pattern insights
    seasonal_strength: float  # 0-1 scale
    trend_strength: float     # 0-1 scale
    dominant_period: int      # days
    seasonal_peaks: List[int] # day indices of peaks
    
    # Recommendations
    optimal_posting_days: List[int]
    growth_acceleration_periods: List[Tuple[datetime, datetime]]


@dataclass
class ModelConfig:
    """Configuration for prediction models."""
    model_type: ModelType
    prediction_horizon: PredictionHorizon
    confidence_level: ConfidenceLevel
    
    # Model-specific parameters
    linear_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        'fit_intercept': True,
        'normalize': False
    })
    
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    })
    
    arima_params: Dict[str, Any] = field(default_factory=lambda: {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 7)  # Weekly seasonality
    })
    
    exponential_smoothing_params: Dict[str, Any] = field(default_factory=lambda: {
        'seasonal': 'add',
        'seasonal_periods': 7,
        'trend': 'add'
    })


class PredictionEngine:
    """
    Core prediction engine with specialized methods for different ML algorithms.
    
    This class provides direct access to specific prediction algorithms with
    fine-grained control over model parameters and evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the prediction engine.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.models_cache = {}
        self.scalers_cache = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
        logger.info(f"PredictionEngine initialized with random_state={random_state}")
    
    def linear_regression_predict(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_predict: np.ndarray,
                                 regularization: str = 'none',
                                 alpha: float = 1.0) -> Dict[str, Any]:
        """
        Perform linear regression prediction with optional regularization.
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
            y_train: Training target vector (n_samples,)
            X_predict: Prediction feature matrix (n_predict, n_features)
            regularization: Type of regularization ('none', 'ridge', 'lasso')
            alpha: Regularization strength (higher values = more regularization)
            
        Returns:
            Dictionary containing predictions, model metrics, and coefficients
        """
        logger.info(f"Running linear regression with regularization={regularization}, alpha={alpha}")
        
        try:
            # Select model based on regularization
            if regularization == 'ridge':
                model = Ridge(alpha=alpha, random_state=self.random_state)
            elif regularization == 'lasso':
                model = Lasso(alpha=alpha, random_state=self.random_state)
            else:
                model = LinearRegression()
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Generate predictions
            y_pred_train = model.predict(X_train)
            y_pred = model.predict(X_predict)
            
            # Calculate metrics
            mse = mean_squared_error(y_train, y_pred_train)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_train, y_pred_train)
            r2 = r2_score(y_train, y_pred_train)
            
            # Get model coefficients
            coefficients = {}
            if hasattr(model, 'coef_'):
                coefficients['slope'] = model.coef_.tolist() if isinstance(model.coef_, np.ndarray) else [model.coef_]
            if hasattr(model, 'intercept_'):
                coefficients['intercept'] = model.intercept_
            
            # Calculate feature importance (absolute coefficients)
            feature_importance = {}
            if hasattr(model, 'coef_'):
                coef_abs = np.abs(model.coef_) if isinstance(model.coef_, np.ndarray) else np.array([abs(model.coef_)])
                coef_sum = np.sum(coef_abs)
                if coef_sum > 0:
                    importance_normalized = coef_abs / coef_sum
                    for i, importance in enumerate(importance_normalized):
                        feature_importance[f'feature_{i}'] = float(importance)
            
            result = {
                'predictions': y_pred.tolist(),
                'training_predictions': y_pred_train.tolist(),
                'metrics': {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2_score': float(r2),
                    'accuracy': max(0.0, float(r2))  # Use R² as accuracy measure
                },
                'model_info': {
                    'type': 'linear_regression',
                    'regularization': regularization,
                    'alpha': alpha,
                    'n_features': X_train.shape[1],
                    'n_training_samples': X_train.shape[0]
                },
                'coefficients': coefficients,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Linear regression completed: R²={r2:.4f}, RMSE={rmse:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in linear regression prediction: {e}")
            raise
    
    def arima_predict(self,
                     time_series: np.ndarray,
                     n_predict: int,
                     order: Tuple[int, int, int] = (1, 1, 1),
                     seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                     auto_select: bool = True) -> Dict[str, Any]:
        """
        Perform ARIMA time series prediction.
        
        Args:
            time_series: Historical time series data (n_samples,)
            n_predict: Number of future points to predict
            order: ARIMA order (p, d, q) for non-seasonal components
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            auto_select: Whether to automatically select best ARIMA parameters
            
        Returns:
            Dictionary containing predictions, confidence intervals, and model info
        """
        logger.info(f"Running ARIMA prediction: order={order}, seasonal={seasonal_order}, n_predict={n_predict}")
        
        try:
            time_series = np.asarray(time_series)
            
            if len(time_series) < 10:
                raise ValueError("Need at least 10 data points for ARIMA modeling")
            
            # Auto-select parameters if requested (simplified version)
            if auto_select:
                best_aic = float('inf')
                best_order = order
                
                # Try different parameter combinations
                for p in range(3):
                    for d in range(2):
                        for q in range(3):
                            try:
                                temp_model = ARIMA(time_series, order=(p, d, q))
                                temp_fitted = temp_model.fit()
                                if temp_fitted.aic < best_aic:
                                    best_aic = temp_fitted.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                order = best_order
                logger.info(f"Auto-selected ARIMA order: {order}")
            
            # Fit ARIMA model
            try:
                model = ARIMA(time_series, order=order)
                fitted_model = model.fit()
            except Exception as e:
                logger.warning(f"ARIMA fitting failed with order {order}, trying simpler model: {e}")
                # Fallback to simple ARIMA(1,1,1)
                model = ARIMA(time_series, order=(1, 1, 1))
                fitted_model = model.fit()
                order = (1, 1, 1)
            
            # Generate predictions
            forecast_result = fitted_model.get_forecast(steps=n_predict)
            predictions = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Calculate in-sample metrics
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid
            
            # Handle potential NaN values in fitted values
            valid_indices = ~np.isnan(fitted_values)
            if np.sum(valid_indices) > 0:
                mse = np.mean((time_series[valid_indices] - fitted_values[valid_indices]) ** 2)
                mae = np.mean(np.abs(time_series[valid_indices] - fitted_values[valid_indices]))
                # R² calculation for time series (comparing to naive forecast)
                ss_res = np.sum((time_series[valid_indices] - fitted_values[valid_indices]) ** 2)
                ss_tot = np.sum((time_series[valid_indices] - np.mean(time_series[valid_indices])) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                mse = mae = r2 = 0.0
            
            rmse = np.sqrt(mse)
            
            # Model diagnostics
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            result = {
                'predictions': predictions.tolist(),
                'confidence_intervals': {
                    'lower': conf_int.iloc[:, 0].tolist(),
                    'upper': conf_int.iloc[:, 1].tolist()
                },
                'fitted_values': fitted_values.tolist(),
                'residuals': residuals.tolist(),
                'metrics': {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2_score': float(r2),
                    'accuracy': max(0.0, float(r2)),
                    'aic': float(aic),
                    'bic': float(bic)
                },
                'model_info': {
                    'type': 'arima',
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'n_observations': len(time_series),
                    'auto_selected': auto_select
                }
            }
            
            logger.info(f"ARIMA prediction completed: AIC={aic:.2f}, RMSE={rmse:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {e}")
            raise
    
    def ml_ensemble_predict(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_predict: np.ndarray,
                           models: List[str] = None,
                           voting_strategy: str = 'soft',
                           weights: List[float] = None) -> Dict[str, Any]:
        """
        Perform ensemble prediction using multiple ML models.
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
            y_train: Training target vector (n_samples,)
            X_predict: Prediction feature matrix (n_predict, n_features)
            models: List of model names to include ('linear', 'rf', 'ridge', 'lasso')
            voting_strategy: 'soft' (weighted average) or 'hard' (majority vote)
            weights: Custom weights for ensemble (if None, uses accuracy-based weighting)
            
        Returns:
            Dictionary containing ensemble predictions and individual model results
        """
        if models is None:
            models = ['linear', 'rf', 'ridge']
        
        logger.info(f"Running ensemble prediction with models: {models}, strategy: {voting_strategy}")
        
        try:
            # Initialize models
            model_instances = {}
            if 'linear' in models:
                model_instances['linear'] = LinearRegression()
            if 'rf' in models:
                model_instances['rf'] = RandomForestRegressor(
                    n_estimators=50, 
                    max_depth=10, 
                    random_state=self.random_state,
                    n_jobs=-1
                )
            if 'ridge' in models:
                model_instances['ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
            if 'lasso' in models:
                model_instances['lasso'] = Lasso(alpha=1.0, random_state=self.random_state)
            
            # Train models and collect predictions
            individual_results = {}
            all_predictions = []
            model_accuracies = []
            
            for name, model in model_instances.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Generate predictions
                    y_pred_train = model.predict(X_train)
                    y_pred = model.predict(X_predict)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_train, y_pred_train)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_train, y_pred_train)
                    r2 = r2_score(y_train, y_pred_train)
                    accuracy = max(0.1, r2)  # Ensure positive weights
                    
                    # Feature importance (where available)
                    feature_importance = {}
                    if hasattr(model, 'feature_importances_'):
                        for i, importance in enumerate(model.feature_importances_):
                            feature_importance[f'feature_{i}'] = float(importance)
                    elif hasattr(model, 'coef_'):
                        coef_abs = np.abs(model.coef_)
                        coef_sum = np.sum(coef_abs)
                        if coef_sum > 0:
                            for i, coef in enumerate(coef_abs):
                                feature_importance[f'feature_{i}'] = float(coef / coef_sum)
                    
                    individual_results[name] = {
                        'predictions': y_pred.tolist(),
                        'training_predictions': y_pred_train.tolist(),
                        'metrics': {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'mae': float(mae),
                            'r2_score': float(r2),
                            'accuracy': float(accuracy)
                        },
                        'feature_importance': feature_importance
                    }
                    
                    all_predictions.append(y_pred)
                    model_accuracies.append(accuracy)
                    
                    logger.info(f"Model {name} trained: R²={r2:.4f}, RMSE={rmse:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Model {name} failed: {e}")
                    continue
            
            if not all_predictions:
                raise ValueError("All ensemble models failed to train")
            
            # Calculate ensemble weights
            if weights is None:
                # Use accuracy-based weighting
                total_accuracy = sum(model_accuracies)
                weights = [acc / total_accuracy for acc in model_accuracies] if total_accuracy > 0 else [1.0 / len(model_accuracies)] * len(model_accuracies)
            
            # Generate ensemble predictions
            all_predictions = np.array(all_predictions)
            if voting_strategy == 'soft':
                ensemble_predictions = np.average(all_predictions, axis=0, weights=weights)
            else:  # hard voting (simple average)
                ensemble_predictions = np.mean(all_predictions, axis=0)
            
            # Calculate ensemble accuracy (weighted average of individual accuracies)
            ensemble_accuracy = np.average(model_accuracies, weights=weights)
            
            # Calculate prediction uncertainty (standard deviation across models)
            prediction_std = np.std(all_predictions, axis=0)
            
            result = {
                'ensemble_predictions': ensemble_predictions.tolist(),
                'prediction_uncertainty': prediction_std.tolist(),
                'individual_models': individual_results,
                'ensemble_metrics': {
                    'accuracy': float(ensemble_accuracy),
                    'n_models': len(all_predictions),
                    'voting_strategy': voting_strategy
                },
                'ensemble_info': {
                    'models_used': list(individual_results.keys()),
                    'weights': weights,
                    'weighted_accuracy': float(ensemble_accuracy)
                }
            }
            
            logger.info(f"Ensemble prediction completed with {len(all_predictions)} models, accuracy: {ensemble_accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def calculate_confidence_interval(self,
                                    predictions: np.ndarray,
                                    residuals: np.ndarray,
                                    confidence_level: float = 0.95,
                                    method: str = 'normal') -> Dict[str, List[float]]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: Model predictions (n_predict,)
            residuals: Training residuals for error estimation (n_residuals,)
            confidence_level: Confidence level (0.0 to 1.0)
            method: Method for CI calculation ('normal', 'bootstrap', 'quantile')
            
        Returns:
            Dictionary with lower and upper confidence bounds
        """
        logger.info(f"Calculating confidence intervals: level={confidence_level}, method={method}")
        
        try:
            predictions = np.asarray(predictions)
            residuals = np.asarray(residuals)
            
            if len(residuals) == 0:
                logger.warning("No residuals provided, using default uncertainty")
                residual_std = np.std(predictions) * 0.1  # 10% of prediction std as fallback
            else:
                residual_std = np.std(residuals)
            
            if method == 'normal':
                # Normal distribution assumption
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                margin = z_score * residual_std
                lower_bounds = predictions - margin
                upper_bounds = predictions + margin
                
            elif method == 'bootstrap' and len(residuals) >= 10:
                # Bootstrap method
                n_bootstrap = 1000
                bootstrap_predictions = []
                
                for _ in range(n_bootstrap):
                    # Resample residuals
                    bootstrap_residuals = np.random.choice(residuals, size=len(predictions), replace=True)
                    bootstrap_pred = predictions + bootstrap_residuals
                    bootstrap_predictions.append(bootstrap_pred)
                
                bootstrap_predictions = np.array(bootstrap_predictions)
                
                # Calculate percentiles
                lower_percentile = (1 - confidence_level) / 2 * 100
                upper_percentile = (1 + confidence_level) / 2 * 100
                
                lower_bounds = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
                upper_bounds = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
                
            elif method == 'quantile' and len(residuals) >= 5:
                # Quantile-based method using residual distribution
                lower_percentile = (1 - confidence_level) / 2
                upper_percentile = (1 + confidence_level) / 2
                
                lower_quantile = np.quantile(residuals, lower_percentile)
                upper_quantile = np.quantile(residuals, upper_percentile)
                
                lower_bounds = predictions + lower_quantile
                upper_bounds = predictions + upper_quantile
                
            else:
                # Fallback to normal method
                logger.warning(f"Method {method} not applicable, using normal distribution")
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                margin = z_score * residual_std
                lower_bounds = predictions - margin
                upper_bounds = predictions + margin
            
            result = {
                'lower_bounds': lower_bounds.tolist(),
                'upper_bounds': upper_bounds.tolist(),
                'confidence_level': confidence_level,
                'method': method,
                'margin_of_error': float(np.mean(upper_bounds - lower_bounds) / 2),
                'residual_std': float(residual_std)
            }
            
            logger.info(f"Confidence intervals calculated: method={method}, avg_margin={result['margin_of_error']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            # Return conservative intervals
            margin = np.std(predictions) * 0.2  # 20% margin as fallback
            return {
                'lower_bounds': (predictions - margin).tolist(),
                'upper_bounds': (predictions + margin).tolist(),
                'confidence_level': confidence_level,
                'method': 'fallback',
                'margin_of_error': float(margin),
                'residual_std': float(margin)
            }
    
    def evaluate_model_accuracy(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model accuracy using multiple metrics.
        
        Args:
            y_true: True target values (n_samples,)
            y_pred: Predicted values (n_samples,)
            metrics: List of metrics to calculate
                    ('mse', 'rmse', 'mae', 'r2', 'mape', 'smape')
            
        Returns:
            Dictionary containing all requested metrics
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
        
        logger.info(f"Evaluating model accuracy with metrics: {metrics}")
        
        try:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            if len(y_true) != len(y_pred):
                raise ValueError(f"Length mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})")
            
            results = {}
            
            # Mean Squared Error
            if 'mse' in metrics:
                results['mse'] = float(mean_squared_error(y_true, y_pred))
            
            # Root Mean Squared Error
            if 'rmse' in metrics:
                results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            
            # Mean Absolute Error
            if 'mae' in metrics:
                results['mae'] = float(mean_absolute_error(y_true, y_pred))
            
            # R-squared Score
            if 'r2' in metrics:
                results['r2'] = float(r2_score(y_true, y_pred))
            
            # Mean Absolute Percentage Error
            if 'mape' in metrics:
                # Avoid division by zero
                non_zero_mask = y_true != 0
                if np.sum(non_zero_mask) > 0:
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                    results['mape'] = float(mape)
                else:
                    results['mape'] = float('inf')
            
            # Symmetric Mean Absolute Percentage Error
            if 'smape' in metrics:
                numerator = np.abs(y_true - y_pred)
                denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
                # Avoid division by zero
                valid_mask = denominator != 0
                if np.sum(valid_mask) > 0:
                    smape = np.mean(numerator[valid_mask] / denominator[valid_mask]) * 100
                    results['smape'] = float(smape)
                else:
                    results['smape'] = 0.0
            
            # Max Error
            if 'max_error' in metrics:
                results['max_error'] = float(np.max(np.abs(y_true - y_pred)))
            
            # Median Absolute Error
            if 'median_ae' in metrics:
                results['median_ae'] = float(np.median(np.abs(y_true - y_pred)))
            
            # Explained Variance Score
            if 'explained_variance' in metrics:
                var_y = np.var(y_true)
                results['explained_variance'] = float(1 - np.var(y_true - y_pred) / var_y) if var_y > 0 else 0.0
            
            # Accuracy classification (custom metric for regression)
            if 'accuracy_score' in metrics:
                # Define accuracy as percentage of predictions within 10% of true values
                relative_error = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))
                accuracy = np.mean(relative_error <= 0.1) * 100  # Within 10%
                results['accuracy_score'] = float(accuracy)
            
            # Log some key metrics
            key_metrics = {k: v for k, v in results.items() if k in ['rmse', 'mae', 'r2']}
            logger.info(f"Model accuracy evaluation completed: {key_metrics}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model accuracy evaluation: {e}")
            return {metric: 0.0 for metric in metrics}


class PredictiveModelingEngine:
    """
    Advanced predictive modeling engine for K-Pop artist performance forecasting.
    
    Provides comprehensive prediction capabilities using multiple ML and statistical
    models with automatic model selection, performance validation, and uncertainty
    quantification.
    """
    
    def __init__(self, 
                 default_config: Optional[ModelConfig] = None,
                 enable_ensemble: bool = True,
                 cache_predictions: bool = True):
        """
        Initialize the predictive modeling engine.
        
        Args:
            default_config: Default configuration for models
            enable_ensemble: Whether to use ensemble methods
            cache_predictions: Whether to cache prediction results
        """
        self.default_config = default_config or ModelConfig(
            model_type=ModelType.ENSEMBLE,
            prediction_horizon=PredictionHorizon.MEDIUM_TERM,
            confidence_level=ConfidenceLevel.MEDIUM
        )
        self.enable_ensemble = enable_ensemble
        self.cache_predictions = cache_predictions
        
        # Model storage and cache
        self.trained_models: Dict[str, Any] = {}
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.scalers: Dict[str, Any] = {}
        
        # Performance tracking
        self.model_performance_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        
        logger.info(f"PredictiveModelingEngine initialized with {len(ModelType)} model types")
    
    def predict_artist_metrics(self,
                             artist_metrics: ArtistMetrics,
                             prediction_days: int = 30,
                             config: Optional[ModelConfig] = None) -> PredictionResult:
        """
        Generate predictions for artist metrics using the best available model.
        
        Args:
            artist_metrics: Historical artist metrics data
            prediction_days: Number of days to predict ahead
            config: Model configuration (uses default if None)
            
        Returns:
            Comprehensive prediction results with confidence intervals
        """
        config = config or self.default_config
        
        logger.info(f"Generating {prediction_days}-day prediction for {artist_metrics.artist_name} "
                   f"on {artist_metrics.platform}")
        
        # Check cache first
        cache_key = f"{artist_metrics.artist_id}_{artist_metrics.platform}_{prediction_days}"
        if self.cache_predictions and cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            if cached_result.expires_at and datetime.now() < cached_result.expires_at:
                logger.info(f"Returning cached prediction for {artist_metrics.artist_name}")
                return cached_result
        
        try:
            # Prepare data
            df = self._prepare_data(artist_metrics)
            
            if len(df) < 10:  # Minimum data requirement
                logger.warning(f"Insufficient data for prediction: {len(df)} points")
                return self._create_fallback_prediction(artist_metrics, prediction_days)
            
            # Select best model based on data characteristics
            best_model_type = self._select_optimal_model(df, config)
            
            # Generate predictions
            if best_model_type == ModelType.ENSEMBLE and self.enable_ensemble:
                result = self._ensemble_predict(df, artist_metrics, prediction_days, config)
            else:
                result = self._single_model_predict(df, artist_metrics, prediction_days, best_model_type, config)
            
            # Cache result
            if self.cache_predictions:
                result.expires_at = datetime.now() + timedelta(hours=6)  # Cache for 6 hours
                self.prediction_cache[cache_key] = result
            
            logger.info(f"Prediction completed with accuracy: {result.model_accuracy:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction generation: {e}")
            return self._create_fallback_prediction(artist_metrics, prediction_days)
    
    def analyze_seasonality(self,
                          artist_metrics: ArtistMetrics,
                          min_periods: int = 2) -> SeasonalityAnalysis:
        """
        Analyze seasonal patterns in artist performance data.
        
        Args:
            artist_metrics: Artist metrics data to analyze
            min_periods: Minimum number of seasonal periods required
            
        Returns:
            Detailed seasonality analysis with insights
        """
        logger.info(f"Analyzing seasonality for {artist_metrics.artist_name} on {artist_metrics.platform}")
        
        try:
            df = self._prepare_data(artist_metrics)
            
            if len(df) < 14:  # Need at least 2 weeks of data
                logger.warning("Insufficient data for seasonality analysis")
                return self._create_minimal_seasonality_analysis(artist_metrics)
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                df['value'], 
                model='additive', 
                period=7,  # Weekly seasonality
                extrapolate_trend='freq'
            )
            
            # Calculate seasonal strength
            seasonal_strength = self._calculate_seasonal_strength(decomposition)
            trend_strength = self._calculate_trend_strength(decomposition)
            
            # Identify optimal posting patterns
            optimal_days = self._identify_optimal_posting_days(decomposition.seasonal)
            growth_periods = self._identify_growth_acceleration_periods(decomposition.trend, df.index)
            
            analysis = SeasonalityAnalysis(
                artist_id=artist_metrics.artist_id,
                platform=artist_metrics.platform,
                metric_type=artist_metrics.metric_type,
                trend=decomposition.trend.fillna(0).tolist(),
                seasonal=decomposition.seasonal.tolist(),
                residual=decomposition.resid.fillna(0).tolist(),
                seasonal_strength=seasonal_strength,
                trend_strength=trend_strength,
                dominant_period=7,  # Weekly
                seasonal_peaks=self._find_seasonal_peaks(decomposition.seasonal),
                optimal_posting_days=optimal_days,
                growth_acceleration_periods=growth_periods
            )
            
            logger.info(f"Seasonality analysis completed: seasonal_strength={seasonal_strength:.3f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in seasonality analysis: {e}")
            return self._create_minimal_seasonality_analysis(artist_metrics)
    
    def predict_milestone_achievement(self,
                                    artist_metrics: ArtistMetrics,
                                    target_milestone: int,
                                    max_days: int = 365) -> Optional[datetime]:
        """
        Predict when an artist will reach a specific milestone.
        
        Args:
            artist_metrics: Current artist metrics
            target_milestone: Target value to achieve
            max_days: Maximum days to look ahead
            
        Returns:
            Predicted achievement date or None if unlikely within timeframe
        """
        if artist_metrics.current_value >= target_milestone:
            return datetime.now()  # Already achieved
        
        logger.info(f"Predicting milestone {target_milestone:,} achievement for {artist_metrics.artist_name}")
        
        try:
            # Generate prediction for max_days
            prediction = self.predict_artist_metrics(artist_metrics, max_days)
            
            # Find when milestone is crossed
            for i, (date, value) in enumerate(zip(prediction.prediction_dates, prediction.predicted_values)):
                if value >= target_milestone:
                    logger.info(f"Milestone predicted to be achieved on {date.strftime('%Y-%m-%d')}")
                    return date
            
            logger.info(f"Milestone unlikely to be achieved within {max_days} days")
            return None
            
        except Exception as e:
            logger.error(f"Error in milestone prediction: {e}")
            return None
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary of all models."""
        summary = {
            'total_predictions_generated': len(self.prediction_cache),
            'cached_predictions': len([p for p in self.prediction_cache.values() if p.expires_at and datetime.now() < p.expires_at]),
            'model_performance': {}
        }
        
        for model_type, performance_list in self.model_performance_history.items():
            if performance_list:
                avg_accuracy = np.mean([p['accuracy'] for p in performance_list])
                avg_rmse = np.mean([p['rmse'] for p in performance_list])
                summary['model_performance'][model_type] = {
                    'average_accuracy': avg_accuracy,
                    'average_rmse': avg_rmse,
                    'total_uses': len(performance_list)
                }
        
        return summary
    
    # Private helper methods
    
    def _prepare_data(self, artist_metrics: ArtistMetrics) -> pd.DataFrame:
        """Prepare and clean data for modeling."""
        data = []
        for dp in sorted(artist_metrics.data_points, key=lambda x: x.timestamp):
            data.append({
                'timestamp': dp.timestamp,
                'value': dp.value,
                'quality_score': dp.quality_score
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Remove outliers using IQR method
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['value'] >= Q1 - 1.5 * IQR) & (df['value'] <= Q3 + 1.5 * IQR)]
        
        # Fill any gaps with interpolation
        df = df.resample('D').mean().interpolate(method='time')
        
        return df
    
    def _select_optimal_model(self, df: pd.DataFrame, config: ModelConfig) -> ModelType:
        """Select the best model based on data characteristics."""
        n_points = len(df)
        
        # Calculate data characteristics
        variance = df['value'].var()
        trend_strength = self._calculate_simple_trend_strength(df['value'])
        
        # Model selection logic
        if n_points < 20:
            return ModelType.LINEAR_REGRESSION
        elif n_points < 50:
            return ModelType.EXPONENTIAL_SMOOTHING
        elif trend_strength > 0.7:
            return ModelType.ARIMA
        else:
            return ModelType.RANDOM_FOREST
    
    def _single_model_predict(self, 
                            df: pd.DataFrame,
                            artist_metrics: ArtistMetrics,
                            prediction_days: int,
                            model_type: ModelType,
                            config: ModelConfig) -> PredictionResult:
        """Generate prediction using a single model."""
        
        if model_type == ModelType.LINEAR_REGRESSION:
            return self._linear_regression_predict(df, artist_metrics, prediction_days, config)
        elif model_type == ModelType.RANDOM_FOREST:
            return self._random_forest_predict(df, artist_metrics, prediction_days, config)
        elif model_type == ModelType.ARIMA:
            return self._arima_predict(df, artist_metrics, prediction_days, config)
        elif model_type == ModelType.EXPONENTIAL_SMOOTHING:
            return self._exponential_smoothing_predict(df, artist_metrics, prediction_days, config)
        else:
            # Fallback to linear regression
            return self._linear_regression_predict(df, artist_metrics, prediction_days, config)
    
    def _linear_regression_predict(self,
                                 df: pd.DataFrame,
                                 artist_metrics: ArtistMetrics,
                                 prediction_days: int,
                                 config: ModelConfig) -> PredictionResult:
        """Generate predictions using linear regression."""
        
        # Prepare features (days since start)
        df['days'] = (df.index - df.index[0]).days
        X = df[['days']].values
        y = df['value'].values
        
        # Train model
        model = LinearRegression(**config.linear_reg_params)
        model.fit(X, y)
        
        # Calculate model performance
        y_pred_train = model.predict(X)
        accuracy = r2_score(y, y_pred_train)
        rmse = np.sqrt(mean_squared_error(y, y_pred_train))
        mae = mean_absolute_error(y, y_pred_train)
        
        # Generate future predictions
        last_day = df['days'].iloc[-1]
        future_days = np.array([[last_day + i + 1] for i in range(prediction_days)])
        predictions = model.predict(future_days)
        
        # Generate confidence intervals (simplified)
        residuals = y - y_pred_train
        residual_std = np.std(residuals)
        
        confidence_intervals = {}
        for conf_name, conf_level in [('low', 0.68), ('medium', 0.90), ('high', 0.95)]:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            margin = z_score * residual_std
            intervals = [(pred - margin, pred + margin) for pred in predictions]
            confidence_intervals[conf_name] = intervals
        
        # Create prediction dates
        last_date = df.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        return PredictionResult(
            artist_id=artist_metrics.artist_id,
            artist_name=artist_metrics.artist_name,
            platform=artist_metrics.platform,
            metric_type=artist_metrics.metric_type,
            model_type=ModelType.LINEAR_REGRESSION,
            prediction_dates=prediction_dates,
            predicted_values=predictions.tolist(),
            confidence_intervals=confidence_intervals,
            model_accuracy=accuracy,
            rmse=rmse,
            mae=mae,
            r2_score=accuracy,
            training_data_points=len(df),
            prediction_horizon=self._days_to_horizon(prediction_days),
            model_parameters=config.linear_reg_params,
            feature_importance={'days': 1.0}
        )
    
    def _random_forest_predict(self,
                             df: pd.DataFrame,
                             artist_metrics: ArtistMetrics,
                             prediction_days: int,
                             config: ModelConfig) -> PredictionResult:
        """Generate predictions using Random Forest."""
        
        # Create features
        df['days'] = (df.index - df.index[0]).days
        df['day_of_week'] = df.index.dayofweek
        df['value_lag1'] = df['value'].shift(1)
        df['value_lag7'] = df['value'].shift(7)
        df['rolling_mean_7'] = df['value'].rolling(7).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        features = ['days', 'day_of_week', 'value_lag1', 'value_lag7', 'rolling_mean_7']
        X = df[features].values
        y = df['value'].values
        
        # Train model
        model = RandomForestRegressor(**config.random_forest_params)
        model.fit(X, y)
        
        # Calculate model performance
        y_pred_train = model.predict(X)
        accuracy = r2_score(y, y_pred_train)
        rmse = np.sqrt(mean_squared_error(y, y_pred_train))
        mae = mean_absolute_error(y, y_pred_train)
        
        # Feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        # Generate future predictions (simplified - using last known values for lags)
        predictions = []
        last_values = df.tail(7)['value'].values
        
        for i in range(prediction_days):
            last_day = df['days'].iloc[-1] + i + 1
            day_of_week = (df.index[-1] + timedelta(days=i+1)).dayofweek
            
            # Use last known values for lag features
            value_lag1 = last_values[-1] if len(last_values) > 0 else df['value'].iloc[-1]
            value_lag7 = last_values[-7] if len(last_values) >= 7 else df['value'].iloc[-7]
            rolling_mean_7 = np.mean(last_values[-7:]) if len(last_values) >= 7 else df['value'].tail(7).mean()
            
            features_array = np.array([[last_day, day_of_week, value_lag1, value_lag7, rolling_mean_7]])
            pred = model.predict(features_array)[0]
            predictions.append(pred)
            
            # Update last_values for next iteration
            last_values = np.append(last_values[1:], pred) if len(last_values) >= 7 else np.append(last_values, pred)
        
        # Simplified confidence intervals
        residuals = y - y_pred_train
        residual_std = np.std(residuals)
        
        confidence_intervals = {}
        for conf_name, conf_level in [('low', 0.68), ('medium', 0.90), ('high', 0.95)]:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            margin = z_score * residual_std
            intervals = [(pred - margin, pred + margin) for pred in predictions]
            confidence_intervals[conf_name] = intervals
        
        # Create prediction dates
        last_date = df.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        return PredictionResult(
            artist_id=artist_metrics.artist_id,
            artist_name=artist_metrics.artist_name,
            platform=artist_metrics.platform,
            metric_type=artist_metrics.metric_type,
            model_type=ModelType.RANDOM_FOREST,
            prediction_dates=prediction_dates,
            predicted_values=predictions,
            confidence_intervals=confidence_intervals,
            model_accuracy=accuracy,
            rmse=rmse,
            mae=mae,
            r2_score=accuracy,
            training_data_points=len(df),
            prediction_horizon=self._days_to_horizon(prediction_days),
            model_parameters=config.random_forest_params,
            feature_importance=feature_importance
        )
    
    def _arima_predict(self,
                     df: pd.DataFrame,
                     artist_metrics: ArtistMetrics,
                     prediction_days: int,
                     config: ModelConfig) -> PredictionResult:
        """Generate predictions using ARIMA model."""
        
        try:
            # Fit ARIMA model
            model = ARIMA(df['value'], order=config.arima_params['order'])
            fitted_model = model.fit()
            
            # Generate predictions
            forecast = fitted_model.forecast(steps=prediction_days)
            conf_int = fitted_model.get_forecast(steps=prediction_days).conf_int()
            
            # Calculate model performance on training data
            fitted_values = fitted_model.fittedvalues
            accuracy = r2_score(df['value'][1:], fitted_values[1:])  # Skip first value due to differencing
            rmse = np.sqrt(mean_squared_error(df['value'][1:], fitted_values[1:]))
            mae = mean_absolute_error(df['value'][1:], fitted_values[1:])
            
            # Create confidence intervals
            confidence_intervals = {
                'medium': [(conf_int.iloc[i, 0], conf_int.iloc[i, 1]) for i in range(len(conf_int))]
            }
            
            # Create prediction dates
            last_date = df.index[-1]
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
            
            return PredictionResult(
                artist_id=artist_metrics.artist_id,
                artist_name=artist_metrics.artist_name,
                platform=artist_metrics.platform,
                metric_type=artist_metrics.metric_type,
                model_type=ModelType.ARIMA,
                prediction_dates=prediction_dates,
                predicted_values=forecast.tolist(),
                confidence_intervals=confidence_intervals,
                model_accuracy=accuracy,
                rmse=rmse,
                mae=mae,
                r2_score=accuracy,
                training_data_points=len(df),
                prediction_horizon=self._days_to_horizon(prediction_days),
                model_parameters=config.arima_params
            )
            
        except Exception as e:
            logger.warning(f"ARIMA model failed, falling back to linear regression: {e}")
            return self._linear_regression_predict(df, artist_metrics, prediction_days, config)
    
    def _exponential_smoothing_predict(self,
                                     df: pd.DataFrame,
                                     artist_metrics: ArtistMetrics,
                                     prediction_days: int,
                                     config: ModelConfig) -> PredictionResult:
        """Generate predictions using Exponential Smoothing."""
        
        try:
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                df['value'],
                trend=config.exponential_smoothing_params.get('trend'),
                seasonal=config.exponential_smoothing_params.get('seasonal'),
                seasonal_periods=config.exponential_smoothing_params.get('seasonal_periods', 7)
            )
            fitted_model = model.fit()
            
            # Generate predictions
            forecast = fitted_model.forecast(prediction_days)
            
            # Calculate model performance
            fitted_values = fitted_model.fittedvalues
            accuracy = r2_score(df['value'], fitted_values)
            rmse = np.sqrt(mean_squared_error(df['value'], fitted_values))
            mae = mean_absolute_error(df['value'], fitted_values)
            
            # Simplified confidence intervals
            residuals = df['value'] - fitted_values
            residual_std = np.std(residuals)
            
            confidence_intervals = {}
            for conf_name, conf_level in [('low', 0.68), ('medium', 0.90), ('high', 0.95)]:
                z_score = stats.norm.ppf((1 + conf_level) / 2)
                margin = z_score * residual_std
                intervals = [(pred - margin, pred + margin) for pred in forecast]
                confidence_intervals[conf_name] = intervals
            
            # Create prediction dates
            last_date = df.index[-1]
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
            
            return PredictionResult(
                artist_id=artist_metrics.artist_id,
                artist_name=artist_metrics.artist_name,
                platform=artist_metrics.platform,
                metric_type=artist_metrics.metric_type,
                model_type=ModelType.EXPONENTIAL_SMOOTHING,
                prediction_dates=prediction_dates,
                predicted_values=forecast.tolist(),
                confidence_intervals=confidence_intervals,
                model_accuracy=accuracy,
                rmse=rmse,
                mae=mae,
                r2_score=accuracy,
                training_data_points=len(df),
                prediction_horizon=self._days_to_horizon(prediction_days),
                model_parameters=config.exponential_smoothing_params
            )
            
        except Exception as e:
            logger.warning(f"Exponential Smoothing failed, falling back to linear regression: {e}")
            return self._linear_regression_predict(df, artist_metrics, prediction_days, config)
    
    def _ensemble_predict(self,
                        df: pd.DataFrame,
                        artist_metrics: ArtistMetrics,
                        prediction_days: int,
                        config: ModelConfig) -> PredictionResult:
        """Generate predictions using ensemble of models."""
        
        models_to_use = [
            ModelType.LINEAR_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.EXPONENTIAL_SMOOTHING
        ]
        
        predictions_list = []
        accuracies = []
        
        # Generate predictions from each model
        for model_type in models_to_use:
            try:
                result = self._single_model_predict(df, artist_metrics, prediction_days, model_type, config)
                predictions_list.append(result.predicted_values)
                accuracies.append(max(0.1, result.model_accuracy))  # Ensure positive weights
            except Exception as e:
                logger.warning(f"Model {model_type} failed in ensemble: {e}")
                continue
        
        if not predictions_list:
            # Fallback to linear regression
            return self._linear_regression_predict(df, artist_metrics, prediction_days, config)
        
        # Calculate weighted average (weight by accuracy)
        weights = np.array(accuracies) / np.sum(accuracies)
        ensemble_predictions = np.average(predictions_list, axis=0, weights=weights)
        
        # Calculate ensemble performance (approximate)
        ensemble_accuracy = np.average(accuracies, weights=weights)
        
        # Create prediction dates
        last_date = df.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        # Simplified confidence intervals for ensemble
        prediction_std = np.std(predictions_list, axis=0)
        confidence_intervals = {}
        for conf_name, conf_level in [('low', 0.68), ('medium', 0.90), ('high', 0.95)]:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            intervals = [(pred - z_score * std, pred + z_score * std) 
                        for pred, std in zip(ensemble_predictions, prediction_std)]
            confidence_intervals[conf_name] = intervals
        
        return PredictionResult(
            artist_id=artist_metrics.artist_id,
            artist_name=artist_metrics.artist_name,
            platform=artist_metrics.platform,
            metric_type=artist_metrics.metric_type,
            model_type=ModelType.ENSEMBLE,
            prediction_dates=prediction_dates,
            predicted_values=ensemble_predictions.tolist(),
            confidence_intervals=confidence_intervals,
            model_accuracy=ensemble_accuracy,
            rmse=0.0,  # Would need to recalculate
            mae=0.0,   # Would need to recalculate
            r2_score=ensemble_accuracy,
            training_data_points=len(df),
            prediction_horizon=self._days_to_horizon(prediction_days),
            model_parameters={'ensemble_weights': weights.tolist(), 'models_used': [m.value for m in models_to_use]},
            feature_importance={'ensemble_composition': dict(zip([m.value for m in models_to_use], weights))}
        )
    
    def _create_fallback_prediction(self, 
                                  artist_metrics: ArtistMetrics, 
                                  prediction_days: int) -> PredictionResult:
        """Create a simple fallback prediction when models fail."""
        
        # Simple linear extrapolation from recent data
        recent_values = [dp.value for dp in artist_metrics.data_points[-5:]]
        if len(recent_values) < 2:
            # Constant prediction
            predictions = [artist_metrics.current_value] * prediction_days
        else:
            # Linear trend from recent data
            x = np.arange(len(recent_values))
            slope, intercept = np.polyfit(x, recent_values, 1)
            start_x = len(recent_values)
            predictions = [slope * (start_x + i) + intercept for i in range(prediction_days)]
        
        # Simple confidence intervals (±10% of current value)
        margin = artist_metrics.current_value * 0.1
        confidence_intervals = {
            'medium': [(pred - margin, pred + margin) for pred in predictions]
        }
        
        # Create prediction dates
        if artist_metrics.data_points:
            last_date = max(dp.timestamp for dp in artist_metrics.data_points)
        else:
            last_date = datetime.now()
            
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        return PredictionResult(
            artist_id=artist_metrics.artist_id,
            artist_name=artist_metrics.artist_name,
            platform=artist_metrics.platform,
            metric_type=artist_metrics.metric_type,
            model_type=ModelType.LINEAR_REGRESSION,
            prediction_dates=prediction_dates,
            predicted_values=predictions,
            confidence_intervals=confidence_intervals,
            model_accuracy=0.5,  # Low accuracy indicator
            rmse=0.0,
            mae=0.0,
            r2_score=0.5,
            training_data_points=len(artist_metrics.data_points),
            prediction_horizon=self._days_to_horizon(prediction_days),
            model_parameters={'fallback': True}
        )
    
    def _days_to_horizon(self, days: int) -> PredictionHorizon:
        """Convert days to prediction horizon enum."""
        if days <= 7:
            return PredictionHorizon.SHORT_TERM
        elif days <= 28:
            return PredictionHorizon.MEDIUM_TERM
        else:
            return PredictionHorizon.LONG_TERM
    
    def _calculate_seasonal_strength(self, decomposition) -> float:
        """Calculate the strength of seasonal component."""
        try:
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            total_var = seasonal_var + residual_var
            return seasonal_var / total_var if total_var > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_trend_strength(self, decomposition) -> float:
        """Calculate the strength of trend component."""
        try:
            trend_var = np.var(decomposition.trend.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            total_var = trend_var + residual_var
            return trend_var / total_var if total_var > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_simple_trend_strength(self, series: pd.Series) -> float:
        """Calculate simple trend strength using correlation with time."""
        try:
            time_index = np.arange(len(series))
            correlation = stats.pearsonr(time_index, series)[0]
            return abs(correlation)
        except:
            return 0.0
    
    def _identify_optimal_posting_days(self, seasonal_component: pd.Series) -> List[int]:
        """Identify days of week with highest seasonal component."""
        try:
            # Group by day of week and get average seasonal value
            day_averages = {}
            for i, value in enumerate(seasonal_component):
                day_of_week = i % 7
                if day_of_week not in day_averages:
                    day_averages[day_of_week] = []
                day_averages[day_of_week].append(value)
            
            # Calculate average for each day
            day_scores = {day: np.mean(values) for day, values in day_averages.items()}
            
            # Return top 3 days
            sorted_days = sorted(day_scores.items(), key=lambda x: x[1], reverse=True)
            return [day for day, score in sorted_days[:3]]
        except:
            return [0, 6]  # Default to Sunday and Saturday
    
    def _identify_growth_acceleration_periods(self, 
                                            trend_component: pd.Series, 
                                            date_index: pd.DatetimeIndex) -> List[Tuple[datetime, datetime]]:
        """Identify periods of accelerating growth."""
        try:
            # Calculate trend acceleration (second derivative)
            trend_clean = trend_component.dropna()
            if len(trend_clean) < 3:
                return []
            
            # Calculate acceleration
            acceleration = np.diff(np.diff(trend_clean))
            
            # Find periods of positive acceleration
            acceleration_threshold = np.percentile(acceleration, 75)  # Top 25%
            
            periods = []
            in_period = False
            start_idx = 0
            
            for i, acc in enumerate(acceleration):
                if acc > acceleration_threshold and not in_period:
                    start_idx = i
                    in_period = True
                elif acc <= acceleration_threshold and in_period:
                    # End of period
                    if i - start_idx > 2:  # At least 3 days
                        start_date = date_index[start_idx]
                        end_date = date_index[i]
                        periods.append((start_date, end_date))
                    in_period = False
            
            return periods[:5]  # Return top 5 periods
        except:
            return []
    
    def _find_seasonal_peaks(self, seasonal_component: pd.Series) -> List[int]:
        """Find indices of seasonal peaks."""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(seasonal_component.dropna(), height=0)
            return peaks.tolist()[:10]  # Return top 10 peaks
        except:
            return []
    
    def _create_minimal_seasonality_analysis(self, artist_metrics: ArtistMetrics) -> SeasonalityAnalysis:
        """Create minimal seasonality analysis when insufficient data."""
        return SeasonalityAnalysis(
            artist_id=artist_metrics.artist_id,
            platform=artist_metrics.platform,
            metric_type=artist_metrics.metric_type,
            trend=[0.0] * 7,
            seasonal=[0.0] * 7,
            residual=[0.0] * 7,
            seasonal_strength=0.0,
            trend_strength=0.0,
            dominant_period=7,
            seasonal_peaks=[],
            optimal_posting_days=[0, 6],  # Default to weekends
            growth_acceleration_periods=[]
        )


# Export main components
__all__ = [
    'PredictionEngine',
    'PredictiveModelingEngine',
    'PredictionResult',
    'SeasonalityAnalysis',
    'ModelConfig',
    'ModelType',
    'PredictionHorizon',
    'ConfidenceLevel'
]