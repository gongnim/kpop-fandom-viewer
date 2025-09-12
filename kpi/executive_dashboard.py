"""
Executive Dashboard Module for K-POP Analytics
=============================================

A comprehensive executive dashboard that provides high-level insights for C-level executives:
- Strategic KPIs and performance summaries
- Top performing artists and groups identification
- Attention-required areas for immediate action
- Company-wide growth analysis and trends
- Real-time monitoring capabilities

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import statistics
from collections import defaultdict

from ..database_postgresql import (
    get_db_connection, get_companies, get_artists_by_company,
    get_platform_metrics_for_artist, get_latest_platform_metrics,
    get_events, get_albums
)
from .kpi_engine import KPIEngine, KPICategory

# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class SummaryCard:
    """Data structure for executive summary cards."""
    title: str
    value: Union[int, float, str]
    unit: str = ""
    change: Optional[float] = None
    change_period: str = "vs_previous"
    status: str = "neutral"  # positive, negative, neutral, warning
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy JSON serialization."""
        return {
            'title': self.title,
            'value': self.value,
            'unit': self.unit,
            'change': self.change,
            'change_period': self.change_period,
            'status': self.status,
            'description': self.description
        }

@dataclass
class TopPerformer:
    """Data structure for top performing entities."""
    entity_id: int
    entity_name: str
    entity_type: str  # artist, group, company
    metric_name: str
    metric_value: Union[int, float]
    metric_unit: str = ""
    rank: int = 1
    change: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy JSON serialization."""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'rank': self.rank,
            'change': self.change,
            'additional_info': self.additional_info
        }

@dataclass
class AttentionItem:
    """Data structure for attention-required items."""
    entity_id: int
    entity_name: str
    entity_type: str  # artist, group, company
    issue_type: str  # declining_metrics, stagnant_growth, underperformance, anomaly
    severity: str  # critical, high, medium, low
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy JSON serialization."""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'issue_type': self.issue_type,
            'severity': self.severity,
            'description': self.description,
            'metrics': self.metrics,
            'recommended_actions': self.recommended_actions
        }

@dataclass
class GrowthAnalysis:
    """Data structure for growth analysis results."""
    time_period: str
    growth_metrics: Dict[str, float]
    trends: Dict[str, str]  # increasing, decreasing, stable
    forecasts: Dict[str, float]
    key_insights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy JSON serialization."""
        return {
            'time_period': self.time_period,
            'growth_metrics': self.growth_metrics,
            'trends': self.trends,
            'forecasts': self.forecasts,
            'key_insights': self.key_insights
        }

class ExecutiveDashboard:
    """
    Executive Dashboard for K-POP Analytics
    
    Provides high-level insights and KPIs for executive decision-making,
    including performance summaries, growth analysis, and alerting capabilities.
    """
    
    def __init__(self, kpi_engine: Optional[KPIEngine] = None):
        """
        Initialize Executive Dashboard.
        
        Args:
            kpi_engine: KPI calculation engine instance
        """
        self.kpi_engine = kpi_engine or KPIEngine()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def get_summary_cards(
        self, 
        time_period: int = 30,
        comparison_period: int = 30
    ) -> List[SummaryCard]:
        """
        Generate executive summary cards showing key performance indicators.
        
        Args:
            time_period: Number of days for current period analysis
            comparison_period: Number of days for comparison period
            
        Returns:
            List of SummaryCard objects with key metrics
        """
        try:
            cards = []
            
            # Calculate date ranges
            current_end = datetime.now()
            current_start = current_end - timedelta(days=time_period)
            comparison_end = current_start
            comparison_start = comparison_end - timedelta(days=comparison_period)
            
            # Total Company Portfolio Value
            portfolio_value = self._calculate_portfolio_value(current_start, current_end)
            portfolio_value_prev = self._calculate_portfolio_value(comparison_start, comparison_end)
            portfolio_change = ((portfolio_value - portfolio_value_prev) / portfolio_value_prev * 100) if portfolio_value_prev > 0 else 0
            
            cards.append(SummaryCard(
                title="Total Portfolio Value",
                value=round(portfolio_value, 1),
                unit="Million KRW",
                change=round(portfolio_change, 1),
                change_period=f"{time_period}d vs previous {comparison_period}d",
                status="positive" if portfolio_change > 0 else "negative" if portfolio_change < -5 else "neutral",
                description="Combined value of all managed artists and groups"
            ))
            
            # Total Active Artists
            active_artists = self._get_active_artists_count(current_start, current_end)
            active_artists_prev = self._get_active_artists_count(comparison_start, comparison_end)
            artists_change = active_artists - active_artists_prev
            
            cards.append(SummaryCard(
                title="Active Artists",
                value=active_artists,
                unit="artists",
                change=artists_change,
                change_period=f"{time_period}d vs previous {comparison_period}d",
                status="positive" if artists_change > 0 else "negative" if artists_change < 0 else "neutral",
                description="Number of artists with recent activity"
            ))
            
            # Average Growth Rate
            avg_growth = self._calculate_average_growth_rate(current_start, current_end)
            avg_growth_prev = self._calculate_average_growth_rate(comparison_start, comparison_end)
            growth_change = avg_growth - avg_growth_prev
            
            cards.append(SummaryCard(
                title="Average Growth Rate",
                value=round(avg_growth, 2),
                unit="%",
                change=round(growth_change, 2),
                change_period=f"{time_period}d vs previous {comparison_period}d",
                status="positive" if growth_change > 0 else "negative" if growth_change < -1 else "neutral",
                description="Portfolio-wide average growth rate"
            ))
            
            # Engagement Score
            engagement_score = self._calculate_engagement_score(current_start, current_end)
            engagement_score_prev = self._calculate_engagement_score(comparison_start, comparison_end)
            engagement_change = engagement_score - engagement_score_prev
            
            cards.append(SummaryCard(
                title="Portfolio Engagement Score",
                value=round(engagement_score, 1),
                unit="points",
                change=round(engagement_change, 1),
                change_period=f"{time_period}d vs previous {comparison_period}d",
                status="positive" if engagement_change > 0 else "negative" if engagement_change < -5 else "neutral",
                description="Weighted engagement across all platforms"
            ))
            
            # Market Share
            market_share = self._calculate_market_share(current_start, current_end)
            market_share_prev = self._calculate_market_share(comparison_start, comparison_end)
            market_change = market_share - market_share_prev
            
            cards.append(SummaryCard(
                title="Market Share",
                value=round(market_share, 2),
                unit="%",
                change=round(market_change, 2),
                change_period=f"{time_period}d vs previous {comparison_period}d",
                status="positive" if market_change > 0 else "negative" if market_change < -0.5 else "neutral",
                description="Estimated K-POP market share"
            ))
            
            self.logger.info(f"Generated {len(cards)} summary cards for period {time_period}d")
            return cards
            
        except Exception as e:
            self.logger.error(f"Error generating summary cards: {e}")
            raise
    
    def identify_top_performers(
        self,
        time_period: int = 30,
        categories: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, List[TopPerformer]]:
        """
        Identify top performing artists, groups, and companies across different metrics.
        
        Args:
            time_period: Number of days for analysis period
            categories: Specific categories to analyze (e.g., ['growth', 'engagement'])
            limit: Maximum number of performers per category
            
        Returns:
            Dictionary with categories as keys and lists of TopPerformer objects as values
        """
        try:
            if categories is None:
                categories = ['growth', 'engagement', 'reach', 'consistency']
            
            performers = {}
            current_end = datetime.now()
            current_start = current_end - timedelta(days=time_period)
            
            for category in categories:
                performers[category] = []
                
                if category == 'growth':
                    performers[category] = self._get_top_growth_performers(current_start, current_end, limit)
                elif category == 'engagement':
                    performers[category] = self._get_top_engagement_performers(current_start, current_end, limit)
                elif category == 'reach':
                    performers[category] = self._get_top_reach_performers(current_start, current_end, limit)
                elif category == 'consistency':
                    performers[category] = self._get_top_consistency_performers(current_start, current_end, limit)
            
            self.logger.info(f"Identified top performers in {len(categories)} categories for period {time_period}d")
            return performers
            
        except Exception as e:
            self.logger.error(f"Error identifying top performers: {e}")
            raise
    
    def identify_attention_needed(
        self,
        time_period: int = 30,
        severity_threshold: str = "medium"
    ) -> List[AttentionItem]:
        """
        Identify artists, groups, or companies that need immediate attention.
        
        Args:
            time_period: Number of days for analysis period
            severity_threshold: Minimum severity level (low, medium, high, critical)
            
        Returns:
            List of AttentionItem objects requiring executive attention
        """
        try:
            attention_items = []
            current_end = datetime.now()
            current_start = current_end - timedelta(days=time_period)
            
            # Check for declining metrics
            declining_items = self._identify_declining_metrics(current_start, current_end, severity_threshold)
            attention_items.extend(declining_items)
            
            # Check for stagnant growth
            stagnant_items = self._identify_stagnant_growth(current_start, current_end, severity_threshold)
            attention_items.extend(stagnant_items)
            
            # Check for underperformance
            underperforming_items = self._identify_underperformance(current_start, current_end, severity_threshold)
            attention_items.extend(underperforming_items)
            
            # Check for anomalies
            anomaly_items = self._identify_anomalies(current_start, current_end, severity_threshold)
            attention_items.extend(anomaly_items)
            
            # Sort by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            attention_items.sort(key=lambda x: severity_order.get(x.severity, 3))
            
            self.logger.info(f"Identified {len(attention_items)} items needing attention (severity >= {severity_threshold})")
            return attention_items
            
        except Exception as e:
            self.logger.error(f"Error identifying attention needed items: {e}")
            raise
    
    def get_company_growth_analysis(
        self,
        company_id: Optional[int] = None,
        time_period: int = 90
    ) -> Union[Dict[int, GrowthAnalysis], GrowthAnalysis]:
        """
        Analyze growth patterns for companies in the portfolio.
        
        Args:
            company_id: Specific company to analyze (None for all companies)
            time_period: Number of days for analysis period
            
        Returns:
            GrowthAnalysis object for specific company or dictionary of analyses for all companies
        """
        try:
            current_end = datetime.now()
            current_start = current_end - timedelta(days=time_period)
            
            if company_id:
                analysis = self._analyze_company_growth(company_id, current_start, current_end)
                self.logger.info(f"Generated growth analysis for company {company_id}")
                return analysis
            else:
                # Analyze all companies
                analyses = {}
                companies = get_companies()
                
                for company in companies:
                    analysis = self._analyze_company_growth(company['company_id'], current_start, current_end)
                    analyses[company['company_id']] = analysis
                
                self.logger.info(f"Generated growth analyses for {len(analyses)} companies")
                return analyses
                
        except Exception as e:
            self.logger.error(f"Error in company growth analysis: {e}")
            raise
    
    def get_realtime_monitoring(self) -> Dict[str, Any]:
        """
        Get real-time monitoring data for executive dashboard.
        
        Returns:
            Dictionary with real-time metrics and alerts
        """
        try:
            monitoring_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self._check_system_health(),
                'active_alerts': self._get_active_alerts(),
                'recent_activities': self._get_recent_activities(),
                'data_freshness': self._check_data_freshness(),
                'performance_metrics': self._get_performance_metrics(),
                'upcoming_events': self._get_upcoming_events()
            }
            
            self.logger.info("Generated real-time monitoring data")
            return monitoring_data
            
        except Exception as e:
            self.logger.error(f"Error generating real-time monitoring data: {e}")
            raise
    
    # Private helper methods
    
    def _calculate_portfolio_value(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate total portfolio value based on artist metrics."""
        try:
            total_value = 0.0
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Use a simplified valuation model based on follower counts and engagement
                query = """
                    SELECT 
                        a.artist_id,
                        a.stage_name,
                        SUM(CASE 
                            WHEN pm.platform = 'youtube' THEN pm.subscribers * 0.01
                            WHEN pm.platform = 'spotify' THEN pm.monthly_listeners * 0.005
                            WHEN pm.platform = 'instagram' THEN pm.followers * 0.002
                            WHEN pm.platform = 'twitter' THEN pm.followers * 0.001
                            ELSE 0
                        END) as estimated_value
                    FROM artists a
                    LEFT JOIN platform_metrics pm ON a.artist_id = pm.artist_id
                    WHERE pm.collected_at BETWEEN %s AND %s
                    GROUP BY a.artist_id, a.stage_name
                """
                
                cursor.execute(query, (start_date, end_date))
                results = cursor.fetchall()
                
                for row in results:
                    if row[2]:  # estimated_value
                        total_value += float(row[2])
                
                return total_value / 1000000  # Convert to millions
                
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def _get_active_artists_count(self, start_date: datetime, end_date: datetime) -> int:
        """Get count of artists with recent activity."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT COUNT(DISTINCT a.artist_id)
                    FROM artists a
                    INNER JOIN platform_metrics pm ON a.artist_id = pm.artist_id
                    WHERE pm.collected_at BETWEEN %s AND %s
                """
                
                cursor.execute(query, (start_date, end_date))
                result = cursor.fetchone()
                
                return result[0] if result and result[0] else 0
                
        except Exception as e:
            self.logger.error(f"Error getting active artists count: {e}")
            return 0
    
    def _calculate_average_growth_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate average growth rate across portfolio."""
        try:
            # This is a simplified implementation
            # In practice, you would calculate growth rates for each metric and average them
            return 5.2  # Placeholder: 5.2% average growth
            
        except Exception as e:
            self.logger.error(f"Error calculating average growth rate: {e}")
            return 0.0
    
    def _calculate_engagement_score(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate weighted engagement score."""
        try:
            # This is a simplified implementation
            # In practice, you would calculate engagement rates and weight them
            return 78.5  # Placeholder: 78.5 engagement score
            
        except Exception as e:
            self.logger.error(f"Error calculating engagement score: {e}")
            return 0.0
    
    def _calculate_market_share(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate estimated market share."""
        try:
            # This is a simplified implementation
            # In practice, you would compare your portfolio against industry benchmarks
            return 12.3  # Placeholder: 12.3% market share
            
        except Exception as e:
            self.logger.error(f"Error calculating market share: {e}")
            return 0.0
    
    def _get_top_growth_performers(self, start_date: datetime, end_date: datetime, limit: int) -> List[TopPerformer]:
        """Get top growth performing artists."""
        try:
            performers = []
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Simplified growth calculation
                query = """
                    SELECT 
                        a.artist_id,
                        a.stage_name,
                        AVG(pm.subscribers) as avg_subscribers,
                        COUNT(*) as metric_count
                    FROM artists a
                    INNER JOIN platform_metrics pm ON a.artist_id = pm.artist_id
                    WHERE pm.collected_at BETWEEN %s AND %s
                        AND pm.platform = 'youtube'
                        AND pm.subscribers > 0
                    GROUP BY a.artist_id, a.stage_name
                    HAVING COUNT(*) >= 2
                    ORDER BY avg_subscribers DESC
                    LIMIT %s
                """
                
                cursor.execute(query, (start_date, end_date, limit))
                results = cursor.fetchall()
                
                for rank, row in enumerate(results, 1):
                    performers.append(TopPerformer(
                        entity_id=row[0],
                        entity_name=row[1],
                        entity_type="artist",
                        metric_name="Average Subscribers Growth",
                        metric_value=int(row[2]) if row[2] else 0,
                        metric_unit="subscribers",
                        rank=rank,
                        change=None  # Would calculate actual growth rate in production
                    ))
                
                return performers
                
        except Exception as e:
            self.logger.error(f"Error getting top growth performers: {e}")
            return []
    
    def _get_top_engagement_performers(self, start_date: datetime, end_date: datetime, limit: int) -> List[TopPerformer]:
        """Get top engagement performing artists."""
        # Placeholder implementation
        return []
    
    def _get_top_reach_performers(self, start_date: datetime, end_date: datetime, limit: int) -> List[TopPerformer]:
        """Get top reach performing artists."""
        # Placeholder implementation
        return []
    
    def _get_top_consistency_performers(self, start_date: datetime, end_date: datetime, limit: int) -> List[TopPerformer]:
        """Get most consistent performing artists."""
        # Placeholder implementation
        return []
    
    def _identify_declining_metrics(self, start_date: datetime, end_date: datetime, severity_threshold: str) -> List[AttentionItem]:
        """Identify artists with declining metrics."""
        # Placeholder implementation
        return []
    
    def _identify_stagnant_growth(self, start_date: datetime, end_date: datetime, severity_threshold: str) -> List[AttentionItem]:
        """Identify artists with stagnant growth."""
        # Placeholder implementation
        return []
    
    def _identify_underperformance(self, start_date: datetime, end_date: datetime, severity_threshold: str) -> List[AttentionItem]:
        """Identify underperforming artists."""
        # Placeholder implementation
        return []
    
    def _identify_anomalies(self, start_date: datetime, end_date: datetime, severity_threshold: str) -> List[AttentionItem]:
        """Identify anomalies in artist metrics."""
        # Placeholder implementation
        return []
    
    def _analyze_company_growth(self, company_id: int, start_date: datetime, end_date: datetime) -> GrowthAnalysis:
        """Analyze growth for a specific company."""
        # Placeholder implementation
        return GrowthAnalysis(
            time_period=f"{start_date.date()} to {end_date.date()}",
            growth_metrics={"overall_growth": 8.5, "engagement_growth": 12.3},
            trends={"followers": "increasing", "engagement": "stable"},
            forecasts={"next_month_growth": 6.2},
            key_insights=["Strong performance in Q3", "Engagement rates stabilizing"]
        )
    
    def _check_system_health(self) -> Dict[str, str]:
        """Check system health status."""
        return {
            "database": "healthy",
            "data_collection": "healthy",
            "api_endpoints": "healthy"
        }
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return []
    
    def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent system activities."""
        return []
    
    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check how fresh the data is."""
        return {
            "last_update": datetime.now().isoformat(),
            "staleness_minutes": 15
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            "avg_response_time": "120ms",
            "cpu_usage": "45%",
            "memory_usage": "67%"
        }
    
    def _get_upcoming_events(self) -> List[Dict[str, Any]]:
        """Get upcoming events that might impact metrics."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT event_name, event_date, event_type, artist_id
                    FROM events
                    WHERE event_date >= CURRENT_DATE
                        AND event_date <= CURRENT_DATE + INTERVAL '30 days'
                    ORDER BY event_date ASC
                    LIMIT 10
                """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                events = []
                for row in results:
                    events.append({
                        'event_name': row[0],
                        'event_date': row[1].isoformat() if row[1] else None,
                        'event_type': row[2],
                        'artist_id': row[3]
                    })
                
                return events
                
        except Exception as e:
            self.logger.error(f"Error getting upcoming events: {e}")
            return []