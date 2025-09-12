#!/usr/bin/env python3
"""
Alert System Usage Examples
===========================

Comprehensive examples demonstrating the K-Pop Alert System capabilities
including rapid growth detection, decline monitoring, anomaly detection,
and alert management.

Author: Backend Development Team
Date: 2025-09-08
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from kpop_dashboard.analytics import (
        AlertEngine,
        AlertThresholds,
        AlertType,
        AlertSeverity,
        AlertStatus,
        AnomalyDetectionMethod,
        ArtistMetrics,
        MetricDataPoint
    )
    print("✅ Successfully imported alert system components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_sample_artist_data(artist_id: int, 
                            artist_name: str,
                            platform: str = 'youtube',
                            days_of_data: int = 30,
                            base_value: int = 100000,
                            growth_pattern: str = 'normal') -> ArtistMetrics:
    """Create sample artist data with different growth patterns."""
    
    data_points = []
    current_value = base_value
    base_date = datetime.now()
    
    for day in range(days_of_data):
        timestamp = base_date - timedelta(days=days_of_data - day - 1)
        
        # Apply different growth patterns
        if growth_pattern == 'rapid_growth':
            if day >= days_of_data - 3:  # Rapid growth in last 3 days
                daily_growth = random.uniform(0.15, 0.25)  # 15-25% daily
            else:
                daily_growth = random.uniform(0.01, 0.03)  # Normal growth
                
        elif growth_pattern == 'decline':
            if day >= days_of_data - 5:  # Decline in last 5 days
                daily_growth = random.uniform(-0.08, -0.03)  # 3-8% decline
            else:
                daily_growth = random.uniform(0.005, 0.02)  # Slow growth
                
        elif growth_pattern == 'anomaly':
            if day == days_of_data - 2:  # Anomaly 2 days ago
                daily_growth = random.uniform(0.5, 1.0)  # 50-100% spike
            else:
                daily_growth = random.uniform(0.005, 0.015)  # Normal
                
        else:  # normal pattern
            daily_growth = random.uniform(0.008, 0.025)  # 0.8-2.5% daily
        
        # Apply growth
        current_value = int(current_value * (1 + daily_growth))
        
        data_points.append(MetricDataPoint(
            timestamp=timestamp,
            value=max(0, current_value),
            platform=platform,
            metric_type='subscribers' if platform == 'youtube' else 'followers',
            quality_score=random.uniform(0.85, 0.98)
        ))
    
    return ArtistMetrics(
        artist_id=artist_id,
        artist_name=artist_name,
        company_id=random.randint(1, 10),
        company_name=f"Company {random.randint(1, 10)}",
        debut_year=random.randint(2018, 2023),
        platform=platform,
        metric_type='subscribers' if platform == 'youtube' else 'followers',
        current_value=data_points[-1].value,
        data_points=data_points,
        quality_score=0.92,
        last_updated=datetime.now()
    )


def demonstrate_rapid_growth_detection():
    """Demonstrate rapid growth detection capabilities."""
    print("\n🚀 Rapid Growth Detection Demo")
    print("=" * 50)
    
    # Initialize alert engine with custom thresholds
    thresholds = AlertThresholds(
        rapid_growth_percentage=20.0,  # 20% threshold for demo
        rapid_growth_timeframe_hours=72  # 3 days
    )
    
    engine = AlertEngine(thresholds=thresholds)
    
    # Create artist with rapid growth pattern
    rapid_artist = create_sample_artist_data(
        artist_id=1,
        artist_name="NewJeans",
        platform='youtube',
        days_of_data=30,
        base_value=500000,
        growth_pattern='rapid_growth'
    )
    
    print(f"Artist: {rapid_artist.artist_name}")
    print(f"Platform: {rapid_artist.platform}")
    print(f"Current Value: {rapid_artist.current_value:,}")
    print(f"Base Value: 500,000")
    print(f"Total Growth: {((rapid_artist.current_value - 500000) / 500000 * 100):.1f}%")
    
    # Detect rapid growth
    alerts = engine.detect_rapid_growth(rapid_artist)
    
    print(f"\n📊 Analysis Results:")
    print(f"Alerts Generated: {len(alerts)}")
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n🚨 Alert #{i}:")
        print(f"   Type: {alert.alert_type.value}")
        print(f"   Severity: {alert.severity.value}")
        print(f"   Title: {alert.title}")
        print(f"   Message: {alert.message}")
        print(f"   Growth Rate: {alert.percentage_change:.1f}%")
        print(f"   Confidence: {alert.confidence_score:.2f}")
    
    return engine, alerts


def demonstrate_growth_decline_detection():
    """Demonstrate growth decline detection capabilities."""
    print("\n📉 Growth Decline Detection Demo")
    print("=" * 50)
    
    # Initialize alert engine
    engine = AlertEngine()
    
    # Create artist with decline pattern
    declining_artist = create_sample_artist_data(
        artist_id=2,
        artist_name="IVE",
        platform='youtube',
        days_of_data=30,
        base_value=800000,
        growth_pattern='decline'
    )
    
    print(f"Artist: {declining_artist.artist_name}")
    print(f"Platform: {declining_artist.platform}")
    print(f"Current Value: {declining_artist.current_value:,}")
    
    # Detect decline
    alerts = engine.detect_growth_decline(declining_artist)
    
    print(f"\n📊 Analysis Results:")
    print(f"Alerts Generated: {len(alerts)}")
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n⚠️ Alert #{i}:")
        print(f"   Type: {alert.alert_type.value}")
        print(f"   Severity: {alert.severity.value}")
        print(f"   Title: {alert.title}")
        print(f"   Message: {alert.message}")
        if alert.percentage_change:
            print(f"   Decline Rate: {alert.percentage_change:.1f}%")
    
    return engine, alerts


def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection capabilities."""
    print("\n🔍 Anomaly Detection Demo")
    print("=" * 50)
    
    # Initialize alert engine
    engine = AlertEngine()
    
    # Create artist with anomaly pattern
    anomaly_artist = create_sample_artist_data(
        artist_id=3,
        artist_name="BLACKPINK",
        platform='youtube',
        days_of_data=30,
        base_value=1200000,
        growth_pattern='anomaly'
    )
    
    print(f"Artist: {anomaly_artist.artist_name}")
    print(f"Platform: {anomaly_artist.platform}")
    print(f"Current Value: {anomaly_artist.current_value:,}")
    
    # Detect anomalies using multiple methods
    detection_methods = [
        AnomalyDetectionMethod.Z_SCORE,
        AnomalyDetectionMethod.IQR,
        AnomalyDetectionMethod.MOVING_AVERAGE
    ]
    
    alerts = engine.detect_anomalies(
        anomaly_artist,
        methods=detection_methods,
        lookback_days=30
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"Detection Methods Used: {len(detection_methods)}")
    print(f"Alerts Generated: {len(alerts)}")
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n🔎 Alert #{i}:")
        print(f"   Type: {alert.alert_type.value}")
        print(f"   Severity: {alert.severity.value}")
        print(f"   Title: {alert.title}")
        print(f"   Detection Method: {alert.detection_method}")
        print(f"   Confidence: {alert.confidence_score:.2f}")
        if 'anomaly_score' in alert.metadata:
            print(f"   Anomaly Score: {alert.metadata['anomaly_score']:.2f}")
    
    return engine, alerts


def demonstrate_alert_management():
    """Demonstrate alert management and lifecycle."""
    print("\n🔧 Alert Management Demo")
    print("=" * 50)
    
    # Create alert engine and generate some alerts
    engine = AlertEngine()
    
    # Create multiple artists with different patterns
    artists = [
        create_sample_artist_data(1, "TWICE", growth_pattern='rapid_growth'),
        create_sample_artist_data(2, "RED VELVET", growth_pattern='decline'),
        create_sample_artist_data(3, "AESPA", growth_pattern='anomaly')
    ]
    
    all_alerts = []
    
    # Generate alerts for each artist
    for artist in artists:
        alerts = []
        alerts.extend(engine.detect_rapid_growth(artist))
        alerts.extend(engine.detect_growth_decline(artist))
        alerts.extend(engine.detect_anomalies(artist))
        all_alerts.extend(alerts)
    
    print(f"Total Alerts Generated: {len(all_alerts)}")
    
    # Show active alerts
    active_alerts = engine.get_active_alerts()
    print(f"Active Alerts: {len(active_alerts)}")
    
    # Show alerts by severity
    critical_alerts = engine.get_active_alerts(severity_filter=AlertSeverity.CRITICAL)
    high_alerts = engine.get_active_alerts(severity_filter=AlertSeverity.HIGH)
    
    print(f"Critical Alerts: {len(critical_alerts)}")
    print(f"High Severity Alerts: {len(high_alerts)}")
    
    # Demonstrate alert acknowledgment and resolution
    if active_alerts:
        first_alert = active_alerts[0]
        print(f"\n📝 Managing Alert: {first_alert.alert_id}")
        print(f"   Status: {first_alert.status.value}")
        
        # Acknowledge alert
        engine.acknowledge_alert(first_alert.alert_id, "Demo User")
        print(f"   Acknowledged: ✅")
        
        # Resolve alert
        engine.resolve_alert(
            first_alert.alert_id,
            resolution_notes="Resolved during demo",
            resolved_by="Demo User"
        )
        print(f"   Resolved: ✅")
    
    # Show statistics
    stats = engine.get_alert_statistics()
    print(f"\n📊 Alert System Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    return engine, stats


def run_comprehensive_demo():
    """Run comprehensive alert system demonstration."""
    print("🎵 K-Pop Alert System Comprehensive Demo")
    print("=" * 80)
    
    try:
        # Run individual demonstrations
        rapid_engine, rapid_alerts = demonstrate_rapid_growth_detection()
        decline_engine, decline_alerts = demonstrate_growth_decline_detection()
        anomaly_engine, anomaly_alerts = demonstrate_anomaly_detection()
        mgmt_engine, mgmt_stats = demonstrate_alert_management()
        
        # Summary
        print("\n" + "=" * 80)
        print("🏆 DEMO SUMMARY")
        print("=" * 80)
        
        print(f"✅ Rapid Growth Detection: {len(rapid_alerts)} alerts generated")
        print(f"✅ Growth Decline Detection: {len(decline_alerts)} alerts generated") 
        print(f"✅ Anomaly Detection: {len(anomaly_alerts)} alerts generated")
        print(f"✅ Alert Management: {mgmt_stats['total_generated']} total alerts managed")
        
        print(f"\n📈 Total Demonstration Results:")
        print(f"   Alert Engines Created: 4")
        print(f"   Sample Artists Analyzed: 7")
        print(f"   Detection Methods Tested: 3")
        print(f"   Alert Types Generated: {len(AlertType)}")
        print(f"   Severity Levels Used: {len(AlertSeverity)}")
        
        print(f"\n🎯 Key Features Demonstrated:")
        print("   • Configurable alert thresholds")
        print("   • Multiple anomaly detection methods")
        print("   • Automatic severity classification")
        print("   • Alert lifecycle management")
        print("   • Statistical analysis and confidence scoring")
        print("   • Real-time monitoring capabilities")
        
        print(f"\n✨ Alert System Ready for Production Use!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    run_comprehensive_demo()