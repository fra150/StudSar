"""Memory Analytics Module for StudSar V4

Provides comprehensive analytics for memory formation, retrieval patterns,
and cognitive insights to optimize memory system performance.
"""

import json
import logging
import math
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats as scipy_stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
except ImportError:
    scipy_stats = None
    dendrogram = None
    linkage = None
    pdist = None

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except ImportError:
    KMeans = None
    DBSCAN = None
    PCA = None
    silhouette_score = None

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Types of memory analytics"""
    FORMATION_PATTERNS = "formation_patterns"  # How memories are formed
    RETRIEVAL_PATTERNS = "retrieval_patterns"  # How memories are accessed
    TEMPORAL_ANALYSIS = "temporal_analysis"  # Time-based patterns
    SEMANTIC_CLUSTERING = "semantic_clustering"  # Content-based grouping
    USAGE_STATISTICS = "usage_statistics"  # General usage metrics
    PERFORMANCE_METRICS = "performance_metrics"  # System performance
    COGNITIVE_INSIGHTS = "cognitive_insights"  # Cognitive behavior analysis
    PREDICTIVE_ANALYTICS = "predictive_analytics"  # Future trend prediction

class MetricType(Enum):
    """Types of metrics to track"""
    COUNT = "count"  # Simple counts
    FREQUENCY = "frequency"  # Frequency distributions
    DURATION = "duration"  # Time durations
    CORRELATION = "correlation"  # Correlations between variables
    TREND = "trend"  # Trend analysis
    DISTRIBUTION = "distribution"  # Statistical distributions
    CLUSTERING = "clustering"  # Cluster analysis
    CLASSIFICATION = "classification"  # Classification metrics

class TimeGranularity(Enum):
    """Time granularity for analytics"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

class AnalyticsScope(Enum):
    """Scope of analytics"""
    GLOBAL = "global"  # Across all memories
    USER = "user"  # Per user
    SESSION = "session"  # Per session
    CATEGORY = "category"  # Per memory category
    TEMPORAL = "temporal"  # Time-based scope

@dataclass
class MemoryEvent:
    """Represents a memory-related event for analytics"""
    event_id: str
    event_type: str  # 'create', 'retrieve', 'update', 'delete'
    memory_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'memory_id': self.memory_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'success': self.success,
            'metadata': self.metadata,
            'context': self.context
        }

@dataclass
class AnalyticsQuery:
    """Query for analytics data"""
    query_id: str
    analytics_type: AnalyticsType
    scope: AnalyticsScope
    time_range: Tuple[datetime, datetime]
    granularity: TimeGranularity = TimeGranularity.DAY
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregations: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalyticsResult:
    """Result of analytics computation"""
    query_id: str
    analytics_type: AnalyticsType
    data: Dict[str, Any]
    metrics: Dict[str, float]
    insights: List[str]
    visualizations: Dict[str, Any] = field(default_factory=dict)
    computation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query_id': self.query_id,
            'analytics_type': self.analytics_type.value,
            'data': self.data,
            'metrics': self.metrics,
            'insights': self.insights,
            'visualizations': self.visualizations,
            'computation_time_ms': self.computation_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class CognitiveInsight:
    """Represents a cognitive insight derived from analytics"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAnalyzer(ABC):
    """Base class for analytics analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        
    @abstractmethod
    def analyze(self, events: List[MemoryEvent], 
               query: AnalyticsQuery) -> AnalyticsResult:
        """Perform analysis on memory events"""
        pass
    
    def filter_events(self, events: List[MemoryEvent], 
                     query: AnalyticsQuery) -> List[MemoryEvent]:
        """Filter events based on query criteria"""
        filtered_events = []
        
        start_time, end_time = query.time_range
        
        for event in events:
            # Time range filter
            if not (start_time <= event.timestamp <= end_time):
                continue
            
            # Apply additional filters
            include_event = True
            for filter_key, filter_value in query.filters.items():
                if filter_key == 'event_type' and event.event_type != filter_value:
                    include_event = False
                    break
                elif filter_key == 'user_id' and event.user_id != filter_value:
                    include_event = False
                    break
                elif filter_key == 'session_id' and event.session_id != filter_value:
                    include_event = False
                    break
                elif filter_key == 'success' and event.success != filter_value:
                    include_event = False
                    break
            
            if include_event:
                filtered_events.append(event)
        
        return filtered_events

class FormationPatternsAnalyzer(BaseAnalyzer):
    """Analyzes memory formation patterns"""
    
    def __init__(self):
        super().__init__("FormationPatternsAnalyzer")
        
    def analyze(self, events: List[MemoryEvent], 
               query: AnalyticsQuery) -> AnalyticsResult:
        """Analyze memory formation patterns"""
        
        start_time = time.time()
        
        # Filter for creation events
        creation_events = [
            event for event in self.filter_events(events, query)
            if event.event_type == 'create'
        ]
        
        if not creation_events:
            return AnalyticsResult(
                query_id=query.query_id,
                analytics_type=AnalyticsType.FORMATION_PATTERNS,
                data={},
                metrics={},
                insights=["No memory creation events found in the specified time range"]
            )
        
        # Analyze patterns
        data = {}
        metrics = {}
        insights = []
        
        # Time-based patterns
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for event in creation_events:
            hour = event.timestamp.hour
            day = event.timestamp.strftime('%Y-%m-%d')
            hourly_counts[hour] += 1
            daily_counts[day] += 1
        
        data['hourly_distribution'] = dict(hourly_counts)
        data['daily_distribution'] = dict(daily_counts)
        
        # Peak hours
        if hourly_counts:
            peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
            metrics['peak_hour'] = peak_hour[0]
            metrics['peak_hour_count'] = peak_hour[1]
            insights.append(f"Peak memory formation occurs at hour {peak_hour[0]} with {peak_hour[1]} memories created")
        
        # Formation rate
        if len(creation_events) > 1:
            time_span = (creation_events[-1].timestamp - creation_events[0].timestamp).total_seconds()
            if time_span > 0:
                formation_rate = len(creation_events) / (time_span / 3600)  # per hour
                metrics['formation_rate_per_hour'] = formation_rate
                insights.append(f"Average memory formation rate: {formation_rate:.2f} memories per hour")
        
        # User patterns
        user_counts = defaultdict(int)
        for event in creation_events:
            if event.user_id:
                user_counts[event.user_id] += 1
        
        if user_counts:
            data['user_distribution'] = dict(user_counts)
            most_active_user = max(user_counts.items(), key=lambda x: x[1])
            metrics['most_active_user'] = most_active_user[0]
            metrics['most_active_user_count'] = most_active_user[1]
        
        # Session patterns
        session_counts = defaultdict(int)
        for event in creation_events:
            if event.session_id:
                session_counts[event.session_id] += 1
        
        if session_counts:
            data['session_distribution'] = dict(session_counts)
            avg_memories_per_session = statistics.mean(session_counts.values())
            metrics['avg_memories_per_session'] = avg_memories_per_session
            insights.append(f"Average memories created per session: {avg_memories_per_session:.2f}")
        
        # Success rate
        successful_creations = sum(1 for event in creation_events if event.success)
        success_rate = successful_creations / len(creation_events) if creation_events else 0
        metrics['success_rate'] = success_rate
        
        if success_rate < 0.95:
            insights.append(f"Memory creation success rate is {success_rate:.1%}, consider investigating failures")
        
        computation_time = (time.time() - start_time) * 1000
        
        return AnalyticsResult(
            query_id=query.query_id,
            analytics_type=AnalyticsType.FORMATION_PATTERNS,
            data=data,
            metrics=metrics,
            insights=insights,
            computation_time_ms=computation_time
        )

class RetrievalPatternsAnalyzer(BaseAnalyzer):
    """Analyzes memory retrieval patterns"""
    
    def __init__(self):
        super().__init__("RetrievalPatternsAnalyzer")
        
    def analyze(self, events: List[MemoryEvent], 
               query: AnalyticsQuery) -> AnalyticsResult:
        """Analyze memory retrieval patterns"""
        
        start_time = time.time()
        
        # Filter for retrieval events
        retrieval_events = [
            event for event in self.filter_events(events, query)
            if event.event_type == 'retrieve'
        ]
        
        if not retrieval_events:
            return AnalyticsResult(
                query_id=query.query_id,
                analytics_type=AnalyticsType.RETRIEVAL_PATTERNS,
                data={},
                metrics={},
                insights=["No memory retrieval events found in the specified time range"]
            )
        
        data = {}
        metrics = {}
        insights = []
        
        # Memory access frequency
        memory_access_counts = defaultdict(int)
        for event in retrieval_events:
            memory_access_counts[event.memory_id] += 1
        
        if memory_access_counts:
            data['memory_access_frequency'] = dict(memory_access_counts)
            
            # Most accessed memories
            most_accessed = sorted(memory_access_counts.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
            data['most_accessed_memories'] = most_accessed
            
            # Access distribution statistics
            access_counts = list(memory_access_counts.values())
            metrics['avg_accesses_per_memory'] = statistics.mean(access_counts)
            metrics['median_accesses_per_memory'] = statistics.median(access_counts)
            metrics['max_accesses'] = max(access_counts)
            metrics['unique_memories_accessed'] = len(memory_access_counts)
            
            # Identify hot memories (top 20%)
            threshold = np.percentile(access_counts, 80) if np else sorted(access_counts)[int(0.8 * len(access_counts))]
            hot_memories = [mid for mid, count in memory_access_counts.items() if count >= threshold]
            data['hot_memories'] = hot_memories
            metrics['hot_memories_count'] = len(hot_memories)
            
            insights.append(f"Top 20% of memories account for {len(hot_memories)} frequently accessed items")
        
        # Retrieval performance
        durations = [event.duration_ms for event in retrieval_events if event.duration_ms is not None]
        if durations:
            metrics['avg_retrieval_time_ms'] = statistics.mean(durations)
            metrics['median_retrieval_time_ms'] = statistics.median(durations)
            metrics['p95_retrieval_time_ms'] = np.percentile(durations, 95) if np else sorted(durations)[int(0.95 * len(durations))]
            
            slow_retrievals = [d for d in durations if d > 1000]  # > 1 second
            if slow_retrievals:
                metrics['slow_retrieval_percentage'] = len(slow_retrievals) / len(durations) * 100
                insights.append(f"{len(slow_retrievals)} retrievals took longer than 1 second")
        
        # Temporal patterns
        hourly_retrievals = defaultdict(int)
        for event in retrieval_events:
            hour = event.timestamp.hour
            hourly_retrievals[hour] += 1
        
        if hourly_retrievals:
            data['hourly_retrieval_pattern'] = dict(hourly_retrievals)
            peak_retrieval_hour = max(hourly_retrievals.items(), key=lambda x: x[1])
            metrics['peak_retrieval_hour'] = peak_retrieval_hour[0]
            insights.append(f"Peak retrieval activity occurs at hour {peak_retrieval_hour[0]}")
        
        # Success rate
        successful_retrievals = sum(1 for event in retrieval_events if event.success)
        success_rate = successful_retrievals / len(retrieval_events) if retrieval_events else 0
        metrics['retrieval_success_rate'] = success_rate
        
        if success_rate < 0.98:
            insights.append(f"Retrieval success rate is {success_rate:.1%}, investigate failed retrievals")
        
        # User behavior
        user_retrieval_counts = defaultdict(int)
        for event in retrieval_events:
            if event.user_id:
                user_retrieval_counts[event.user_id] += 1
        
        if user_retrieval_counts:
            data['user_retrieval_distribution'] = dict(user_retrieval_counts)
            avg_retrievals_per_user = statistics.mean(user_retrieval_counts.values())
            metrics['avg_retrievals_per_user'] = avg_retrievals_per_user
        
        computation_time = (time.time() - start_time) * 1000
        
        return AnalyticsResult(
            query_id=query.query_id,
            analytics_type=AnalyticsType.RETRIEVAL_PATTERNS,
            data=data,
            metrics=metrics,
            insights=insights,
            computation_time_ms=computation_time
        )

class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes temporal patterns in memory usage"""
    
    def __init__(self):
        super().__init__("TemporalAnalyzer")
        
    def analyze(self, events: List[MemoryEvent], 
               query: AnalyticsQuery) -> AnalyticsResult:
        """Analyze temporal patterns"""
        
        start_time = time.time()
        
        filtered_events = self.filter_events(events, query)
        
        if not filtered_events:
            return AnalyticsResult(
                query_id=query.query_id,
                analytics_type=AnalyticsType.TEMPORAL_ANALYSIS,
                data={},
                metrics={},
                insights=["No events found in the specified time range"]
            )
        
        data = {}
        metrics = {}
        insights = []
        
        # Time series analysis
        time_series = self._create_time_series(filtered_events, query.granularity)
        data['time_series'] = time_series
        
        # Trend analysis
        if len(time_series) > 2:
            trend = self._calculate_trend(time_series)
            metrics['trend_slope'] = trend
            
            if trend > 0.1:
                insights.append("Memory activity is showing an increasing trend")
            elif trend < -0.1:
                insights.append("Memory activity is showing a decreasing trend")
            else:
                insights.append("Memory activity is relatively stable")
        
        # Seasonality detection
        seasonality = self._detect_seasonality(time_series, query.granularity)
        if seasonality:
            data['seasonality'] = seasonality
            insights.append(f"Detected seasonal patterns in memory activity")
        
        # Peak detection
        peaks = self._detect_peaks(time_series)
        if peaks:
            data['peaks'] = peaks
            metrics['peak_count'] = len(peaks)
            insights.append(f"Identified {len(peaks)} activity peaks")
        
        # Activity distribution by time periods
        activity_by_period = self._analyze_activity_by_period(filtered_events)
        data['activity_by_period'] = activity_by_period
        
        # Busiest periods
        if activity_by_period:
            busiest_hour = max(activity_by_period.get('hourly', {}).items(), 
                             key=lambda x: x[1], default=(None, 0))
            busiest_day = max(activity_by_period.get('daily', {}).items(), 
                            key=lambda x: x[1], default=(None, 0))
            
            if busiest_hour[0] is not None:
                metrics['busiest_hour'] = busiest_hour[0]
                insights.append(f"Busiest hour: {busiest_hour[0]}:00 with {busiest_hour[1]} events")
            
            if busiest_day[0] is not None:
                metrics['busiest_day_of_week'] = busiest_day[0]
                insights.append(f"Busiest day of week: {busiest_day[0]} with {busiest_day[1]} events")
        
        # Event type temporal distribution
        event_type_temporal = defaultdict(lambda: defaultdict(int))
        for event in filtered_events:
            time_key = self._get_time_key(event.timestamp, query.granularity)
            event_type_temporal[event.event_type][time_key] += 1
        
        data['event_type_temporal_distribution'] = {
            event_type: dict(temporal_data) 
            for event_type, temporal_data in event_type_temporal.items()
        }
        
        computation_time = (time.time() - start_time) * 1000
        
        return AnalyticsResult(
            query_id=query.query_id,
            analytics_type=AnalyticsType.TEMPORAL_ANALYSIS,
            data=data,
            metrics=metrics,
            insights=insights,
            computation_time_ms=computation_time
        )
    
    def _create_time_series(self, events: List[MemoryEvent], 
                           granularity: TimeGranularity) -> Dict[str, int]:
        """Create time series data"""
        time_series = defaultdict(int)
        
        for event in events:
            time_key = self._get_time_key(event.timestamp, granularity)
            time_series[time_key] += 1
        
        return dict(time_series)
    
    def _get_time_key(self, timestamp: datetime, 
                     granularity: TimeGranularity) -> str:
        """Get time key based on granularity"""
        if granularity == TimeGranularity.MINUTE:
            return timestamp.strftime('%Y-%m-%d %H:%M')
        elif granularity == TimeGranularity.HOUR:
            return timestamp.strftime('%Y-%m-%d %H:00')
        elif granularity == TimeGranularity.DAY:
            return timestamp.strftime('%Y-%m-%d')
        elif granularity == TimeGranularity.WEEK:
            # Get Monday of the week
            monday = timestamp - timedelta(days=timestamp.weekday())
            return monday.strftime('%Y-%m-%d')
        elif granularity == TimeGranularity.MONTH:
            return timestamp.strftime('%Y-%m')
        elif granularity == TimeGranularity.YEAR:
            return timestamp.strftime('%Y')
        else:
            return timestamp.strftime('%Y-%m-%d')
    
    def _calculate_trend(self, time_series: Dict[str, int]) -> float:
        """Calculate trend slope"""
        if len(time_series) < 2:
            return 0.0
        
        # Simple linear regression
        x_values = list(range(len(time_series)))
        y_values = list(time_series.values())
        
        if np:
            slope, _ = np.polyfit(x_values, y_values, 1)
            return float(slope)
        else:
            # Manual calculation
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
    
    def _detect_seasonality(self, time_series: Dict[str, int], 
                           granularity: TimeGranularity) -> Optional[Dict[str, Any]]:
        """Detect seasonal patterns"""
        if len(time_series) < 7:  # Need at least a week of data
            return None
        
        values = list(time_series.values())
        
        # Simple seasonality detection based on autocorrelation
        if granularity == TimeGranularity.HOUR and len(values) >= 24:
            # Daily seasonality
            daily_pattern = self._calculate_autocorrelation(values, 24)
            if daily_pattern > 0.3:  # Threshold for significant correlation
                return {'type': 'daily', 'strength': daily_pattern}
        
        elif granularity == TimeGranularity.DAY and len(values) >= 7:
            # Weekly seasonality
            weekly_pattern = self._calculate_autocorrelation(values, 7)
            if weekly_pattern > 0.3:
                return {'type': 'weekly', 'strength': weekly_pattern}
        
        return None
    
    def _calculate_autocorrelation(self, values: List[int], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        if len(values) <= lag:
            return 0.0
        
        # Simple autocorrelation calculation
        mean_val = statistics.mean(values)
        
        numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) 
                       for i in range(len(values) - lag))
        
        denominator = sum((val - mean_val) ** 2 for val in values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _detect_peaks(self, time_series: Dict[str, int]) -> List[Tuple[str, int]]:
        """Detect peaks in time series"""
        if len(time_series) < 3:
            return []
        
        values = list(time_series.values())
        timestamps = list(time_series.keys())
        
        # Simple peak detection: value higher than neighbors
        peaks = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                # Additional check: peak should be significantly higher
                if values[i] > statistics.mean(values) + statistics.stdev(values) if len(values) > 1 else 0:
                    peaks.append((timestamps[i], values[i]))
        
        return peaks
    
    def _analyze_activity_by_period(self, events: List[MemoryEvent]) -> Dict[str, Dict[str, int]]:
        """Analyze activity by different time periods"""
        hourly = defaultdict(int)
        daily = defaultdict(int)
        monthly = defaultdict(int)
        
        for event in events:
            hour = event.timestamp.hour
            day_of_week = event.timestamp.strftime('%A')
            month = event.timestamp.strftime('%B')
            
            hourly[hour] += 1
            daily[day_of_week] += 1
            monthly[month] += 1
        
        return {
            'hourly': dict(hourly),
            'daily': dict(daily),
            'monthly': dict(monthly)
        }

class UsageStatisticsAnalyzer(BaseAnalyzer):
    """Analyzes general usage statistics"""
    
    def __init__(self):
        super().__init__("UsageStatisticsAnalyzer")
        
    def analyze(self, events: List[MemoryEvent], 
               query: AnalyticsQuery) -> AnalyticsResult:
        """Analyze usage statistics"""
        
        start_time = time.time()
        
        filtered_events = self.filter_events(events, query)
        
        data = {}
        metrics = {}
        insights = []
        
        if not filtered_events:
            insights.append("No events found in the specified time range")
        else:
            # Basic counts
            metrics['total_events'] = len(filtered_events)
            
            # Event type distribution
            event_type_counts = Counter(event.event_type for event in filtered_events)
            data['event_type_distribution'] = dict(event_type_counts)
            
            # Most common event type
            if event_type_counts:
                most_common_event = event_type_counts.most_common(1)[0]
                metrics['most_common_event_type'] = most_common_event[0]
                metrics['most_common_event_count'] = most_common_event[1]
                insights.append(f"Most common event type: {most_common_event[0]} ({most_common_event[1]} occurrences)")
            
            # User activity
            user_counts = Counter(event.user_id for event in filtered_events if event.user_id)
            if user_counts:
                data['user_activity'] = dict(user_counts)
                metrics['unique_users'] = len(user_counts)
                metrics['avg_events_per_user'] = statistics.mean(user_counts.values())
                
                most_active_user = user_counts.most_common(1)[0]
                metrics['most_active_user'] = most_active_user[0]
                insights.append(f"Most active user: {most_active_user[0]} with {most_active_user[1]} events")
            
            # Session activity
            session_counts = Counter(event.session_id for event in filtered_events if event.session_id)
            if session_counts:
                data['session_activity'] = dict(session_counts)
                metrics['unique_sessions'] = len(session_counts)
                metrics['avg_events_per_session'] = statistics.mean(session_counts.values())
            
            # Success rates
            successful_events = sum(1 for event in filtered_events if event.success)
            success_rate = successful_events / len(filtered_events)
            metrics['overall_success_rate'] = success_rate
            
            if success_rate < 0.95:
                insights.append(f"Overall success rate is {success_rate:.1%}, investigate failures")
            
            # Performance metrics
            durations = [event.duration_ms for event in filtered_events if event.duration_ms is not None]
            if durations:
                metrics['avg_duration_ms'] = statistics.mean(durations)
                metrics['median_duration_ms'] = statistics.median(durations)
                metrics['p95_duration_ms'] = np.percentile(durations, 95) if np else sorted(durations)[int(0.95 * len(durations))]
                
                slow_events = [d for d in durations if d > 1000]
                if slow_events:
                    metrics['slow_events_percentage'] = len(slow_events) / len(durations) * 100
                    insights.append(f"{len(slow_events)} events took longer than 1 second")
            
            # Memory usage patterns
            unique_memories = len(set(event.memory_id for event in filtered_events))
            metrics['unique_memories_involved'] = unique_memories
            
            if unique_memories > 0:
                metrics['avg_events_per_memory'] = len(filtered_events) / unique_memories
            
            # Time span analysis
            if len(filtered_events) > 1:
                time_span = (filtered_events[-1].timestamp - filtered_events[0].timestamp).total_seconds()
                metrics['time_span_hours'] = time_span / 3600
                metrics['events_per_hour'] = len(filtered_events) / (time_span / 3600) if time_span > 0 else 0
        
        computation_time = (time.time() - start_time) * 1000
        
        return AnalyticsResult(
            query_id=query.query_id,
            analytics_type=AnalyticsType.USAGE_STATISTICS,
            data=data,
            metrics=metrics,
            insights=insights,
            computation_time_ms=computation_time
        )

class CognitiveInsightsAnalyzer(BaseAnalyzer):
    """Analyzes cognitive patterns and generates insights"""
    
    def __init__(self):
        super().__init__("CognitiveInsightsAnalyzer")
        
    def analyze(self, events: List[MemoryEvent], 
               query: AnalyticsQuery) -> AnalyticsResult:
        """Analyze cognitive patterns"""
        
        start_time = time.time()
        
        filtered_events = self.filter_events(events, query)
        
        data = {}
        metrics = {}
        insights = []
        
        if not filtered_events:
            insights.append("No events found for cognitive analysis")
        else:
            # Memory formation vs retrieval ratio
            creation_events = [e for e in filtered_events if e.event_type == 'create']
            retrieval_events = [e for e in filtered_events if e.event_type == 'retrieve']
            
            if creation_events and retrieval_events:
                formation_retrieval_ratio = len(creation_events) / len(retrieval_events)
                metrics['formation_retrieval_ratio'] = formation_retrieval_ratio
                
                if formation_retrieval_ratio > 1.0:
                    insights.append("High memory formation activity - learning phase detected")
                elif formation_retrieval_ratio < 0.1:
                    insights.append("High retrieval activity - knowledge application phase detected")
                else:
                    insights.append("Balanced formation and retrieval activity")
            
            # Memory consolidation patterns
            consolidation_insights = self._analyze_consolidation_patterns(filtered_events)
            if consolidation_insights:
                data['consolidation_patterns'] = consolidation_insights
                insights.extend(consolidation_insights.get('insights', []))
            
            # Attention patterns
            attention_insights = self._analyze_attention_patterns(filtered_events)
            if attention_insights:
                data['attention_patterns'] = attention_insights
                insights.extend(attention_insights.get('insights', []))
            
            # Learning efficiency
            efficiency_metrics = self._calculate_learning_efficiency(filtered_events)
            if efficiency_metrics:
                metrics.update(efficiency_metrics)
                
                if efficiency_metrics.get('learning_efficiency', 0) > 0.8:
                    insights.append("High learning efficiency detected")
                elif efficiency_metrics.get('learning_efficiency', 0) < 0.4:
                    insights.append("Low learning efficiency - consider optimization")
            
            # Memory interference detection
            interference_analysis = self._detect_memory_interference(filtered_events)
            if interference_analysis:
                data['interference_analysis'] = interference_analysis
                if interference_analysis.get('interference_score', 0) > 0.5:
                    insights.append("Potential memory interference detected")
            
            # Cognitive load assessment
            cognitive_load = self._assess_cognitive_load(filtered_events)
            if cognitive_load:
                metrics['cognitive_load_score'] = cognitive_load
                
                if cognitive_load > 0.8:
                    insights.append("High cognitive load detected - consider reducing complexity")
                elif cognitive_load < 0.3:
                    insights.append("Low cognitive load - capacity for more complex tasks")
        
        computation_time = (time.time() - start_time) * 1000
        
        return AnalyticsResult(
            query_id=query.query_id,
            analytics_type=AnalyticsType.COGNITIVE_INSIGHTS,
            data=data,
            metrics=metrics,
            insights=insights,
            computation_time_ms=computation_time
        )
    
    def _analyze_consolidation_patterns(self, events: List[MemoryEvent]) -> Dict[str, Any]:
        """Analyze memory consolidation patterns"""
        # Group events by memory ID
        memory_events = defaultdict(list)
        for event in events:
            memory_events[event.memory_id].append(event)
        
        consolidation_data = {
            'memories_with_multiple_accesses': 0,
            'avg_time_between_accesses': 0,
            'consolidation_score': 0,
            'insights': []
        }
        
        multiple_access_memories = []
        time_intervals = []
        
        for memory_id, mem_events in memory_events.items():
            if len(mem_events) > 1:
                multiple_access_memories.append(memory_id)
                
                # Calculate time intervals between accesses
                sorted_events = sorted(mem_events, key=lambda x: x.timestamp)
                for i in range(1, len(sorted_events)):
                    interval = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds()
                    time_intervals.append(interval)
        
        consolidation_data['memories_with_multiple_accesses'] = len(multiple_access_memories)
        
        if time_intervals:
            avg_interval = statistics.mean(time_intervals)
            consolidation_data['avg_time_between_accesses'] = avg_interval
            
            # Consolidation score based on optimal spacing
            # Optimal spacing is around 1-7 days (86400-604800 seconds)
            optimal_intervals = [i for i in time_intervals if 86400 <= i <= 604800]
            consolidation_score = len(optimal_intervals) / len(time_intervals)
            consolidation_data['consolidation_score'] = consolidation_score
            
            if consolidation_score > 0.6:
                consolidation_data['insights'].append("Good memory consolidation patterns detected")
            else:
                consolidation_data['insights'].append("Memory consolidation could be improved with better spacing")
        
        return consolidation_data
    
    def _analyze_attention_patterns(self, events: List[MemoryEvent]) -> Dict[str, Any]:
        """Analyze attention and focus patterns"""
        attention_data = {
            'focus_sessions': [],
            'attention_switches': 0,
            'avg_focus_duration': 0,
            'insights': []
        }
        
        # Group events by session
        session_events = defaultdict(list)
        for event in events:
            if event.session_id:
                session_events[event.session_id].append(event)
        
        focus_durations = []
        attention_switches = 0
        
        for session_id, sess_events in session_events.items():
            if len(sess_events) > 1:
                sorted_events = sorted(sess_events, key=lambda x: x.timestamp)
                
                # Calculate focus duration
                session_duration = (sorted_events[-1].timestamp - sorted_events[0].timestamp).total_seconds()
                focus_durations.append(session_duration)
                
                # Count attention switches (changes in memory being accessed)
                prev_memory = None
                for event in sorted_events:
                    if prev_memory and event.memory_id != prev_memory:
                        attention_switches += 1
                    prev_memory = event.memory_id
        
        attention_data['attention_switches'] = attention_switches
        
        if focus_durations:
            avg_focus = statistics.mean(focus_durations)
            attention_data['avg_focus_duration'] = avg_focus
            
            if avg_focus > 1800:  # 30 minutes
                attention_data['insights'].append("Good sustained attention patterns")
            elif avg_focus < 300:  # 5 minutes
                attention_data['insights'].append("Short attention spans detected - consider focus training")
        
        return attention_data
    
    def _calculate_learning_efficiency(self, events: List[MemoryEvent]) -> Dict[str, float]:
        """Calculate learning efficiency metrics"""
        creation_events = [e for e in events if e.event_type == 'create']
        retrieval_events = [e for e in events if e.event_type == 'retrieve']
        
        if not creation_events:
            return {}
        
        # Success rate
        successful_creations = sum(1 for e in creation_events if e.success)
        creation_success_rate = successful_creations / len(creation_events)
        
        # Retrieval efficiency
        retrieval_success_rate = 1.0
        if retrieval_events:
            successful_retrievals = sum(1 for e in retrieval_events if e.success)
            retrieval_success_rate = successful_retrievals / len(retrieval_events)
        
        # Time efficiency
        creation_durations = [e.duration_ms for e in creation_events if e.duration_ms is not None]
        time_efficiency = 1.0
        if creation_durations:
            avg_creation_time = statistics.mean(creation_durations)
            # Normalize to 0-1 scale (assuming 5 seconds is optimal)
            time_efficiency = max(0, 1 - (avg_creation_time - 5000) / 10000)
        
        # Overall learning efficiency
        learning_efficiency = (creation_success_rate + retrieval_success_rate + time_efficiency) / 3
        
        return {
            'creation_success_rate': creation_success_rate,
            'retrieval_success_rate': retrieval_success_rate,
            'time_efficiency': time_efficiency,
            'learning_efficiency': learning_efficiency
        }
    
    def _detect_memory_interference(self, events: List[MemoryEvent]) -> Dict[str, Any]:
        """Detect potential memory interference"""
        # Group events by time windows
        time_windows = defaultdict(list)
        
        for event in events:
            # 1-hour time windows
            window_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            time_windows[window_key].append(event)
        
        interference_scores = []
        
        for window, window_events in time_windows.items():
            if len(window_events) > 1:
                # Calculate interference based on memory switching
                unique_memories = len(set(e.memory_id for e in window_events))
                total_events = len(window_events)
                
                # High switching indicates potential interference
                switching_ratio = unique_memories / total_events
                interference_scores.append(switching_ratio)
        
        if interference_scores:
            avg_interference = statistics.mean(interference_scores)
            return {
                'interference_score': avg_interference,
                'high_interference_windows': len([s for s in interference_scores if s > 0.7])
            }
        
        return {}
    
    def _assess_cognitive_load(self, events: List[MemoryEvent]) -> float:
        """Assess cognitive load based on event patterns"""
        if not events:
            return 0.0
        
        # Factors contributing to cognitive load
        load_factors = []
        
        # Event frequency (events per minute)
        if len(events) > 1:
            time_span = (events[-1].timestamp - events[0].timestamp).total_seconds() / 60
            if time_span > 0:
                event_frequency = len(events) / time_span
                # Normalize to 0-1 scale (10 events per minute = high load)
                frequency_load = min(1.0, event_frequency / 10)
                load_factors.append(frequency_load)
        
        # Memory switching frequency
        memory_switches = 0
        prev_memory = None
        for event in sorted(events, key=lambda x: x.timestamp):
            if prev_memory and event.memory_id != prev_memory:
                memory_switches += 1
            prev_memory = event.memory_id
        
        if len(events) > 1:
            switch_rate = memory_switches / len(events)
            load_factors.append(switch_rate)
        
        # Error rate (failed operations)
        failed_events = sum(1 for e in events if not e.success)
        error_rate = failed_events / len(events)
        load_factors.append(error_rate)
        
        # Average cognitive load
        if load_factors:
            return statistics.mean(load_factors)
        
        return 0.0

class MemoryAnalyticsManager:
    """Main manager for memory analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.events: List[MemoryEvent] = []
        self.analyzers: Dict[AnalyticsType, BaseAnalyzer] = {
            AnalyticsType.FORMATION_PATTERNS: FormationPatternsAnalyzer(),
            AnalyticsType.RETRIEVAL_PATTERNS: RetrievalPatternsAnalyzer(),
            AnalyticsType.TEMPORAL_ANALYSIS: TemporalAnalyzer(),
            AnalyticsType.USAGE_STATISTICS: UsageStatisticsAnalyzer(),
            AnalyticsType.COGNITIVE_INSIGHTS: CognitiveInsightsAnalyzer()
        }
        
        # Results cache
        self.results_cache: Dict[str, AnalyticsResult] = {}
        self.cache_ttl = config.get('cache_ttl_hours', 1)
        
        # Background processing
        self.background_processing = config.get('background_processing', True)
        self.executor = ThreadPoolExecutor(max_workers=2) if self.background_processing else None
        
        # Event storage limits
        self.max_events = config.get('max_events', 100000)
        self.cleanup_threshold = config.get('cleanup_threshold', 0.8)
        
        # Statistics
        self.queries_processed = 0
        self.insights_generated = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Initialized Memory Analytics Manager")
    
    def record_event(self, event: MemoryEvent):
        """Record a memory event for analytics"""
        with self.lock:
            self.events.append(event)
            
            # Cleanup old events if needed
            if len(self.events) > self.max_events * self.cleanup_threshold:
                self._cleanup_old_events()
        
        logger.debug(f"Recorded event: {event.event_type} for memory {event.memory_id}")
    
    def create_memory_event(self, event_type: str, memory_id: str,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           duration_ms: Optional[float] = None,
                           success: bool = True,
                           metadata: Optional[Dict[str, Any]] = None,
                           context: Optional[Dict[str, Any]] = None) -> MemoryEvent:
        """Create and record a memory event"""
        
        event = MemoryEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {},
            context=context or {}
        )
        
        self.record_event(event)
        return event
    
    def run_analytics(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Run analytics query"""
        
        # Check cache first
        cache_key = self._generate_cache_key(query)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Returning cached result for query {query.query_id}")
            return cached_result
        
        # Get appropriate analyzer
        analyzer = self.analyzers.get(query.analytics_type)
        if not analyzer:
            raise ValueError(f"No analyzer available for {query.analytics_type}")
        
        # Run analysis
        with self.lock:
            events_copy = self.events.copy()
        
        result = analyzer.analyze(events_copy, query)
        
        # Cache result
        self._cache_result(cache_key, result)
        
        # Update statistics
        self.queries_processed += 1
        self.insights_generated += len(result.insights)
        
        logger.info(f"Completed analytics query {query.query_id} "
                   f"({query.analytics_type.value}) in {result.computation_time_ms:.2f}ms")
        
        return result
    
    def run_analytics_async(self, query: AnalyticsQuery) -> str:
        """Run analytics query asynchronously"""
        
        if not self.executor:
            raise RuntimeError("Background processing not enabled")
        
        future = self.executor.submit(self.run_analytics, query)
        
        # Store future with query ID for later retrieval
        # In a real implementation, you'd want a proper job management system
        
        return query.query_id
    
    def get_formation_patterns(self, time_range: Tuple[datetime, datetime],
                              user_id: Optional[str] = None,
                              granularity: TimeGranularity = TimeGranularity.DAY) -> AnalyticsResult:
        """Convenience method for formation patterns analysis"""
        
        query = AnalyticsQuery(
            query_id=str(uuid.uuid4()),
            analytics_type=AnalyticsType.FORMATION_PATTERNS,
            scope=AnalyticsScope.USER if user_id else AnalyticsScope.GLOBAL,
            time_range=time_range,
            granularity=granularity,
            filters={'user_id': user_id} if user_id else {}
        )
        
        return self.run_analytics(query)
    
    def get_retrieval_patterns(self, time_range: Tuple[datetime, datetime],
                              user_id: Optional[str] = None,
                              granularity: TimeGranularity = TimeGranularity.DAY) -> AnalyticsResult:
        """Convenience method for retrieval patterns analysis"""
        
        query = AnalyticsQuery(
            query_id=str(uuid.uuid4()),
            analytics_type=AnalyticsType.RETRIEVAL_PATTERNS,
            scope=AnalyticsScope.USER if user_id else AnalyticsScope.GLOBAL,
            time_range=time_range,
            granularity=granularity,
            filters={'user_id': user_id} if user_id else {}
        )
        
        return self.run_analytics(query)
    
    def get_temporal_analysis(self, time_range: Tuple[datetime, datetime],
                             granularity: TimeGranularity = TimeGranularity.HOUR) -> AnalyticsResult:
        """Convenience method for temporal analysis"""
        
        query = AnalyticsQuery(
            query_id=str(uuid.uuid4()),
            analytics_type=AnalyticsType.TEMPORAL_ANALYSIS,
            scope=AnalyticsScope.GLOBAL,
            time_range=time_range,
            granularity=granularity
        )
        
        return self.run_analytics(query)
    
    def get_usage_statistics(self, time_range: Tuple[datetime, datetime]) -> AnalyticsResult:
        """Convenience method for usage statistics"""
        
        query = AnalyticsQuery(
            query_id=str(uuid.uuid4()),
            analytics_type=AnalyticsType.USAGE_STATISTICS,
            scope=AnalyticsScope.GLOBAL,
            time_range=time_range
        )
        
        return self.run_analytics(query)
    
    def get_cognitive_insights(self, time_range: Tuple[datetime, datetime],
                              user_id: Optional[str] = None) -> AnalyticsResult:
        """Convenience method for cognitive insights"""
        
        query = AnalyticsQuery(
            query_id=str(uuid.uuid4()),
            analytics_type=AnalyticsType.COGNITIVE_INSIGHTS,
            scope=AnalyticsScope.USER if user_id else AnalyticsScope.GLOBAL,
            time_range=time_range,
            filters={'user_id': user_id} if user_id else {}
        )
        
        return self.run_analytics(query)
    
    def generate_comprehensive_report(self, time_range: Tuple[datetime, datetime],
                                    user_id: Optional[str] = None) -> Dict[str, AnalyticsResult]:
        """Generate comprehensive analytics report"""
        
        report = {}
        
        # Run all available analytics
        analytics_types = [
            AnalyticsType.FORMATION_PATTERNS,
            AnalyticsType.RETRIEVAL_PATTERNS,
            AnalyticsType.TEMPORAL_ANALYSIS,
            AnalyticsType.USAGE_STATISTICS,
            AnalyticsType.COGNITIVE_INSIGHTS
        ]
        
        for analytics_type in analytics_types:
            try:
                query = AnalyticsQuery(
                    query_id=str(uuid.uuid4()),
                    analytics_type=analytics_type,
                    scope=AnalyticsScope.USER if user_id else AnalyticsScope.GLOBAL,
                    time_range=time_range,
                    filters={'user_id': user_id} if user_id else {}
                )
                
                result = self.run_analytics(query)
                report[analytics_type.value] = result
                
            except Exception as e:
                logger.error(f"Error running {analytics_type.value} analytics: {e}")
                continue
        
        logger.info(f"Generated comprehensive report with {len(report)} analytics")
        
        return report
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of analytics system"""
        
        with self.lock:
            total_events = len(self.events)
            
            # Event type distribution
            event_types = Counter(event.event_type for event in self.events)
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_events = [e for e in self.events if e.timestamp >= recent_cutoff]
        
        return {
            'total_events_recorded': total_events,
            'recent_events_24h': len(recent_events),
            'event_type_distribution': dict(event_types),
            'queries_processed': self.queries_processed,
            'insights_generated': self.insights_generated,
            'cache_size': len(self.results_cache),
            'analyzers_available': list(self.analyzers.keys()),
            'background_processing_enabled': self.background_processing,
            'max_events_limit': self.max_events
        }
    
    def _cleanup_old_events(self):
        """Remove old events to maintain performance"""
        
        # Keep only the most recent events
        target_size = int(self.max_events * 0.7)  # Keep 70% after cleanup
        
        if len(self.events) > target_size:
            # Sort by timestamp and keep most recent
            self.events.sort(key=lambda x: x.timestamp)
            self.events = self.events[-target_size:]
            
            logger.info(f"Cleaned up old events, kept {len(self.events)} most recent")
    
    def _generate_cache_key(self, query: AnalyticsQuery) -> str:
        """Generate cache key for query"""
        
        key_data = {
            'analytics_type': query.analytics_type.value,
            'scope': query.scope.value,
            'time_range': (query.time_range[0].isoformat(), query.time_range[1].isoformat()),
            'granularity': query.granularity.value,
            'filters': sorted(query.filters.items()),
            'aggregations': sorted(query.aggregations),
            'group_by': sorted(query.group_by)
        }
        
        import hashlib
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[AnalyticsResult]:
        """Get cached result if available and not expired"""
        
        if cache_key not in self.results_cache:
            return None
        
        result = self.results_cache[cache_key]
        
        # Check if expired
        cache_age = datetime.now() - result.timestamp
        if cache_age > timedelta(hours=self.cache_ttl):
            del self.results_cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: AnalyticsResult):
        """Cache analytics result"""
        
        self.results_cache[cache_key] = result
        
        # Cleanup old cache entries
        current_time = datetime.now()
        expired_keys = []
        
        for key, cached_result in self.results_cache.items():
            cache_age = current_time - cached_result.timestamp
            if cache_age > timedelta(hours=self.cache_ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.results_cache[key]
    
    def cleanup_resources(self):
        """Cleanup resources and shutdown"""
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Clear caches
        self.results_cache.clear()
        
        logger.info("Memory Analytics Manager resources cleaned up")

# Example usage and utility functions
def create_sample_events(count: int = 100) -> List[MemoryEvent]:
    """Create sample events for testing"""
    
    events = []
    base_time = datetime.now() - timedelta(days=7)
    
    event_types = ['create', 'retrieve', 'update', 'delete']
    user_ids = ['user1', 'user2', 'user3', 'user4']
    session_ids = ['session1', 'session2', 'session3']
    
    for i in range(count):
        event = MemoryEvent(
            event_id=str(uuid.uuid4()),
            event_type=np.random.choice(event_types) if np else event_types[i % len(event_types)],
            memory_id=f"memory_{i % 20}",  # 20 different memories
            user_id=np.random.choice(user_ids) if np else user_ids[i % len(user_ids)],
            session_id=np.random.choice(session_ids) if np else session_ids[i % len(session_ids)],
            timestamp=base_time + timedelta(minutes=i * 10),
            duration_ms=np.random.uniform(100, 2000) if np else 500 + (i % 1000),
            success=np.random.random() > 0.05 if np else i % 20 != 0,  # 95% success rate
            metadata={'test': True, 'batch': i // 10},
            context={'source': 'test'}
        )
        events.append(event)
    
    return events

def example_usage():
    """Example usage of Memory Analytics Manager"""
    
    # Initialize manager
    config = {
        'cache_ttl_hours': 2,
        'background_processing': True,
        'max_events': 10000,
        'cleanup_threshold': 0.8
    }
    
    manager = MemoryAnalyticsManager(config)
    
    # Create sample events
    sample_events = create_sample_events(200)
    
    # Record events
    for event in sample_events:
        manager.record_event(event)
    
    print(f"Recorded {len(sample_events)} sample events")
    
    # Define time range for analysis
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    time_range = (start_time, end_time)
    
    # Run formation patterns analysis
    print("\n=== Formation Patterns Analysis ===")
    formation_result = manager.get_formation_patterns(time_range)
    print(f"Insights: {formation_result.insights}")
    print(f"Metrics: {formation_result.metrics}")
    
    # Run retrieval patterns analysis
    print("\n=== Retrieval Patterns Analysis ===")
    retrieval_result = manager.get_retrieval_patterns(time_range)
    print(f"Insights: {retrieval_result.insights}")
    print(f"Most accessed memories: {retrieval_result.data.get('most_accessed_memories', [])}")
    
    # Run temporal analysis
    print("\n=== Temporal Analysis ===")
    temporal_result = manager.get_temporal_analysis(time_range, TimeGranularity.HOUR)
    print(f"Insights: {temporal_result.insights}")
    print(f"Trend slope: {temporal_result.metrics.get('trend_slope', 'N/A')}")
    
    # Run usage statistics
    print("\n=== Usage Statistics ===")
    usage_result = manager.get_usage_statistics(time_range)
    print(f"Total events: {usage_result.metrics.get('total_events', 0)}")
    print(f"Success rate: {usage_result.metrics.get('overall_success_rate', 0):.1%}")
    
    # Run cognitive insights
    print("\n=== Cognitive Insights ===")
    cognitive_result = manager.get_cognitive_insights(time_range)
    print(f"Insights: {cognitive_result.insights}")
    print(f"Learning efficiency: {cognitive_result.metrics.get('learning_efficiency', 'N/A')}")
    
    # Generate comprehensive report
    print("\n=== Comprehensive Report ===")
    report = manager.generate_comprehensive_report(time_range)
    print(f"Generated report with {len(report)} analytics sections")
    
    # Get system summary
    print("\n=== System Summary ===")
    summary = manager.get_analytics_summary()
    print(f"Total events recorded: {summary['total_events_recorded']}")
    print(f"Queries processed: {summary['queries_processed']}")
    print(f"Insights generated: {summary['insights_generated']}")
    
    # Cleanup
    manager.cleanup_resources()
    print("\nAnalytics example completed successfully!")

if __name__ == "__main__":
    example_usage()