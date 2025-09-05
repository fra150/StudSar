"""Episodic Memory Module for StudSar V4

Implements time-aware episodic memory with temporal relationships,
event sequences, and contextual reconstruction capabilities.
"""

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
import json
from collections import defaultdict, deque
import bisect

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of episodic events"""
    MEMORY_CREATION = "memory_creation"
    MEMORY_ACCESS = "memory_access"
    MEMORY_UPDATE = "memory_update"
    SEARCH_QUERY = "search_query"
    ASSOCIATION_FORMED = "association_formed"
    CONTEXT_SWITCH = "context_switch"
    USER_INTERACTION = "user_interaction"
    SYSTEM_EVENT = "system_event"
    LEARNING_EVENT = "learning_event"
    ERROR_EVENT = "error_event"

class TemporalRelation(Enum):
    """Types of temporal relationships between events"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    STARTS = "starts"
    FINISHES = "finishes"
    EQUALS = "equals"
    CONTAINS = "contains"
    CONTAINED_BY = "contained_by"

class EpisodeType(Enum):
    """Types of episodes"""
    LEARNING_SESSION = "learning_session"
    PROBLEM_SOLVING = "problem_solving"
    EXPLORATION = "exploration"
    ROUTINE_TASK = "routine_task"
    CREATIVE_WORK = "creative_work"
    ERROR_RECOVERY = "error_recovery"
    SOCIAL_INTERACTION = "social_interaction"
    SYSTEM_MAINTENANCE = "system_maintenance"

@dataclass
class TemporalContext:
    """Temporal context information"""
    timestamp: datetime
    duration: Optional[timedelta] = None
    time_of_day: Optional[str] = None  # morning, afternoon, evening, night
    day_of_week: Optional[str] = None
    season: Optional[str] = None
    relative_time: Optional[str] = None  # recent, distant, etc.
    
    def __post_init__(self):
        if self.time_of_day is None:
            hour = self.timestamp.hour
            if 5 <= hour < 12:
                self.time_of_day = "morning"
            elif 12 <= hour < 17:
                self.time_of_day = "afternoon"
            elif 17 <= hour < 21:
                self.time_of_day = "evening"
            else:
                self.time_of_day = "night"
        
        if self.day_of_week is None:
            self.day_of_week = self.timestamp.strftime("%A").lower()
        
        if self.season is None:
            month = self.timestamp.month
            if month in [12, 1, 2]:
                self.season = "winter"
            elif month in [3, 4, 5]:
                self.season = "spring"
            elif month in [6, 7, 8]:
                self.season = "summer"
            else:
                self.season = "autumn"

@dataclass
class EpisodicEvent:
    """Individual episodic event"""
    id: str
    event_type: EventType
    content: Any
    temporal_context: TemporalContext
    spatial_context: Optional[Dict[str, Any]] = None
    emotional_context: Optional[Dict[str, float]] = None
    participants: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    causal_links: List[str] = field(default_factory=list)  # Events that caused this
    consequence_links: List[str] = field(default_factory=list)  # Events caused by this
    importance_score: float = 0.0
    vividness: float = 1.0
    confidence: float = 1.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_timestamp(self) -> datetime:
        """Get event timestamp"""
        return self.temporal_context.timestamp
    
    def get_duration(self) -> Optional[timedelta]:
        """Get event duration"""
        return self.temporal_context.duration
    
    def update_importance(self, factor: float):
        """Update importance score"""
        self.importance_score = max(0.0, min(1.0, self.importance_score + factor))
    
    def decay_vividness(self, time_elapsed: float, decay_rate: float = 0.01):
        """Decay vividness over time"""
        decay_factor = math.exp(-decay_rate * time_elapsed)
        self.vividness *= decay_factor
    
    def add_causal_link(self, cause_event_id: str):
        """Add causal relationship"""
        if cause_event_id not in self.causal_links:
            self.causal_links.append(cause_event_id)
    
    def add_consequence_link(self, consequence_event_id: str):
        """Add consequence relationship"""
        if consequence_event_id not in self.consequence_links:
            self.consequence_links.append(consequence_event_id)

@dataclass
class Episode:
    """Collection of related events forming an episode"""
    id: str
    episode_type: EpisodeType
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[str] = field(default_factory=list)  # Event IDs
    main_participants: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    emotional_tone: Optional[Dict[str, float]] = None
    importance_score: float = 0.0
    coherence_score: float = 0.0
    tags: Set[str] = field(default_factory=set)
    
    def get_duration(self) -> Optional[timedelta]:
        """Get episode duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def add_event(self, event_id: str):
        """Add event to episode"""
        if event_id not in self.events:
            self.events.append(event_id)
    
    def remove_event(self, event_id: str):
        """Remove event from episode"""
        if event_id in self.events:
            self.events.remove(event_id)
    
    def update_coherence(self, score: float):
        """Update coherence score"""
        self.coherence_score = max(0.0, min(1.0, score))
    
    def is_active(self) -> bool:
        """Check if episode is currently active"""
        return self.end_time is None
    
    def close_episode(self, end_time: Optional[datetime] = None):
        """Close the episode"""
        self.end_time = end_time or datetime.now()

class TemporalIndex:
    """Efficient temporal indexing for events"""
    
    def __init__(self):
        self.timeline: List[Tuple[datetime, str]] = []  # (timestamp, event_id)
        self.event_lookup: Dict[str, int] = {}  # event_id -> index in timeline
    
    def add_event(self, event_id: str, timestamp: datetime):
        """Add event to temporal index"""
        # Use binary search to maintain sorted order
        index = bisect.bisect_left(self.timeline, (timestamp, event_id))
        self.timeline.insert(index, (timestamp, event_id))
        
        # Update lookup indices
        for i in range(index, len(self.timeline)):
            event_id_at_i = self.timeline[i][1]
            self.event_lookup[event_id_at_i] = i
    
    def remove_event(self, event_id: str):
        """Remove event from temporal index"""
        if event_id in self.event_lookup:
            index = self.event_lookup[event_id]
            self.timeline.pop(index)
            del self.event_lookup[event_id]
            
            # Update lookup indices
            for i in range(index, len(self.timeline)):
                event_id_at_i = self.timeline[i][1]
                self.event_lookup[event_id_at_i] = i
    
    def get_events_in_range(self, start_time: datetime, end_time: datetime) -> List[str]:
        """Get events within time range"""
        start_index = bisect.bisect_left(self.timeline, (start_time, ""))
        end_index = bisect.bisect_right(self.timeline, (end_time, "\uffff"))
        
        return [event_id for _, event_id in self.timeline[start_index:end_index]]
    
    def get_events_before(self, timestamp: datetime, limit: int = 10) -> List[str]:
        """Get events before timestamp"""
        index = bisect.bisect_left(self.timeline, (timestamp, ""))
        start_index = max(0, index - limit)
        
        return [event_id for _, event_id in self.timeline[start_index:index]]
    
    def get_events_after(self, timestamp: datetime, limit: int = 10) -> List[str]:
        """Get events after timestamp"""
        index = bisect.bisect_right(self.timeline, (timestamp, "\uffff"))
        end_index = min(len(self.timeline), index + limit)
        
        return [event_id for _, event_id in self.timeline[index:end_index]]
    
    def get_nearest_events(self, timestamp: datetime, limit: int = 10) -> List[str]:
        """Get nearest events to timestamp"""
        index = bisect.bisect_left(self.timeline, (timestamp, ""))
        
        # Get events before and after
        before_events = self.get_events_before(timestamp, limit // 2)
        after_events = self.get_events_after(timestamp, limit // 2)
        
        # Combine and sort by distance
        all_events = []
        for event_id in before_events + after_events:
            event_index = self.event_lookup[event_id]
            event_timestamp = self.timeline[event_index][0]
            distance = abs((timestamp - event_timestamp).total_seconds())
            all_events.append((distance, event_id))
        
        all_events.sort(key=lambda x: x[0])
        return [event_id for _, event_id in all_events[:limit]]

class TemporalRelationshipAnalyzer:
    """Analyzes temporal relationships between events"""
    
    @staticmethod
    def analyze_relationship(event1: EpisodicEvent, event2: EpisodicEvent) -> TemporalRelation:
        """Analyze temporal relationship between two events"""
        t1_start = event1.get_timestamp()
        t1_end = t1_start + (event1.get_duration() or timedelta(0))
        
        t2_start = event2.get_timestamp()
        t2_end = t2_start + (event2.get_duration() or timedelta(0))
        
        # Allen's interval algebra
        if t1_end < t2_start:
            return TemporalRelation.BEFORE
        elif t2_end < t1_start:
            return TemporalRelation.AFTER
        elif t1_start == t2_start and t1_end == t2_end:
            return TemporalRelation.EQUALS
        elif t1_start <= t2_start and t1_end >= t2_end:
            return TemporalRelation.CONTAINS
        elif t2_start <= t1_start and t2_end >= t1_end:
            return TemporalRelation.CONTAINED_BY
        elif t1_start < t2_start and t1_end > t2_start and t1_end < t2_end:
            return TemporalRelation.OVERLAPS
        elif t1_start == t2_start and t1_end < t2_end:
            return TemporalRelation.STARTS
        elif t1_end == t2_end and t1_start > t2_start:
            return TemporalRelation.FINISHES
        elif t1_end == t2_start:
            return TemporalRelation.MEETS
        else:
            return TemporalRelation.DURING
    
    @staticmethod
    def calculate_temporal_distance(event1: EpisodicEvent, event2: EpisodicEvent) -> float:
        """Calculate temporal distance between events in hours"""
        t1 = event1.get_timestamp()
        t2 = event2.get_timestamp()
        return abs((t2 - t1).total_seconds()) / 3600.0
    
    @staticmethod
    def find_temporal_patterns(events: List[EpisodicEvent]) -> Dict[str, Any]:
        """Find temporal patterns in event sequence"""
        if len(events) < 2:
            return {}
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get_timestamp())
        
        patterns = {
            'intervals': [],
            'recurring_patterns': {},
            'temporal_clusters': [],
            'rhythm_analysis': {}
        }
        
        # Analyze intervals between consecutive events
        for i in range(len(sorted_events) - 1):
            interval = (sorted_events[i + 1].get_timestamp() - 
                       sorted_events[i].get_timestamp()).total_seconds()
            patterns['intervals'].append(interval)
        
        # Find recurring time patterns
        time_of_day_counts = defaultdict(int)
        day_of_week_counts = defaultdict(int)
        
        for event in sorted_events:
            time_of_day_counts[event.temporal_context.time_of_day] += 1
            day_of_week_counts[event.temporal_context.day_of_week] += 1
        
        patterns['recurring_patterns'] = {
            'time_of_day': dict(time_of_day_counts),
            'day_of_week': dict(day_of_week_counts)
        }
        
        # Identify temporal clusters (events close in time)
        clusters = []
        current_cluster = [sorted_events[0]]
        cluster_threshold = 3600  # 1 hour
        
        for i in range(1, len(sorted_events)):
            time_diff = (sorted_events[i].get_timestamp() - 
                        sorted_events[i-1].get_timestamp()).total_seconds()
            
            if time_diff <= cluster_threshold:
                current_cluster.append(sorted_events[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append([e.id for e in current_cluster])
                current_cluster = [sorted_events[i]]
        
        if len(current_cluster) > 1:
            clusters.append([e.id for e in current_cluster])
        
        patterns['temporal_clusters'] = clusters
        
        # Rhythm analysis
        if len(patterns['intervals']) > 0:
            avg_interval = sum(patterns['intervals']) / len(patterns['intervals'])
            interval_variance = sum((x - avg_interval) ** 2 for x in patterns['intervals']) / len(patterns['intervals'])
            
            patterns['rhythm_analysis'] = {
                'average_interval': avg_interval,
                'interval_variance': interval_variance,
                'regularity_score': 1.0 / (1.0 + interval_variance / (avg_interval ** 2)) if avg_interval > 0 else 0.0
            }
        
        return patterns

class EpisodicMemoryManager:
    """Main manager for episodic memory system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.events: Dict[str, EpisodicEvent] = {}
        self.episodes: Dict[str, Episode] = {}
        
        # Indexing
        self.temporal_index = TemporalIndex()
        self.type_index: Dict[EventType, Set[str]] = defaultdict(set)
        self.participant_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Analysis
        self.relationship_analyzer = TemporalRelationshipAnalyzer()
        
        # Configuration
        self.max_events = config.get('max_events', 100000)
        self.max_episodes = config.get('max_episodes', 10000)
        self.auto_episode_detection = config.get('auto_episode_detection', True)
        self.episode_gap_threshold = config.get('episode_gap_threshold', 3600)  # seconds
        self.importance_threshold = config.get('importance_threshold', 0.3)
        self.vividness_decay_rate = config.get('vividness_decay_rate', 0.01)
        
        # Current episode tracking
        self.current_episode: Optional[Episode] = None
        self.last_event_time: Optional[datetime] = None
        
        # Statistics
        self.total_events_created = 0
        self.total_episodes_created = 0
    
    async def add_event(self, event_type: EventType, content: Any, 
                       temporal_context: Optional[TemporalContext] = None,
                       spatial_context: Optional[Dict[str, Any]] = None,
                       emotional_context: Optional[Dict[str, float]] = None,
                       participants: Optional[List[str]] = None,
                       tags: Optional[Set[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add new episodic event"""
        
        # Generate event ID
        event_id = f"event_{int(time.time() * 1000000)}"
        
        # Create temporal context if not provided
        if temporal_context is None:
            temporal_context = TemporalContext(timestamp=datetime.now())
        
        # Create event
        event = EpisodicEvent(
            id=event_id,
            event_type=event_type,
            content=content,
            temporal_context=temporal_context,
            spatial_context=spatial_context,
            emotional_context=emotional_context,
            participants=participants or [],
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        # Calculate importance
        event.importance_score = self._calculate_event_importance(event)
        
        # Store event
        self.events[event_id] = event
        self.total_events_created += 1
        
        # Update indices
        self.temporal_index.add_event(event_id, event.get_timestamp())
        self.type_index[event_type].add(event_id)
        
        for participant in event.participants:
            self.participant_index[participant].add(event_id)
        
        for tag in event.tags:
            self.tag_index[tag].add(event_id)
        
        # Auto-detect causal relationships
        await self._detect_causal_relationships(event)
        
        # Episode management
        if self.auto_episode_detection:
            await self._manage_episodes(event)
        
        # Cleanup if necessary
        if len(self.events) > self.max_events:
            await self._cleanup_old_events()
        
        self.last_event_time = event.get_timestamp()
        
        logger.debug(f"Added episodic event: {event_id}")
        return event_id
    
    async def create_episode(self, episode_type: EpisodeType, title: str, 
                           description: str, event_ids: Optional[List[str]] = None,
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Create new episode"""
        
        episode_id = f"episode_{int(time.time() * 1000000)}"
        
        episode = Episode(
            id=episode_id,
            episode_type=episode_type,
            title=title,
            description=description,
            start_time=datetime.now(),
            events=event_ids or [],
            context=context or {}
        )
        
        # Calculate initial importance and coherence
        if event_ids:
            episode.importance_score = self._calculate_episode_importance(episode)
            episode.coherence_score = self._calculate_episode_coherence(episode)
        
        self.episodes[episode_id] = episode
        self.total_episodes_created += 1
        
        logger.info(f"Created episode: {episode_id} - {title}")
        return episode_id
    
    async def close_current_episode(self, outcome: Optional[str] = None,
                                  lessons_learned: Optional[List[str]] = None):
        """Close current active episode"""
        if self.current_episode:
            self.current_episode.close_episode()
            
            if outcome:
                self.current_episode.outcome = outcome
            
            if lessons_learned:
                self.current_episode.lessons_learned.extend(lessons_learned)
            
            # Recalculate scores
            self.current_episode.importance_score = self._calculate_episode_importance(self.current_episode)
            self.current_episode.coherence_score = self._calculate_episode_coherence(self.current_episode)
            
            logger.info(f"Closed episode: {self.current_episode.id}")
            self.current_episode = None
    
    async def retrieve_events(self, query_type: str = "temporal", 
                            **kwargs) -> List[EpisodicEvent]:
        """Retrieve events based on various criteria"""
        
        if query_type == "temporal":
            return await self._retrieve_by_temporal_criteria(**kwargs)
        elif query_type == "content":
            return await self._retrieve_by_content(**kwargs)
        elif query_type == "participant":
            return await self._retrieve_by_participant(**kwargs)
        elif query_type == "type":
            return await self._retrieve_by_type(**kwargs)
        elif query_type == "causal":
            return await self._retrieve_by_causal_chain(**kwargs)
        elif query_type == "contextual":
            return await self._retrieve_by_context(**kwargs)
        else:
            return []
    
    async def retrieve_episodes(self, episode_type: Optional[EpisodeType] = None,
                              time_range: Optional[Tuple[datetime, datetime]] = None,
                              min_importance: Optional[float] = None,
                              tags: Optional[Set[str]] = None) -> List[Episode]:
        """Retrieve episodes based on criteria"""
        
        episodes = list(self.episodes.values())
        
        # Filter by type
        if episode_type:
            episodes = [ep for ep in episodes if ep.episode_type == episode_type]
        
        # Filter by time range
        if time_range:
            start_time, end_time = time_range
            episodes = [ep for ep in episodes 
                       if ep.start_time >= start_time and 
                       (ep.end_time is None or ep.end_time <= end_time)]
        
        # Filter by importance
        if min_importance is not None:
            episodes = [ep for ep in episodes if ep.importance_score >= min_importance]
        
        # Filter by tags
        if tags:
            episodes = [ep for ep in episodes if tags.issubset(ep.tags)]
        
        # Sort by importance and recency
        episodes.sort(key=lambda ep: (ep.importance_score, ep.start_time), reverse=True)
        
        return episodes
    
    async def reconstruct_temporal_sequence(self, start_time: datetime, 
                                          end_time: datetime,
                                          include_context: bool = True) -> Dict[str, Any]:
        """Reconstruct temporal sequence of events"""
        
        event_ids = self.temporal_index.get_events_in_range(start_time, end_time)
        events = [self.events[eid] for eid in event_ids if eid in self.events]
        
        # Sort by timestamp
        events.sort(key=lambda e: e.get_timestamp())
        
        # Analyze temporal patterns
        patterns = self.relationship_analyzer.find_temporal_patterns(events)
        
        # Build causal chains
        causal_chains = self._build_causal_chains(events)
        
        # Identify key episodes
        relevant_episodes = []
        for episode in self.episodes.values():
            if (episode.start_time <= end_time and 
                (episode.end_time is None or episode.end_time >= start_time)):
                relevant_episodes.append(episode)
        
        reconstruction = {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration': (end_time - start_time).total_seconds()
            },
            'events': [{
                'id': event.id,
                'type': event.event_type.value,
                'timestamp': event.get_timestamp().isoformat(),
                'content': str(event.content)[:200],  # Truncate for summary
                'importance': event.importance_score,
                'vividness': event.vividness
            } for event in events],
            'episodes': [{
                'id': episode.id,
                'type': episode.episode_type.value,
                'title': episode.title,
                'start_time': episode.start_time.isoformat(),
                'end_time': episode.end_time.isoformat() if episode.end_time else None,
                'importance': episode.importance_score,
                'coherence': episode.coherence_score
            } for episode in relevant_episodes],
            'temporal_patterns': patterns,
            'causal_chains': causal_chains,
            'statistics': {
                'total_events': len(events),
                'total_episodes': len(relevant_episodes),
                'average_importance': sum(e.importance_score for e in events) / len(events) if events else 0,
                'time_coverage': len(events) / max(1, (end_time - start_time).total_seconds() / 3600)  # events per hour
            }
        }
        
        if include_context:
            reconstruction['contextual_analysis'] = await self._analyze_contextual_factors(events)
        
        return reconstruction
    
    async def find_similar_episodes(self, target_episode_id: str, 
                                  similarity_threshold: float = 0.7) -> List[Tuple[Episode, float]]:
        """Find episodes similar to target episode"""
        
        if target_episode_id not in self.episodes:
            return []
        
        target_episode = self.episodes[target_episode_id]
        similar_episodes = []
        
        for episode_id, episode in self.episodes.items():
            if episode_id == target_episode_id:
                continue
            
            similarity = self._calculate_episode_similarity(target_episode, episode)
            
            if similarity >= similarity_threshold:
                similar_episodes.append((episode, similarity))
        
        # Sort by similarity
        similar_episodes.sort(key=lambda x: x[1], reverse=True)
        
        return similar_episodes
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        current_time = datetime.now()
        
        # Event statistics
        event_types = defaultdict(int)
        importance_distribution = []
        vividness_distribution = []
        
        for event in self.events.values():
            event_types[event.event_type.value] += 1
            importance_distribution.append(event.importance_score)
            vividness_distribution.append(event.vividness)
        
        # Episode statistics
        episode_types = defaultdict(int)
        episode_durations = []
        
        for episode in self.episodes.values():
            episode_types[episode.episode_type.value] += 1
            if episode.get_duration():
                episode_durations.append(episode.get_duration().total_seconds())
        
        # Temporal statistics
        if self.events:
            oldest_event = min(self.events.values(), key=lambda e: e.get_timestamp())
            newest_event = max(self.events.values(), key=lambda e: e.get_timestamp())
            memory_span = (newest_event.get_timestamp() - oldest_event.get_timestamp()).total_seconds()
        else:
            memory_span = 0
        
        return {
            'events': {
                'total': len(self.events),
                'by_type': dict(event_types),
                'average_importance': sum(importance_distribution) / len(importance_distribution) if importance_distribution else 0,
                'average_vividness': sum(vividness_distribution) / len(vividness_distribution) if vividness_distribution else 0
            },
            'episodes': {
                'total': len(self.episodes),
                'active': sum(1 for ep in self.episodes.values() if ep.is_active()),
                'by_type': dict(episode_types),
                'average_duration': sum(episode_durations) / len(episode_durations) if episode_durations else 0
            },
            'temporal': {
                'memory_span_hours': memory_span / 3600,
                'events_per_hour': len(self.events) / max(1, memory_span / 3600),
                'last_event_time': self.last_event_time.isoformat() if self.last_event_time else None
            },
            'system': {
                'total_events_created': self.total_events_created,
                'total_episodes_created': self.total_episodes_created,
                'current_episode_active': self.current_episode is not None
            }
        }
    
    # Private methods
    
    def _calculate_event_importance(self, event: EpisodicEvent) -> float:
        """Calculate importance score for event"""
        importance = 0.0
        
        # Base importance by event type
        type_weights = {
            EventType.ERROR_EVENT: 0.8,
            EventType.LEARNING_EVENT: 0.7,
            EventType.USER_INTERACTION: 0.6,
            EventType.MEMORY_CREATION: 0.5,
            EventType.ASSOCIATION_FORMED: 0.4,
            EventType.MEMORY_ACCESS: 0.3,
            EventType.SEARCH_QUERY: 0.2,
            EventType.SYSTEM_EVENT: 0.1
        }
        
        importance += type_weights.get(event.event_type, 0.3)
        
        # Emotional context boost
        if event.emotional_context:
            emotional_intensity = sum(abs(v) for v in event.emotional_context.values())
            importance += min(0.3, emotional_intensity / 10.0)
        
        # Participant count boost
        importance += min(0.2, len(event.participants) * 0.05)
        
        # Tag count boost
        importance += min(0.1, len(event.tags) * 0.02)
        
        return min(1.0, importance)
    
    def _calculate_episode_importance(self, episode: Episode) -> float:
        """Calculate importance score for episode"""
        if not episode.events:
            return 0.0
        
        # Average importance of constituent events
        event_importances = []
        for event_id in episode.events:
            if event_id in self.events:
                event_importances.append(self.events[event_id].importance_score)
        
        if not event_importances:
            return 0.0
        
        avg_importance = sum(event_importances) / len(event_importances)
        
        # Duration factor
        duration_factor = 0.0
        if episode.get_duration():
            duration_hours = episode.get_duration().total_seconds() / 3600
            duration_factor = min(0.3, duration_hours / 24)  # Up to 0.3 for day-long episodes
        
        # Outcome factor
        outcome_factor = 0.1 if episode.outcome else 0.0
        
        # Lessons learned factor
        lessons_factor = min(0.2, len(episode.lessons_learned) * 0.05)
        
        return min(1.0, avg_importance + duration_factor + outcome_factor + lessons_factor)
    
    def _calculate_episode_coherence(self, episode: Episode) -> float:
        """Calculate coherence score for episode"""
        if len(episode.events) < 2:
            return 1.0
        
        # Temporal coherence - events should be close in time
        temporal_coherence = 0.0
        event_times = []
        
        for event_id in episode.events:
            if event_id in self.events:
                event_times.append(self.events[event_id].get_timestamp())
        
        if len(event_times) >= 2:
            event_times.sort()
            total_span = (event_times[-1] - event_times[0]).total_seconds()
            avg_gap = total_span / (len(event_times) - 1) if len(event_times) > 1 else 0
            
            # Coherence decreases with larger gaps
            temporal_coherence = 1.0 / (1.0 + avg_gap / 3600)  # Normalize by hour
        
        # Causal coherence - events should be causally related
        causal_coherence = 0.0
        causal_links = 0
        total_possible_links = len(episode.events) * (len(episode.events) - 1) / 2
        
        for event_id in episode.events:
            if event_id in self.events:
                event = self.events[event_id]
                causal_links += len([link for link in event.causal_links if link in episode.events])
                causal_links += len([link for link in event.consequence_links if link in episode.events])
        
        if total_possible_links > 0:
            causal_coherence = causal_links / total_possible_links
        
        # Thematic coherence - events should share participants or tags
        thematic_coherence = 0.0
        all_participants = set()
        all_tags = set()
        
        for event_id in episode.events:
            if event_id in self.events:
                event = self.events[event_id]
                all_participants.update(event.participants)
                all_tags.update(event.tags)
        
        if episode.events:
            participant_overlap = len(all_participants) / len(episode.events)
            tag_overlap = len(all_tags) / len(episode.events)
            thematic_coherence = min(1.0, (participant_overlap + tag_overlap) / 2)
        
        # Weighted average
        return (temporal_coherence * 0.4 + causal_coherence * 0.4 + thematic_coherence * 0.2)
    
    def _calculate_episode_similarity(self, episode1: Episode, episode2: Episode) -> float:
        """Calculate similarity between two episodes"""
        similarity = 0.0
        
        # Type similarity
        if episode1.episode_type == episode2.episode_type:
            similarity += 0.3
        
        # Temporal similarity
        time_diff = abs((episode1.start_time - episode2.start_time).total_seconds())
        temporal_similarity = 1.0 / (1.0 + time_diff / 86400)  # Normalize by day
        similarity += temporal_similarity * 0.2
        
        # Duration similarity
        dur1 = episode1.get_duration()
        dur2 = episode2.get_duration()
        if dur1 and dur2:
            duration_ratio = min(dur1.total_seconds(), dur2.total_seconds()) / max(dur1.total_seconds(), dur2.total_seconds())
            similarity += duration_ratio * 0.1
        
        # Participant similarity
        participants1 = set(episode1.main_participants)
        participants2 = set(episode2.main_participants)
        if participants1 or participants2:
            participant_similarity = len(participants1 & participants2) / len(participants1 | participants2)
            similarity += participant_similarity * 0.2
        
        # Tag similarity
        tags1 = episode1.tags
        tags2 = episode2.tags
        if tags1 or tags2:
            tag_similarity = len(tags1 & tags2) / len(tags1 | tags2)
            similarity += tag_similarity * 0.2
        
        return min(1.0, similarity)
    
    async def _detect_causal_relationships(self, event: EpisodicEvent):
        """Detect causal relationships with recent events"""
        # Look for events in the recent past that might have caused this event
        recent_threshold = timedelta(minutes=30)
        recent_events = self.temporal_index.get_events_before(
            event.get_timestamp(), limit=10
        )
        
        for recent_event_id in recent_events:
            if recent_event_id in self.events:
                recent_event = self.events[recent_event_id]
                time_diff = event.get_timestamp() - recent_event.get_timestamp()
                
                if time_diff <= recent_threshold:
                    # Simple causal detection based on event types and content
                    if self._is_causal_relationship(recent_event, event):
                        event.add_causal_link(recent_event_id)
                        recent_event.add_consequence_link(event.id)
    
    def _is_causal_relationship(self, cause_event: EpisodicEvent, effect_event: EpisodicEvent) -> bool:
        """Determine if one event likely caused another"""
        # Simple heuristics for causal relationships
        causal_patterns = [
            (EventType.SEARCH_QUERY, EventType.MEMORY_ACCESS),
            (EventType.MEMORY_ACCESS, EventType.ASSOCIATION_FORMED),
            (EventType.USER_INTERACTION, EventType.MEMORY_CREATION),
            (EventType.ERROR_EVENT, EventType.SYSTEM_EVENT),
            (EventType.LEARNING_EVENT, EventType.MEMORY_UPDATE)
        ]
        
        for cause_type, effect_type in causal_patterns:
            if cause_event.event_type == cause_type and effect_event.event_type == effect_type:
                return True
        
        # Check for shared participants or content
        shared_participants = set(cause_event.participants) & set(effect_event.participants)
        if shared_participants:
            return True
        
        # Check for content similarity (simple string matching)
        if isinstance(cause_event.content, str) and isinstance(effect_event.content, str):
            cause_words = set(cause_event.content.lower().split())
            effect_words = set(effect_event.content.lower().split())
            overlap = len(cause_words & effect_words)
            if overlap >= 2:  # At least 2 shared words
                return True
        
        return False
    
    async def _manage_episodes(self, event: EpisodicEvent):
        """Manage episode creation and closure based on event patterns"""
        current_time = event.get_timestamp()
        
        # Check if we should close current episode
        if self.current_episode and self.last_event_time:
            time_gap = (current_time - self.last_event_time).total_seconds()
            
            if time_gap > self.episode_gap_threshold:
                await self.close_current_episode()
        
        # Check if we should start new episode
        if not self.current_episode:
            episode_type = self._infer_episode_type(event)
            
            if episode_type:
                episode_id = await self.create_episode(
                    episode_type=episode_type,
                    title=f"Auto-detected {episode_type.value}",
                    description=f"Episode starting with {event.event_type.value}",
                    event_ids=[event.id]
                )
                
                self.current_episode = self.episodes[episode_id]
        
        # Add event to current episode
        if self.current_episode:
            self.current_episode.add_event(event.id)
    
    def _infer_episode_type(self, event: EpisodicEvent) -> Optional[EpisodeType]:
        """Infer episode type from event characteristics"""
        type_mapping = {
            EventType.LEARNING_EVENT: EpisodeType.LEARNING_SESSION,
            EventType.ERROR_EVENT: EpisodeType.ERROR_RECOVERY,
            EventType.USER_INTERACTION: EpisodeType.SOCIAL_INTERACTION,
            EventType.SYSTEM_EVENT: EpisodeType.SYSTEM_MAINTENANCE
        }
        
        return type_mapping.get(event.event_type)
    
    async def _retrieve_by_temporal_criteria(self, start_time: Optional[datetime] = None,
                                           end_time: Optional[datetime] = None,
                                           relative_time: Optional[str] = None,
                                           limit: int = 50) -> List[EpisodicEvent]:
        """Retrieve events by temporal criteria"""
        
        if relative_time:
            current_time = datetime.now()
            if relative_time == "recent":
                start_time = current_time - timedelta(hours=24)
                end_time = current_time
            elif relative_time == "today":
                start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = current_time
            elif relative_time == "this_week":
                days_since_monday = current_time.weekday()
                start_time = current_time - timedelta(days=days_since_monday)
                start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = current_time
        
        if start_time and end_time:
            event_ids = self.temporal_index.get_events_in_range(start_time, end_time)
        elif start_time:
            event_ids = self.temporal_index.get_events_after(start_time, limit)
        elif end_time:
            event_ids = self.temporal_index.get_events_before(end_time, limit)
        else:
            # Return most recent events
            event_ids = list(self.events.keys())[-limit:]
        
        events = [self.events[eid] for eid in event_ids if eid in self.events]
        return sorted(events, key=lambda e: e.get_timestamp(), reverse=True)[:limit]
    
    async def _retrieve_by_content(self, content_query: str, limit: int = 50) -> List[EpisodicEvent]:
        """Retrieve events by content similarity"""
        matching_events = []
        query_words = set(content_query.lower().split())
        
        for event in self.events.values():
            if isinstance(event.content, str):
                content_words = set(event.content.lower().split())
                overlap = len(query_words & content_words)
                
                if overlap > 0:
                    similarity = overlap / len(query_words | content_words)
                    matching_events.append((event, similarity))
        
        # Sort by similarity
        matching_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, _ in matching_events[:limit]]
    
    async def _retrieve_by_participant(self, participant: str, limit: int = 50) -> List[EpisodicEvent]:
        """Retrieve events by participant"""
        event_ids = self.participant_index.get(participant, set())
        events = [self.events[eid] for eid in event_ids if eid in self.events]
        return sorted(events, key=lambda e: e.get_timestamp(), reverse=True)[:limit]
    
    async def _retrieve_by_type(self, event_type: EventType, limit: int = 50) -> List[EpisodicEvent]:
        """Retrieve events by type"""
        event_ids = self.type_index.get(event_type, set())
        events = [self.events[eid] for eid in event_ids if eid in self.events]
        return sorted(events, key=lambda e: e.get_timestamp(), reverse=True)[:limit]
    
    async def _retrieve_by_causal_chain(self, root_event_id: str, 
                                      direction: str = "forward",
                                      max_depth: int = 5) -> List[EpisodicEvent]:
        """Retrieve events in causal chain"""
        if root_event_id not in self.events:
            return []
        
        visited = set()
        result = []
        queue = [(root_event_id, 0)]  # (event_id, depth)
        
        while queue and len(result) < 100:  # Limit results
            event_id, depth = queue.pop(0)
            
            if event_id in visited or depth > max_depth:
                continue
            
            visited.add(event_id)
            
            if event_id in self.events:
                event = self.events[event_id]
                result.append(event)
                
                # Add related events to queue
                if direction == "forward":
                    for consequence_id in event.consequence_links:
                        if consequence_id not in visited:
                            queue.append((consequence_id, depth + 1))
                elif direction == "backward":
                    for cause_id in event.causal_links:
                        if cause_id not in visited:
                            queue.append((cause_id, depth + 1))
                else:  # both directions
                    for related_id in event.causal_links + event.consequence_links:
                        if related_id not in visited:
                            queue.append((related_id, depth + 1))
        
        return result
    
    async def _retrieve_by_context(self, context_query: Dict[str, Any], 
                                 limit: int = 50) -> List[EpisodicEvent]:
        """Retrieve events by contextual criteria"""
        matching_events = []
        
        for event in self.events.values():
            match_score = 0.0
            total_criteria = 0
            
            # Check temporal context
            if "time_of_day" in context_query:
                total_criteria += 1
                if event.temporal_context.time_of_day == context_query["time_of_day"]:
                    match_score += 1.0
            
            if "day_of_week" in context_query:
                total_criteria += 1
                if event.temporal_context.day_of_week == context_query["day_of_week"]:
                    match_score += 1.0
            
            # Check spatial context
            if "location" in context_query and event.spatial_context:
                total_criteria += 1
                if event.spatial_context.get("location") == context_query["location"]:
                    match_score += 1.0
            
            # Check emotional context
            if "emotion" in context_query and event.emotional_context:
                total_criteria += 1
                if context_query["emotion"] in event.emotional_context:
                    match_score += event.emotional_context[context_query["emotion"]]
            
            # Check tags
            if "tags" in context_query:
                query_tags = set(context_query["tags"])
                total_criteria += 1
                if query_tags.issubset(event.tags):
                    match_score += 1.0
                elif query_tags & event.tags:
                    match_score += len(query_tags & event.tags) / len(query_tags)
            
            if total_criteria > 0:
                final_score = match_score / total_criteria
                if final_score > 0.5:  # Threshold for relevance
                    matching_events.append((event, final_score))
        
        # Sort by match score
        matching_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, _ in matching_events[:limit]]
    
    def _build_causal_chains(self, events: List[EpisodicEvent]) -> List[Dict[str, Any]]:
        """Build causal chains from events"""
        chains = []
        processed = set()
        
        for event in events:
            if event.id in processed:
                continue
            
            # Find root events (no causal links to other events in the set)
            event_ids = {e.id for e in events}
            root_causes = [link for link in event.causal_links if link not in event_ids]
            
            if not root_causes or not event.causal_links:
                # This could be a root event
                chain = self._trace_causal_chain(event, events, processed)
                if len(chain) > 1:
                    chains.append({
                        'root_event': event.id,
                        'chain': chain,
                        'length': len(chain)
                    })
        
        return chains
    
    def _trace_causal_chain(self, root_event: EpisodicEvent, 
                          available_events: List[EpisodicEvent],
                          processed: Set[str]) -> List[str]:
        """Trace causal chain from root event"""
        chain = [root_event.id]
        processed.add(root_event.id)
        
        available_ids = {e.id: e for e in available_events}
        current_event = root_event
        
        while current_event.consequence_links:
            next_event_id = None
            
            # Find next event in chain
            for consequence_id in current_event.consequence_links:
                if consequence_id in available_ids and consequence_id not in processed:
                    next_event_id = consequence_id
                    break
            
            if next_event_id:
                chain.append(next_event_id)
                processed.add(next_event_id)
                current_event = available_ids[next_event_id]
            else:
                break
        
        return chain
    
    async def _analyze_contextual_factors(self, events: List[EpisodicEvent]) -> Dict[str, Any]:
        """Analyze contextual factors in event sequence"""
        if not events:
            return {}
        
        # Temporal context analysis
        time_of_day_dist = defaultdict(int)
        day_of_week_dist = defaultdict(int)
        
        # Spatial context analysis
        locations = defaultdict(int)
        
        # Emotional context analysis
        emotions = defaultdict(list)
        
        # Participant analysis
        participants = defaultdict(int)
        
        for event in events:
            # Temporal
            time_of_day_dist[event.temporal_context.time_of_day] += 1
            day_of_week_dist[event.temporal_context.day_of_week] += 1
            
            # Spatial
            if event.spatial_context and "location" in event.spatial_context:
                locations[event.spatial_context["location"]] += 1
            
            # Emotional
            if event.emotional_context:
                for emotion, intensity in event.emotional_context.items():
                    emotions[emotion].append(intensity)
            
            # Participants
            for participant in event.participants:
                participants[participant] += 1
        
        # Calculate averages for emotions
        avg_emotions = {}
        for emotion, intensities in emotions.items():
            avg_emotions[emotion] = sum(intensities) / len(intensities)
        
        return {
            'temporal_patterns': {
                'time_of_day_distribution': dict(time_of_day_dist),
                'day_of_week_distribution': dict(day_of_week_dist)
            },
            'spatial_patterns': {
                'location_distribution': dict(locations)
            },
            'emotional_patterns': {
                'average_emotions': avg_emotions,
                'emotional_events': len([e for e in events if e.emotional_context])
            },
            'social_patterns': {
                'participant_frequency': dict(participants),
                'social_events': len([e for e in events if e.participants])
            }
        }
    
    async def _cleanup_old_events(self):
        """Clean up old, unimportant events"""
        if len(self.events) <= self.max_events:
            return
        
        # Sort events by importance and age
        events_to_remove = []
        current_time = datetime.now()
        
        for event in self.events.values():
            age_hours = (current_time - event.get_timestamp()).total_seconds() / 3600
            
            # Remove old, unimportant events
            if (age_hours > 168 and  # Older than a week
                event.importance_score < self.importance_threshold and
                event.vividness < 0.3):
                events_to_remove.append(event.id)
        
        # Remove oldest events if still over limit
        if len(self.events) - len(events_to_remove) > self.max_events:
            remaining_events = [e for e in self.events.values() if e.id not in events_to_remove]
            remaining_events.sort(key=lambda e: (e.importance_score, e.get_timestamp()))
            
            excess_count = len(self.events) - len(events_to_remove) - self.max_events
            for i in range(excess_count):
                events_to_remove.append(remaining_events[i].id)
        
        # Remove events
        for event_id in events_to_remove:
            if event_id in self.events:
                event = self.events[event_id]
                
                # Remove from indices
                self.temporal_index.remove_event(event_id)
                self.type_index[event.event_type].discard(event_id)
                
                for participant in event.participants:
                    self.participant_index[participant].discard(event_id)
                
                for tag in event.tags:
                    self.tag_index[tag].discard(event_id)
                
                # Remove from episodes
                for episode in self.episodes.values():
                    episode.remove_event(event_id)
                
                # Remove event
                del self.events[event_id]
        
        logger.info(f"Cleaned up {len(events_to_remove)} old events")

# Default configuration
DEFAULT_EPISODIC_CONFIG = {
    'max_events': 100000,
    'max_episodes': 10000,
    'auto_episode_detection': True,
    'episode_gap_threshold': 3600,  # 1 hour
    'importance_threshold': 0.3,
    'vividness_decay_rate': 0.01
}

def create_episodic_memory(config: Optional[Dict[str, Any]] = None) -> EpisodicMemoryManager:
    """Create episodic memory manager with default or custom config"""
    if config is None:
        config = DEFAULT_EPISODIC_CONFIG
    
    return EpisodicMemoryManager(config)