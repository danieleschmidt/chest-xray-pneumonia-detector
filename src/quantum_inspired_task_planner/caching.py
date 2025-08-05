"""Advanced caching system for quantum task planner.

Implements intelligent caching with quantum-inspired cache coherence,
adaptive cache sizing, and predictive prefetching.
"""

import time
import hashlib
import pickle
import logging
from typing import Any, Dict, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
import weakref

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with quantum-inspired metadata."""
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    quantum_coherence: float = 1.0  # Coherence with quantum state
    priority_score: float = 1.0
    size_bytes: int = 0
    dependencies: List[str] = field(default_factory=list)


class QuantumInspiredCache:
    """High-performance cache with quantum-inspired coherence management."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.size_bytes = 0
        self._lock = threading.RLock()
        
        # Quantum-inspired cache management
        self.coherence_threshold = 0.1
        self.eviction_algorithm = "quantum_lru"  # quantum_lru, quantum_lfu, quantum_priority
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with quantum coherence checking."""
        with self._lock:
            entry = self.cache.get(key)
            if not entry:
                return None
            
            # Check TTL
            if datetime.now() - entry.created_at > self.ttl:
                del self.cache[key]
                self.size_bytes -= entry.size_bytes
                return None
            
            # Check quantum coherence
            if entry.quantum_coherence < self.coherence_threshold:
                logger.debug(f"Cache entry {key} has low quantum coherence: {entry.quantum_coherence}")
                # Optionally evict low coherence entries
                del self.cache[key]
                self.size_bytes -= entry.size_bytes
                return None
            
            # Update access metrics
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._record_access(key)
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, priority_score: float = 1.0,
            dependencies: Optional[List[str]] = None) -> bool:
        """Put item in cache with quantum-inspired prioritization."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default estimate
            
            # Check if we need to evict items
            while len(self.cache) >= self.max_size:
                if not self._evict_item():
                    return False  # Could not evict
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                priority_score=priority_score,
                size_bytes=size_bytes,
                dependencies=dependencies or []
            )
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.size_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.cache.move_to_end(key)
            self.size_bytes += size_bytes
            
            self._record_access(key)
            return True
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry and dependent entries."""
        with self._lock:
            if key not in self.cache:
                return False
            
            # Find dependent entries
            dependent_keys = []
            for cache_key, entry in self.cache.items():
                if key in entry.dependencies:
                    dependent_keys.append(cache_key)
            
            # Remove main entry
            entry = self.cache.pop(key)
            self.size_bytes -= entry.size_bytes
            
            # Remove dependent entries
            for dep_key in dependent_keys:
                if dep_key in self.cache:
                    dep_entry = self.cache.pop(dep_key)
                    self.size_bytes -= dep_entry.size_bytes
                    logger.debug(f"Invalidated dependent cache entry: {dep_key}")
            
            logger.debug(f"Invalidated cache entry {key} and {len(dependent_keys)} dependencies")
            return True
    
    def _evict_item(self) -> bool:
        """Evict item using quantum-inspired algorithm."""
        if not self.cache:
            return False
        
        if self.eviction_algorithm == "quantum_lru":
            return self._evict_quantum_lru()
        elif self.eviction_algorithm == "quantum_lfu":
            return self._evict_quantum_lfu()
        elif self.eviction_algorithm == "quantum_priority":
            return self._evict_quantum_priority()
        else:
            # Fallback to simple LRU
            key, entry = self.cache.popitem(last=False)
            self.size_bytes -= entry.size_bytes
            return True
    
    def _evict_quantum_lru(self) -> bool:
        """Evict using quantum-weighted LRU algorithm."""
        if not self.cache:
            return False
        
        # Calculate quantum-weighted scores for eviction
        eviction_scores = []
        
        for key, entry in self.cache.items():
            # Base LRU score (older = higher eviction score)
            age_score = (datetime.now() - entry.last_accessed).total_seconds()
            
            # Quantum coherence factor (lower coherence = higher eviction score)
            coherence_factor = 1.0 / (entry.quantum_coherence + 0.01)
            
            # Priority factor (lower priority = higher eviction score)
            priority_factor = 1.0 / (entry.priority_score + 0.01)
            
            # Combined quantum score
            quantum_score = age_score * coherence_factor * priority_factor
            eviction_scores.append((quantum_score, key, entry))
        
        # Sort by eviction score (highest first)
        eviction_scores.sort(reverse=True)
        
        # Evict highest scoring item
        _, key, entry = eviction_scores[0]
        del self.cache[key]
        self.size_bytes -= entry.size_bytes
        
        logger.debug(f"Quantum LRU evicted: {key}")
        return True
    
    def _evict_quantum_lfu(self) -> bool:
        """Evict using quantum-weighted LFU algorithm."""
        if not self.cache:
            return False
        
        eviction_scores = []
        
        for key, entry in self.cache.items():
            # Base LFU score (lower frequency = higher eviction score)
            frequency_score = 1.0 / (entry.access_count + 1)
            
            # Quantum coherence factor
            coherence_factor = 1.0 / (entry.quantum_coherence + 0.01)
            
            quantum_score = frequency_score * coherence_factor
            eviction_scores.append((quantum_score, key, entry))
        
        eviction_scores.sort(reverse=True)
        _, key, entry = eviction_scores[0]
        del self.cache[key]
        self.size_bytes -= entry.size_bytes
        
        return True
    
    def _evict_quantum_priority(self) -> bool:
        """Evict using quantum priority-based algorithm."""
        if not self.cache:
            return False
        
        # Find lowest priority item with consideration for quantum coherence
        min_score = float('inf')
        evict_key = None
        evict_entry = None
        
        for key, entry in self.cache.items():
            # Combined score: priority + coherence + recency
            recency_factor = (datetime.now() - entry.last_accessed).total_seconds() / 3600  # Hours
            combined_score = entry.priority_score * entry.quantum_coherence / (1 + recency_factor)
            
            if combined_score < min_score:
                min_score = combined_score
                evict_key = key
                evict_entry = entry
        
        if evict_key:
            del self.cache[evict_key]
            self.size_bytes -= evict_entry.size_bytes
            return True
        
        return False
    
    def _record_access(self, key: str) -> None:
        """Record access pattern for predictive caching."""
        now = datetime.now()
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(now)
        
        # Keep only recent access history
        cutoff = now - timedelta(hours=24)
        self.access_patterns[key] = [
            access_time for access_time in self.access_patterns[key]
            if access_time > cutoff
        ]
    
    def update_quantum_coherence(self, quantum_state_change: float) -> None:
        """Update cache coherence based on quantum state changes."""
        with self._lock:
            coherence_decay = abs(quantum_state_change) * 0.1
            
            for entry in self.cache.values():
                entry.quantum_coherence = max(0.0, entry.quantum_coherence - coherence_decay)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "size_bytes": self.size_bytes,
                "hit_rate": self._calculate_hit_rate(),
                "average_coherence": self._calculate_average_coherence(),
                "eviction_algorithm": self.eviction_algorithm,
                "oldest_entry_age": self._get_oldest_entry_age(),
                "access_patterns_tracked": len(self.access_patterns)
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate based on access patterns."""
        total_accesses = sum(
            len(accesses) for accesses in self.access_patterns.values()
        )
        
        cache_hits = sum(
            entry.access_count for entry in self.cache.values()
        )
        
        return cache_hits / total_accesses if total_accesses > 0 else 0.0
    
    def _calculate_average_coherence(self) -> float:
        """Calculate average quantum coherence of cached items."""
        if not self.cache:
            return 0.0
        
        return sum(entry.quantum_coherence for entry in self.cache.values()) / len(self.cache)
    
    def _get_oldest_entry_age(self) -> float:
        """Get age of oldest cache entry in hours."""
        if not self.cache:
            return 0.0
        
        oldest_time = min(entry.created_at for entry in self.cache.values())
        return (datetime.now() - oldest_time).total_seconds() / 3600


class PredictiveCache:
    """Predictive caching system using machine learning for quantum operations."""
    
    def __init__(self, base_cache: QuantumInspiredCache):
        self.base_cache = base_cache
        self.prediction_model = None  # Would be ML model in production
        self.prediction_accuracy = 0.0
        self.prefetch_queue: List[str] = []
        
    def predict_future_accesses(self, window_minutes: int = 60) -> List[str]:
        """Predict which cache keys will be accessed in the near future."""
        # Simplified predictive algorithm
        # In production, this would use ML models
        
        predictions = []
        current_time = datetime.now()
        
        for key, access_times in self.base_cache.access_patterns.items():
            if not access_times:
                continue
            
            # Calculate access frequency
            recent_accesses = [
                t for t in access_times
                if (current_time - t).total_seconds() < 3600  # Last hour
            ]
            
            frequency = len(recent_accesses) / max(1, len(access_times))
            
            # Predict based on frequency and recency
            if frequency > 0.3 and recent_accesses:  # High frequency
                time_since_last = (current_time - recent_accesses[-1]).total_seconds()
                if time_since_last > 600:  # 10 minutes since last access
                    predictions.append(key)
        
        return predictions[:10]  # Limit predictions
    
    def prefetch_predicted_items(self, prefetch_function: Callable[[str], Any]) -> int:
        """Prefetch items that are predicted to be accessed soon."""
        predictions = self.predict_future_accesses()
        prefetched_count = 0
        
        for key in predictions:
            if key not in self.base_cache.cache:
                try:
                    # Generate value using provided function
                    value = prefetch_function(key)
                    if value is not None:
                        # Cache with lower priority (prefetched items)
                        self.base_cache.put(key, value, priority_score=0.5)
                        prefetched_count += 1
                        logger.debug(f"Prefetched cache item: {key}")
                except Exception as e:
                    logger.warning(f"Prefetch failed for {key}: {e}")
        
        return prefetched_count


class DistributedQuantumCache:
    """Distributed caching system for scalable quantum task planning."""
    
    def __init__(self, node_id: str, max_local_size: int = 500):
        self.node_id = node_id
        self.local_cache = QuantumInspiredCache(max_size=max_local_size)
        self.peer_nodes: Dict[str, Any] = {}  # Would be network connections in production
        self.replication_factor = 2
        self.consistency_level = "eventual"  # eventual, strong, quantum
        
    def get_distributed(self, key: str) -> Optional[Any]:
        """Get item from distributed cache with quantum consistency."""
        # Try local cache first
        local_value = self.local_cache.get(key)
        if local_value is not None:
            return local_value
        
        # Try peer nodes
        for peer_id, peer_connection in self.peer_nodes.items():
            try:
                remote_value = self._fetch_from_peer(peer_id, key)
                if remote_value is not None:
                    # Cache locally with quantum entanglement to remote
                    self.local_cache.put(key, remote_value, priority_score=0.8)
                    return remote_value
            except Exception as e:
                logger.warning(f"Failed to fetch {key} from peer {peer_id}: {e}")
        
        return None
    
    def put_distributed(self, key: str, value: Any, priority_score: float = 1.0) -> bool:
        """Put item in distributed cache with replication."""
        # Store locally
        local_success = self.local_cache.put(key, value, priority_score)
        
        # Replicate to peer nodes
        replication_count = 0
        target_replications = min(self.replication_factor, len(self.peer_nodes))
        
        for peer_id in list(self.peer_nodes.keys())[:target_replications]:
            try:
                if self._replicate_to_peer(peer_id, key, value):
                    replication_count += 1
            except Exception as e:
                logger.warning(f"Replication to {peer_id} failed: {e}")
        
        logger.debug(f"Cached {key} locally and replicated to {replication_count} peers")
        return local_success
    
    def _fetch_from_peer(self, peer_id: str, key: str) -> Optional[Any]:
        """Fetch value from peer node."""
        # Simulated peer fetch - in production would use network protocols
        # Return None for now since no actual peers are connected
        return None
    
    def _replicate_to_peer(self, peer_id: str, key: str, value: Any) -> bool:
        """Replicate value to peer node."""
        # Simulated replication - in production would use network protocols
        return True
    
    def synchronize_quantum_state(self) -> None:
        """Synchronize quantum state across distributed cache nodes."""
        # Quantum state synchronization protocol
        # In production, this would implement quantum state consensus
        
        local_coherence = self.local_cache._calculate_average_coherence()
        
        # Broadcast coherence state to peers
        for peer_id in self.peer_nodes:
            try:
                self._broadcast_coherence_state(peer_id, local_coherence)
            except Exception as e:
                logger.warning(f"Coherence sync to {peer_id} failed: {e}")
    
    def _broadcast_coherence_state(self, peer_id: str, coherence: float) -> None:
        """Broadcast quantum coherence state to peer."""
        # Simulated coherence broadcast
        pass


class CacheOptimizer:
    """Optimizes cache performance using quantum-inspired algorithms."""
    
    def __init__(self, cache: QuantumInspiredCache):
        self.cache = cache
        self.optimization_history: List[Dict] = []
        
    def optimize_cache_parameters(self) -> Dict[str, Any]:
        """Optimize cache parameters using quantum algorithms."""
        optimization_start = time.time()
        
        current_stats = self.cache.get_cache_stats()
        
        # Analyze access patterns
        access_analysis = self._analyze_access_patterns()
        
        # Optimize eviction algorithm
        optimal_algorithm = self._select_optimal_eviction_algorithm()
        
        # Optimize cache size
        optimal_size = self._calculate_optimal_cache_size()
        
        # Apply optimizations
        changes_made = []
        
        if optimal_algorithm != self.cache.eviction_algorithm:
            self.cache.eviction_algorithm = optimal_algorithm
            changes_made.append(f"Changed eviction algorithm to {optimal_algorithm}")
        
        if optimal_size != self.cache.max_size:
            self.cache.max_size = optimal_size
            changes_made.append(f"Adjusted cache size to {optimal_size}")
        
        optimization_time = time.time() - optimization_start
        
        # Record optimization
        optimization_record = {
            "timestamp": datetime.now(),
            "optimization_time": optimization_time,
            "changes_made": changes_made,
            "performance_before": current_stats,
            "access_analysis": access_analysis
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"Cache optimization completed in {optimization_time:.2f}s: {changes_made}")
        
        return optimization_record
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze cache access patterns for optimization insights."""
        patterns = self.cache.access_patterns
        
        if not patterns:
            return {"pattern_type": "insufficient_data"}
        
        # Calculate access frequency distribution
        frequencies = [len(accesses) for accesses in patterns.values()]
        
        if frequencies:
            avg_frequency = sum(frequencies) / len(frequencies)
            max_frequency = max(frequencies)
            
            # Determine pattern type
            if max_frequency > avg_frequency * 3:
                pattern_type = "hotspot"  # Few heavily accessed items
            elif max(frequencies) - min(frequencies) < 2:
                pattern_type = "uniform"   # Even access distribution
            else:
                pattern_type = "mixed"     # Mixed access pattern
        else:
            pattern_type = "unknown"
        
        return {
            "pattern_type": pattern_type,
            "unique_keys": len(patterns),
            "average_frequency": avg_frequency if frequencies else 0,
            "max_frequency": max_frequency if frequencies else 0,
            "total_accesses": sum(frequencies)
        }
    
    def _select_optimal_eviction_algorithm(self) -> str:
        """Select optimal eviction algorithm based on access patterns."""
        analysis = self._analyze_access_patterns()
        
        pattern_type = analysis.get("pattern_type", "unknown")
        
        # Choose algorithm based on access patterns
        if pattern_type == "hotspot":
            return "quantum_lfu"  # Frequency-based for hotspot patterns
        elif pattern_type == "uniform":
            return "quantum_lru"  # Recency-based for uniform patterns
        elif pattern_type == "mixed":
            return "quantum_priority"  # Priority-based for mixed patterns
        else:
            return "quantum_lru"  # Default
    
    def _calculate_optimal_cache_size(self) -> int:
        """Calculate optimal cache size based on usage patterns."""
        current_stats = self.cache.get_cache_stats()
        
        hit_rate = current_stats.get("hit_rate", 0.0)
        current_size = current_stats.get("size", 0)
        max_size = current_stats.get("max_size", 1000)
        
        # If hit rate is low and cache is full, increase size
        if hit_rate < 0.7 and current_size >= max_size * 0.9:
            return min(max_size * 2, 5000)  # Double size, max 5000
        
        # If hit rate is very high and cache is not full, could reduce size
        elif hit_rate > 0.95 and current_size < max_size * 0.5:
            return max(max_size // 2, 100)  # Half size, min 100
        
        return max_size  # Keep current size


# Global cache instances for quantum operations
quantum_cache = QuantumInspiredCache(max_size=1000, ttl_seconds=3600)
predictive_cache = PredictiveCache(quantum_cache)
cache_optimizer = CacheOptimizer(quantum_cache)


def cached_quantum_operation(cache_key_func: Optional[Callable] = None, 
                           ttl_seconds: int = 3600, priority: float = 1.0):
    """Decorator for caching quantum operation results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = f"{func.__name__}_{args}_{sorted(kwargs.items())}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache first
            cached_result = quantum_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            quantum_cache.put(cache_key, result, priority_score=priority)
            
            return result
        
        return wrapper
    return decorator