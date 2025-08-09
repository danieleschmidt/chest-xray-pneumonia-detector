"""Intelligent caching system with adaptive policies."""

import time
import hashlib
import pickle
from typing import Any, Optional, Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
from abc import ABC, abstractmethod


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def idle_seconds(self) -> float:
        """Get time since last access in seconds."""
        return (datetime.utcnow() - self.last_accessed).total_seconds()


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        """Select cache entries for eviction.
        
        Args:
            entries: Dictionary of cache entries
            target_count: Number of entries to evict
            
        Returns:
            List of keys to evict
        """
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        # Sort by last accessed time, oldest first
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].last_accessed)
        return [key for key, _ in sorted_entries[:target_count]]


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        # Sort by access count, lowest first
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].access_count)
        return [key for key, _ in sorted_entries[:target_count]]


class IntelligentEvictionPolicy(EvictionPolicy):
    """Intelligent eviction policy based on multiple factors."""
    
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        """Select victims based on composite score of recency, frequency, and size."""
        now = datetime.utcnow()
        
        # Calculate scores for each entry
        scored_entries = []
        for key, entry in entries.items():
            # Factors: recency (lower is better), frequency (higher is better), size (higher is worse)
            recency_score = entry.idle_seconds / 3600.0  # Hours since last access
            frequency_score = 1.0 / max(1, entry.access_count)  # Inverse frequency
            size_score = entry.size_bytes / (1024 * 1024)  # Size in MB
            
            # Composite score (higher = more likely to evict)
            composite_score = recency_score + frequency_score + size_score * 0.1
            
            scored_entries.append((key, composite_score))
        
        # Sort by score, highest first (most likely to evict)
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        return [key for key, _ in scored_entries[:target_count]]


class IntelligentCache:
    """High-performance intelligent cache with adaptive policies."""
    
    def __init__(
        self,
        max_size_bytes: int = 512 * 1024 * 1024,  # 512MB default
        max_entries: int = 10000,
        default_ttl_seconds: Optional[int] = 3600,  # 1 hour default
        eviction_policy: Optional[EvictionPolicy] = None
    ):
        """Initialize intelligent cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of cache entries
            default_ttl_seconds: Default TTL for entries
            eviction_policy: Eviction policy to use
        """
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.eviction_policy = eviction_policy or IntelligentEvictionPolicy()
        
        self._entries: Dict[str, CacheEntry] = {}
        self._current_size_bytes = 0
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._stats['misses'] += 1
                self._stats['expired_evictions'] += 1
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            
            self._stats['hits'] += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate size
            try:
                serialized = pickle.dumps(value)
                size_bytes = len(serialized) + len(key.encode('utf-8'))
            except Exception:
                return False
            
            # Check if single entry exceeds max size
            if size_bytes > self.max_size_bytes:
                return False
            
            now = datetime.utcnow()
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl
            )
            
            # Remove existing entry if present
            if key in self._entries:
                self._remove_entry(key)
            
            # Make space if needed
            self._ensure_space(size_bytes)
            
            # Add entry
            self._entries[key] = entry
            self._current_size_bytes += size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(1, total_requests)
            
            return {
                'entries': len(self._entries),
                'size_bytes': self._current_size_bytes,
                'size_mb': round(self._current_size_bytes / (1024 * 1024), 2),
                'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
                'utilization_percent': round(
                    (self._current_size_bytes / self.max_size_bytes) * 100, 2
                ),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': round(hit_rate, 4),
                'evictions': self._stats['evictions'],
                'expired_evictions': self._stats['expired_evictions']
            }
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Ensure sufficient space in cache."""
        # Clean up expired entries first
        self._cleanup_expired()
        
        # Check if we need to evict entries
        while (
            self._current_size_bytes + required_bytes > self.max_size_bytes or
            len(self._entries) >= self.max_entries
        ):
            if not self._entries:
                break
            
            # Calculate how many entries to evict
            if self._current_size_bytes + required_bytes > self.max_size_bytes:
                # Evict based on size pressure
                target_count = max(1, len(self._entries) // 10)  # Evict 10%
            else:
                # Evict single entry for count pressure
                target_count = 1
            
            victims = self.eviction_policy.select_victims(self._entries, target_count)
            if not victims:
                break
            
            for victim_key in victims:
                self._remove_entry(victim_key)
                self._stats['evictions'] += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._entries:
            entry = self._entries[key]
            self._current_size_bytes -= entry.size_bytes
            del self._entries[key]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
            self._stats['expired_evictions'] += 1


class ModelCache:
    """Specialized cache for ML model predictions."""
    
    def __init__(self, max_size_mb: int = 256):
        """Initialize model prediction cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache = IntelligentCache(
            max_size_bytes=max_size_mb * 1024 * 1024,
            default_ttl_seconds=1800,  # 30 minutes
            eviction_policy=IntelligentEvictionPolicy()
        )
    
    def get_prediction_key(self, image_hash: str, model_version: str) -> str:
        """Generate cache key for prediction.
        
        Args:
            image_hash: Hash of input image
            model_version: Model version identifier
            
        Returns:
            Cache key string
        """
        return f"pred:{image_hash}:{model_version}"
    
    def get_prediction(self, image_hash: str, model_version: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction result.
        
        Args:
            image_hash: Hash of input image
            model_version: Model version identifier
            
        Returns:
            Cached prediction result or None
        """
        key = self.get_prediction_key(image_hash, model_version)
        return self.cache.get(key)
    
    def cache_prediction(
        self,
        image_hash: str,
        model_version: str,
        prediction_result: Dict[str, Any]
    ) -> bool:
        """Cache prediction result.
        
        Args:
            image_hash: Hash of input image
            model_version: Model version identifier
            prediction_result: Prediction result to cache
            
        Returns:
            True if successfully cached
        """
        key = self.get_prediction_key(image_hash, model_version)
        return self.cache.set(key, prediction_result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# Global model cache instance
model_cache = ModelCache()