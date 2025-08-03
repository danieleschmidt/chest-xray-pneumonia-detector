"""
Caching layer for the pneumonia detection system.
Implements Redis-based caching with fallback to in-memory cache.
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from functools import wraps
import hashlib

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache management."""
    enabled: bool = True
    default_ttl: int = 3600  # 1 hour
    max_memory_items: int = 1000
    redis_url: Optional[str] = None
    key_prefix: str = "pneumonia_detector"
    compression_enabled: bool = True
    serialization_method: str = "pickle"  # pickle, json


class InMemoryCache:
    """Thread-safe in-memory cache with LRU eviction."""
    
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry["expires_at"] and time.time() > entry["expires_at"]:
                    await self._delete_key(key)
                    return None
                
                # Update access time for LRU
                self._access_times[key] = time.time()
                return entry["value"]
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        async with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = time.time() + ttl
            
            # Evict if at capacity
            if len(self._cache) >= self.max_items and key not in self._cache:
                await self._evict_lru()
            
            # Store entry
            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": time.time()
            }
            self._access_times[key] = time.time()
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        async with self._lock:
            return await self._delete_key(key)
    
    async def _delete_key(self, key: str) -> bool:
        """Internal key deletion."""
        if key in self._cache:
            del self._cache[key]
            self._access_times.pop(key, None)
            return True
        return False
    
    async def _evict_lru(self):
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=self._access_times.get)
        await self._delete_key(lru_key)
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            return True
    
    async def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._cache.keys())


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        try:
            data = await self.redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in Redis cache."""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries (use with caution)."""
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {str(e)}")
            return False
    
    async def size(self) -> int:
        """Get current cache size."""
        try:
            return await self.redis.dbsize()
        except Exception as e:
            logger.error(f"Redis size error: {str(e)}")
            return 0
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get cache keys matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {str(e)}")
            return []


class CacheManager:
    """
    Unified cache manager with Redis and in-memory fallback.
    Provides caching for predictions, model outputs, and metadata.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_cache: Optional[RedisCache] = None
        self.memory_cache = InMemoryCache(config.max_memory_items)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize cache connections."""
        if not self.config.enabled:
            logger.info("Caching is disabled")
            return
        
        # Try to initialize Redis if URL provided
        if self.config.redis_url and REDIS_AVAILABLE:
            try:
                redis_client = redis.from_url(self.config.redis_url)
                await redis_client.ping()
                self.redis_cache = RedisCache(redis_client)
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis initialization failed, using memory cache: {str(e)}")
        
        self._initialized = True
        logger.info("Cache manager initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (Redis first, then memory)."""
        if not self.config.enabled or not self._initialized:
            return None
        
        full_key = self._build_key(key)
        
        # Try Redis first
        if self.redis_cache:
            value = await self.redis_cache.get(full_key)
            if value is not None:
                return value
        
        # Fallback to memory cache
        return await self.memory_cache.get(full_key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        if not self.config.enabled or not self._initialized:
            return False
        
        full_key = self._build_key(key)
        ttl = ttl or self.config.default_ttl
        
        # Set in both caches
        redis_success = True
        memory_success = True
        
        if self.redis_cache:
            redis_success = await self.redis_cache.set(full_key, value, ttl)
        
        memory_success = await self.memory_cache.set(full_key, value, ttl)
        
        return redis_success or memory_success
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        if not self.config.enabled or not self._initialized:
            return False
        
        full_key = self._build_key(key)
        
        # Delete from both caches
        redis_deleted = True
        memory_deleted = True
        
        if self.redis_cache:
            redis_deleted = await self.redis_cache.delete(full_key)
        
        memory_deleted = await self.memory_cache.delete(full_key)
        
        return redis_deleted or memory_deleted
    
    async def get_or_set(
        self, 
        key: str, 
        factory: Callable, 
        ttl: Optional[int] = None,
        *args,
        **kwargs
    ) -> Any:
        """Get from cache or generate and cache the value."""
        value = await self.get(key)
        
        if value is None:
            # Generate value
            if asyncio.iscoroutinefunction(factory):
                value = await factory(*args, **kwargs)
            else:
                value = factory(*args, **kwargs)
            
            # Cache the generated value
            await self.set(key, value, ttl)
        
        return value
    
    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix."""
        return f"{self.config.key_prefix}:{key}"
    
    async def cache_prediction(
        self,
        image_hash: str,
        model_version: str,
        prediction_result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache prediction result."""
        cache_key = f"prediction:{image_hash}:{model_version}"
        return await self.set(cache_key, prediction_result, ttl)
    
    async def get_cached_prediction(
        self,
        image_hash: str,
        model_version: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached prediction result."""
        cache_key = f"prediction:{image_hash}:{model_version}"
        return await self.get(cache_key)
    
    async def cache_model_metadata(
        self,
        model_version: str,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache model metadata."""
        cache_key = f"model_metadata:{model_version}"
        return await self.set(cache_key, metadata, ttl)
    
    async def get_model_metadata(self, model_version: str) -> Optional[Dict[str, Any]]:
        """Get cached model metadata."""
        cache_key = f"model_metadata:{model_version}"
        return await self.get(cache_key)
    
    async def cache_image_features(
        self,
        image_hash: str,
        features: np.ndarray,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache extracted image features."""
        cache_key = f"features:{image_hash}"
        # Convert numpy array to list for serialization
        features_list = features.tolist() if isinstance(features, np.ndarray) else features
        return await self.set(cache_key, features_list, ttl)
    
    async def get_image_features(self, image_hash: str) -> Optional[np.ndarray]:
        """Get cached image features."""
        cache_key = f"features:{image_hash}"
        features = await self.get(cache_key)
        if features:
            return np.array(features) if isinstance(features, list) else features
        return None
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        count = 0
        
        # Redis pattern invalidation
        if self.redis_cache:
            keys = await self.redis_cache.keys(f"{self.config.key_prefix}:{pattern}")
            for key in keys:
                if await self.redis_cache.delete(key):
                    count += 1
        
        # Memory cache pattern invalidation (simple prefix matching)
        memory_keys = await self.memory_cache.keys()
        pattern_key = self._build_key(pattern.replace("*", ""))
        
        for key in memory_keys:
            if key.startswith(pattern_key):
                if await self.memory_cache.delete(key):
                    count += 1
        
        return count
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "enabled": self.config.enabled,
            "redis_available": self.redis_cache is not None,
            "memory_cache_size": await self.memory_cache.size(),
            "memory_cache_max_items": self.config.max_memory_items
        }
        
        if self.redis_cache:
            stats["redis_cache_size"] = await self.redis_cache.size()
        
        return stats
    
    async def clear_all(self) -> bool:
        """Clear all caches (use with extreme caution)."""
        redis_cleared = True
        memory_cleared = True
        
        if self.redis_cache:
            redis_cleared = await self.redis_cache.clear()
        
        memory_cleared = await self.memory_cache.clear()
        
        return redis_cleared and memory_cleared


def cache_key_from_args(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create a hash of all arguments
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    cache_instance: Optional[CacheManager] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        cache_instance: Cache manager instance (if None, uses global instance)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not cache_instance or not cache_instance.config.enabled:
                # No caching, call function directly
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            # Generate cache key
            func_name = func.__name__
            args_key = cache_key_from_args(*args, **kwargs)
            cache_key = f"{key_prefix}:{func_name}:{args_key}" if key_prefix else f"{func_name}:{args_key}"
            
            # Try to get from cache
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache_instance.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator