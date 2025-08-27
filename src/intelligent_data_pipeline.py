#!/usr/bin/env python3
"""
Intelligent Data Pipeline for Medical AI
Progressive Enhancement - Generation 1: Smart Data Processing
"""

import asyncio
import json
import logging
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, AsyncIterator
import numpy as np

class DataQuality(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

class ProcessingStage(Enum):
    """Data processing pipeline stages"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    AUGMENTATION = "augmentation"
    QUALITY_CHECK = "quality_check"
    EXPORT = "export"

@dataclass
class DataSample:
    """Individual data sample with metadata"""
    id: str
    file_path: Path
    label: str
    quality: DataQuality
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_history: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum and self.file_path.exists():
            self.checksum = self._calculate_checksum()
            
    def _calculate_checksum(self) -> str:
        """Calculate file checksum for integrity verification"""
        try:
            with open(self.file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return ""

@dataclass
class PipelineMetrics:
    """Pipeline performance and quality metrics"""
    samples_processed: int = 0
    samples_rejected: int = 0
    processing_time_ms: float = 0.0
    quality_distribution: Dict[DataQuality, int] = field(default_factory=dict)
    error_count: int = 0
    throughput_per_second: float = 0.0
    memory_usage_mb: float = 0.0

class IntelligentDataPipeline:
    """
    Intelligent data pipeline with adaptive processing and quality assurance.
    
    Features:
    - Automatic data quality assessment
    - Adaptive preprocessing based on data characteristics
    - Real-time performance monitoring
    - Integrity verification with checksums
    - Smart caching and batch optimization
    """
    
    def __init__(self, base_path: str = "data_pipeline", batch_size: int = 32):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.batch_size = batch_size
        self.samples: Dict[str, DataSample] = {}
        self.metrics = PipelineMetrics()
        self.quality_thresholds = {
            DataQuality.EXCELLENT: 0.95,
            DataQuality.GOOD: 0.85,
            DataQuality.FAIR: 0.70,
            DataQuality.POOR: 0.50
        }
        
        self.logger = self._setup_logging()
        self.cache_dir = self.base_path / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Processing stages configuration
        self.enabled_stages = list(ProcessingStage)
        self.stage_configs = self._initialize_stage_configs()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data pipeline"""
        logger = logging.getLogger("IntelligentDataPipeline")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # File handler for pipeline logs
        log_file = self.base_path / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def _initialize_stage_configs(self) -> Dict[ProcessingStage, Dict[str, Any]]:
        """Initialize configuration for each processing stage"""
        return {
            ProcessingStage.INGESTION: {
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
                "max_file_size_mb": 50,
                "parallel_workers": 4
            },
            ProcessingStage.VALIDATION: {
                "min_image_size": (64, 64),
                "max_image_size": (4096, 4096),
                "required_channels": [1, 3],
                "check_corruption": True
            },
            ProcessingStage.PREPROCESSING: {
                "target_size": (224, 224),
                "normalize": True,
                "denoise": True,
                "histogram_equalization": True
            },
            ProcessingStage.AUGMENTATION: {
                "rotation_range": 15,
                "width_shift_range": 0.1,
                "height_shift_range": 0.1,
                "brightness_range": [0.8, 1.2],
                "zoom_range": 0.1
            },
            ProcessingStage.QUALITY_CHECK: {
                "blur_threshold": 100,
                "contrast_threshold": 0.2,
                "brightness_range": [0.1, 0.9],
                "noise_threshold": 0.15
            }
        }
        
    async def ingest_data(self, data_sources: List[str]) -> int:
        """Ingest data from multiple sources"""
        self.logger.info(f"Starting data ingestion from {len(data_sources)} sources")
        start_time = time.time()
        
        ingested_count = 0
        
        for source in data_sources:
            source_path = Path(source)
            
            if source_path.is_dir():
                # Process directory
                count = await self._ingest_directory(source_path)
                ingested_count += count
            elif source_path.is_file():
                # Process single file
                sample = await self._ingest_file(source_path)
                if sample:
                    ingested_count += 1
                    
        processing_time = (time.time() - start_time) * 1000
        self.metrics.processing_time_ms = processing_time
        self.metrics.samples_processed = ingested_count
        
        self.logger.info(f"Ingestion complete: {ingested_count} samples in {processing_time:.2f}ms")
        return ingested_count
        
    async def _ingest_directory(self, directory: Path) -> int:
        """Ingest all valid files from a directory"""
        supported_formats = self.stage_configs[ProcessingStage.INGESTION]["supported_formats"]
        count = 0
        
        # Find all image files
        for ext in supported_formats:
            for file_path in directory.rglob(f"*{ext}"):
                sample = await self._ingest_file(file_path)
                if sample:
                    count += 1
                    
        return count
        
    async def _ingest_file(self, file_path: Path) -> Optional[DataSample]:
        """Ingest a single file"""
        try:
            # Validate file
            if not await self._validate_file(file_path):
                self.metrics.samples_rejected += 1
                return None
                
            # Determine label from directory structure
            label = file_path.parent.name.upper()
            if label not in ["NORMAL", "PNEUMONIA"]:
                label = "UNKNOWN"
                
            # Create data sample
            sample_id = f"{file_path.stem}_{int(time.time() * 1000)}"
            sample = DataSample(
                id=sample_id,
                file_path=file_path,
                label=label,
                quality=DataQuality.GOOD,  # Will be assessed later
                metadata={
                    "original_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "ingestion_time": datetime.now().isoformat()
                }
            )
            
            sample.processing_history.append("ingested")
            self.samples[sample_id] = sample
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Failed to ingest {file_path}: {e}")
            self.metrics.error_count += 1
            return None
            
    async def _validate_file(self, file_path: Path) -> bool:
        """Validate file meets requirements"""
        config = self.stage_configs[ProcessingStage.VALIDATION]
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config["max_file_size_mb"]:
            self.logger.warning(f"File too large: {file_path} ({file_size_mb:.1f}MB)")
            return False
            
        # Check file format
        if file_path.suffix.lower() not in self.stage_configs[ProcessingStage.INGESTION]["supported_formats"]:
            return False
            
        # Mock image validation (in production, would use actual image processing)
        # Simulate image loading and basic checks
        await asyncio.sleep(0.01)  # Mock processing time
        
        return True
        
    async def assess_quality(self, sample_ids: Optional[List[str]] = None) -> Dict[str, DataQuality]:
        """Assess quality of data samples using intelligent algorithms"""
        if sample_ids is None:
            sample_ids = list(self.samples.keys())
            
        self.logger.info(f"Assessing quality for {len(sample_ids)} samples")
        quality_results = {}
        
        for sample_id in sample_ids:
            if sample_id not in self.samples:
                continue
                
            sample = self.samples[sample_id]
            
            try:
                # Intelligent quality assessment
                quality_score = await self._calculate_quality_score(sample)
                quality_level = self._score_to_quality_level(quality_score)
                
                # Update sample
                sample.quality = quality_level
                sample.metadata["quality_score"] = quality_score
                sample.processing_history.append("quality_assessed")
                
                quality_results[sample_id] = quality_level
                
                # Update metrics
                if quality_level not in self.metrics.quality_distribution:
                    self.metrics.quality_distribution[quality_level] = 0
                self.metrics.quality_distribution[quality_level] += 1
                
            except Exception as e:
                self.logger.error(f"Quality assessment failed for {sample_id}: {e}")
                quality_results[sample_id] = DataQuality.INVALID
                self.metrics.error_count += 1
                
        return quality_results
        
    async def _calculate_quality_score(self, sample: DataSample) -> float:
        """Calculate intelligent quality score for a sample"""
        # Mock quality assessment - in production would analyze actual image
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Simulate quality factors
        blur_score = np.random.uniform(0.7, 1.0)
        contrast_score = np.random.uniform(0.6, 1.0)
        noise_score = np.random.uniform(0.8, 1.0)
        composition_score = np.random.uniform(0.75, 0.95)
        
        # Weighted quality score
        weights = [0.3, 0.25, 0.25, 0.2]
        scores = [blur_score, contrast_score, noise_score, composition_score]
        
        quality_score = sum(w * s for w, s in zip(weights, scores))
        
        # Add metadata
        sample.metadata.update({
            "blur_score": blur_score,
            "contrast_score": contrast_score,
            "noise_score": noise_score,
            "composition_score": composition_score
        })
        
        return quality_score
        
    def _score_to_quality_level(self, score: float) -> DataQuality:
        """Convert quality score to quality level"""
        for quality, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return quality
        return DataQuality.INVALID
        
    async def preprocess_batch(self, sample_ids: List[str]) -> List[str]:
        """Preprocess a batch of samples with intelligent optimization"""
        self.logger.info(f"Preprocessing batch of {len(sample_ids)} samples")
        start_time = time.time()
        
        processed_ids = []
        
        # Group by quality for optimized processing
        quality_groups = self._group_by_quality(sample_ids)
        
        for quality, ids in quality_groups.items():
            if quality == DataQuality.INVALID:
                continue  # Skip invalid samples
                
            # Apply quality-specific preprocessing
            for sample_id in ids:
                try:
                    success = await self._preprocess_sample(sample_id, quality)
                    if success:
                        processed_ids.append(sample_id)
                        
                except Exception as e:
                    self.logger.error(f"Preprocessing failed for {sample_id}: {e}")
                    self.metrics.error_count += 1
                    
        processing_time = (time.time() - start_time) * 1000
        throughput = len(processed_ids) / (processing_time / 1000) if processing_time > 0 else 0
        
        self.metrics.throughput_per_second = throughput
        self.logger.info(f"Batch preprocessing complete: {len(processed_ids)} samples, "
                        f"{throughput:.1f} samples/sec")
        
        return processed_ids
        
    def _group_by_quality(self, sample_ids: List[str]) -> Dict[DataQuality, List[str]]:
        """Group samples by quality level for optimized processing"""
        groups = {}
        
        for sample_id in sample_ids:
            if sample_id in self.samples:
                quality = self.samples[sample_id].quality
                if quality not in groups:
                    groups[quality] = []
                groups[quality].append(sample_id)
                
        return groups
        
    async def _preprocess_sample(self, sample_id: str, quality: DataQuality) -> bool:
        """Preprocess individual sample with quality-adapted parameters"""
        sample = self.samples[sample_id]
        
        # Mock preprocessing - in production would do actual image processing
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Quality-specific preprocessing
        processing_params = self._get_quality_adapted_params(quality)
        
        # Simulate preprocessing steps
        sample.metadata.update({
            "preprocessing_applied": processing_params,
            "processed_at": datetime.now().isoformat()
        })
        
        sample.processing_history.append("preprocessed")
        return True
        
    def _get_quality_adapted_params(self, quality: DataQuality) -> Dict[str, Any]:
        """Get preprocessing parameters adapted to data quality"""
        base_config = self.stage_configs[ProcessingStage.PREPROCESSING]
        
        if quality == DataQuality.POOR:
            # More aggressive processing for poor quality
            return {
                **base_config,
                "denoise": True,
                "histogram_equalization": True,
                "sharpening": True
            }
        elif quality == DataQuality.EXCELLENT:
            # Minimal processing for excellent quality
            return {
                **base_config,
                "denoise": False,
                "histogram_equalization": False
            }
        else:
            # Standard processing
            return base_config
            
    async def generate_batch_iterator(self, batch_size: int = None) -> AsyncIterator[List[DataSample]]:
        """Generate batches of samples for training"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # Filter valid samples
        valid_samples = [
            sample for sample in self.samples.values()
            if sample.quality != DataQuality.INVALID
        ]
        
        # Sort by quality for better batch composition
        valid_samples.sort(key=lambda s: s.quality.value)
        
        # Generate batches
        for i in range(0, len(valid_samples), batch_size):
            batch = valid_samples[i:i + batch_size]
            self.logger.debug(f"Generated batch {i//batch_size + 1} with {len(batch)} samples")
            yield batch
            
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance metrics"""
        total_samples = len(self.samples)
        
        return {
            "total_samples": total_samples,
            "samples_processed": self.metrics.samples_processed,
            "samples_rejected": self.metrics.samples_rejected,
            "rejection_rate": self.metrics.samples_rejected / total_samples if total_samples > 0 else 0,
            "processing_time_ms": self.metrics.processing_time_ms,
            "throughput_per_second": self.metrics.throughput_per_second,
            "error_count": self.metrics.error_count,
            "quality_distribution": {q.value: count for q, count in self.metrics.quality_distribution.items()},
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "enabled_stages": [stage.value for stage in self.enabled_stages],
            "timestamp": datetime.now().isoformat()
        }
        
    def save_pipeline_state(self, filename: str = "pipeline_state.json"):
        """Save current pipeline state for reproducibility"""
        state = {
            "samples": {
                sid: {
                    "id": sample.id,
                    "file_path": str(sample.file_path),
                    "label": sample.label,
                    "quality": sample.quality.value,
                    "metadata": sample.metadata,
                    "processing_history": sample.processing_history,
                    "created_at": sample.created_at.isoformat(),
                    "checksum": sample.checksum
                }
                for sid, sample in self.samples.items()
            },
            "metrics": {
                "samples_processed": self.metrics.samples_processed,
                "samples_rejected": self.metrics.samples_rejected,
                "processing_time_ms": self.metrics.processing_time_ms,
                "error_count": self.metrics.error_count,
                "quality_distribution": {q.value: count for q, count in self.metrics.quality_distribution.items()}
            },
            "configuration": self.stage_configs,
            "timestamp": datetime.now().isoformat()
        }
        
        state_file = self.base_path / filename
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"Pipeline state saved to {state_file}")
        return state_file


async def demo_intelligent_pipeline():
    """Demonstrate the Intelligent Data Pipeline"""
    print("🔬 Intelligent Data Pipeline Demo")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = IntelligentDataPipeline("demo_pipeline", batch_size=16)
    
    try:
        # Mock data ingestion
        print("\n📥 Ingesting data...")
        mock_sources = ["data_train_engine/train", "data_train_engine/val"]
        ingested_count = await pipeline.ingest_data(mock_sources)
        print(f"✅ Ingested {ingested_count} samples")
        
        # Quality assessment
        print("\n🔍 Assessing data quality...")
        quality_results = await pipeline.assess_quality()
        quality_summary = {}
        for quality in quality_results.values():
            quality_summary[quality.value] = quality_summary.get(quality.value, 0) + 1
        
        print("Quality distribution:")
        for quality, count in quality_summary.items():
            print(f"  {quality}: {count} samples")
            
        # Preprocessing
        print("\n⚙️ Preprocessing samples...")
        sample_ids = list(pipeline.samples.keys())[:10]  # Process first 10 samples
        processed_ids = await pipeline.preprocess_batch(sample_ids)
        print(f"✅ Preprocessed {len(processed_ids)} samples")
        
        # Generate batches
        print("\n📦 Generating training batches...")
        batch_count = 0
        async for batch in pipeline.generate_batch_iterator(batch_size=4):
            batch_count += 1
            if batch_count > 3:  # Limit demo to 3 batches
                break
            print(f"  Batch {batch_count}: {len(batch)} samples")
            
        # Get metrics
        print("\n📊 Pipeline Metrics:")
        metrics = pipeline.get_pipeline_metrics()
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
                
        # Save state
        state_file = pipeline.save_pipeline_state()
        print(f"\n💾 Pipeline state saved to {state_file}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    print("\n🎯 Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_intelligent_pipeline())