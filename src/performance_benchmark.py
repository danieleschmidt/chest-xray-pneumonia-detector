"""Performance benchmarking utilities for training and inference operations."""

import argparse
import time
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
import json
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Local imports
try:
    from .data_loader import create_data_generators
    from .model_builder import (
        create_simple_cnn,
        create_transfer_learning_model,
        create_cnn_with_attention,
    )
    from .inference import predict_directory
except ImportError:
    # Handle import for testing/development
    try:
        from data_loader import create_data_generators
        from model_builder import (
            create_simple_cnn,
            create_transfer_learning_model,
            create_cnn_with_attention,
        )
        from inference import predict_directory
    except ImportError:
        # Fallback for tests
        create_data_generators = None
        create_simple_cnn = None
        create_transfer_learning_model = None
        create_cnn_with_attention = None
        predict_directory = None


@dataclass
class BenchmarkResults:
    """Container for performance benchmark results."""
    
    operation: str
    total_time: float
    peak_memory_mb: float = 0.0
    avg_time_per_epoch: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert results to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(self):
        self.initial_memory_mb = 0
        self.peak_memory_mb = 0
        
    def __enter__(self):
        if PSUTIL_AVAILABLE:
            try:
                self.initial_memory_mb = psutil.virtual_memory().used / 1024 / 1024
                self.peak_memory_mb = self.initial_memory_mb
            except Exception:
                self.initial_memory_mb = 0
                self.peak_memory_mb = 0
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def update_peak(self):
        """Update peak memory if current usage is higher."""
        if PSUTIL_AVAILABLE:
            try:
                current_memory = psutil.virtual_memory().used / 1024 / 1024
                if current_memory > self.peak_memory_mb:
                    self.peak_memory_mb = current_memory
            except Exception:
                pass


@contextmanager
def memory_usage() -> Generator[MemoryTracker, None, None]:
    """Context manager for tracking peak memory usage."""
    tracker = MemoryTracker()
    with tracker:
        yield tracker


def benchmark_training(
    train_dir: str = None,
    val_dir: str = None,
    epochs: int = 3,
    batch_size: int = 32,
    img_size: tuple = (150, 150),
    num_classes: int = 1,
    use_dummy_data: bool = True,
    use_transfer_learning: bool = False,
    use_attention_model: bool = False,
    base_model_name: str = "MobileNetV2",
    **kwargs
) -> BenchmarkResults:
    """Benchmark training performance.
    
    Parameters
    ----------
    train_dir : str, optional
        Training data directory
    val_dir : str, optional  
        Validation data directory
    epochs : int, default=3
        Number of training epochs
    batch_size : int, default=32
        Training batch size
    img_size : tuple, default=(150, 150)
        Input image size
    num_classes : int, default=1
        Number of output classes
    use_dummy_data : bool, default=True
        Use dummy data for benchmarking
    use_transfer_learning : bool, default=False
        Use transfer learning model
    use_attention_model : bool, default=False
        Use attention-based model
    base_model_name : str, default="MobileNetV2"
        Base model for transfer learning
    **kwargs
        Additional arguments
    
    Returns
    -------
    BenchmarkResults
        Benchmark results including timing and memory usage
    """
    start_time = time.time()
    epoch_times = []
    
    with memory_usage() as memory_tracker:
        # Create data generators
        train_gen, val_gen = create_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            use_dummy_data=use_dummy_data,
            img_size=img_size,
            batch_size=batch_size,
        )
        
        # Create model based on configuration
        if use_attention_model:
            model = create_cnn_with_attention(
                input_shape=(*img_size, 3),
                num_classes=num_classes
            )
        elif use_transfer_learning:
            model = create_transfer_learning_model(
                input_shape=(*img_size, 3),
                num_classes=num_classes,
                base_model_name=base_model_name
            )
        else:
            model = create_simple_cnn(
                input_shape=(*img_size, 3),
                num_classes=num_classes
            )
        
        # Calculate total samples for throughput
        total_samples = len(train_gen) * batch_size
        
        # Track epoch timing
        class EpochTimeCallback:
            def __init__(self):
                self.epoch_start_time = None
                self.epoch_times = []
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                memory_tracker.update_peak()
                
            def on_epoch_end(self, epoch, logs=None):
                if self.epoch_start_time:
                    epoch_time = time.time() - self.epoch_start_time
                    self.epoch_times.append(epoch_time)
                memory_tracker.update_peak()
        
        # Use a simpler approach for timing epochs
        for epoch in range(epochs):
            epoch_start = time.time()
            memory_tracker.update_peak()
            
            # Simulate training (in real scenario, this would be model.fit)
            history = model.fit(
                train_gen,
                epochs=1,
                validation_data=val_gen,
                verbose=0
            )
            
            epoch_end = time.time()
            epoch_times.append(epoch_end - epoch_start)
            memory_tracker.update_peak()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    throughput = total_samples * epochs / total_time if total_time > 0 else 0
    
    return BenchmarkResults(
        operation="training",
        total_time=total_time,
        avg_time_per_epoch=avg_time_per_epoch,
        peak_memory_mb=memory_tracker.peak_memory_mb,
        throughput_samples_per_sec=throughput,
        metadata={
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "num_classes": num_classes,
            "use_transfer_learning": use_transfer_learning,
            "use_attention_model": use_attention_model,
            "base_model_name": base_model_name,
            "total_samples": total_samples,
            "use_dummy_data": use_dummy_data,
        }
    )


def benchmark_inference(
    model_path: str,
    data_dir: str,
    num_classes: int = 1,
    img_size: tuple = (150, 150),
    num_samples: int = None,
    **kwargs
) -> BenchmarkResults:
    """Benchmark inference performance.
    
    Parameters
    ----------
    model_path : str
        Path to trained model
    data_dir : str
        Directory containing test images
    num_classes : int, default=1
        Number of output classes
    img_size : tuple, default=(150, 150)
        Input image size
    num_samples : int, optional
        Number of samples to process (None = all)
    **kwargs
        Additional arguments
    
    Returns
    -------
    BenchmarkResults
        Benchmark results including timing and memory usage
    """
    start_time = time.time()
    
    with memory_usage() as memory_tracker:
        # Run inference
        predictions_df = predict_directory(
            model_path=model_path,
            data_dir=data_dir,
            img_size=img_size,
            num_classes=num_classes
        )
        
        memory_tracker.update_peak()
        
        # Limit samples if specified
        if num_samples is not None:
            predictions_df = predictions_df.head(num_samples)
        
        actual_samples = len(predictions_df)
    
    end_time = time.time()
    total_time = end_time - start_time
    throughput = actual_samples / total_time if total_time > 0 else 0
    
    return BenchmarkResults(
        operation="inference",
        total_time=total_time,
        peak_memory_mb=memory_tracker.peak_memory_mb,
        throughput_samples_per_sec=throughput,
        metadata={
            "num_classes": num_classes,
            "img_size": img_size,
            "num_samples": actual_samples,
            "model_path": model_path,
        }
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for chest X-ray pneumonia detector"
    )
    
    subparsers = parser.add_subparsers(dest="operation", help="Benchmark operation")
    
    # Training benchmark parser
    train_parser = subparsers.add_parser("training", help="Benchmark training performance")
    train_parser.add_argument("--train_dir", type=str, help="Training data directory")
    train_parser.add_argument("--val_dir", type=str, help="Validation data directory")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
    train_parser.add_argument("--use_dummy_data", action="store_true", default=True,
                             help="Use dummy data")
    train_parser.add_argument("--use_transfer_learning", action="store_true",
                             help="Use transfer learning")
    train_parser.add_argument("--use_attention_model", action="store_true",
                             help="Use attention model")
    train_parser.add_argument("--base_model_name", type=str, default="MobileNetV2",
                             help="Base model name")
    
    # Inference benchmark parser
    infer_parser = subparsers.add_parser("inference", help="Benchmark inference performance")
    infer_parser.add_argument("--model_path", type=str, required=True,
                             help="Path to trained model")
    infer_parser.add_argument("--data_dir", type=str, required=True,
                             help="Test data directory")
    infer_parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
    infer_parser.add_argument("--num_samples", type=int, help="Number of samples to process")
    
    # Output options
    parser.add_argument("--output_json", type=str, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.operation == "training":
        results = benchmark_training(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            use_dummy_data=args.use_dummy_data,
            use_transfer_learning=args.use_transfer_learning,
            use_attention_model=args.use_attention_model,
            base_model_name=args.base_model_name,
        )
    elif args.operation == "inference":
        results = benchmark_inference(
            model_path=args.model_path,
            data_dir=args.data_dir,
            num_classes=args.num_classes,
            num_samples=args.num_samples,
        )
    else:
        parser.print_help()
        return
    
    # Print results
    print(f"\n=== {results.operation.title()} Benchmark Results ===")
    print(f"Total Time: {results.total_time:.2f} seconds")
    if results.avg_time_per_epoch:
        print(f"Average Time per Epoch: {results.avg_time_per_epoch:.2f} seconds")
    if results.throughput_samples_per_sec:
        print(f"Throughput: {results.throughput_samples_per_sec:.1f} samples/second")
    print(f"Peak Memory Usage: {results.peak_memory_mb:.1f} MB")
    
    if args.verbose:
        print("\nMetadata:")
        for key, value in results.metadata.items():
            print(f"  {key}: {value}")
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(results.to_json())
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()