#!/usr/bin/env python3
"""
Unified CLI for Chest X-Ray Pneumonia Detection System
====================================================

This is a comprehensive command-line interface that provides access to all
functionality of the quantum-enhanced medical AI system in a simple, unified way.

Usage:
    python cxr_cli.py --help
    python cxr_cli.py train --help
    python cxr_cli.py predict --help
    python cxr_cli.py demo --help
"""

import argparse
import sys
from pathlib import Path
import subprocess
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config


class CXRCLIManager:
    """Unified CLI manager for the chest X-ray system."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
    
    def run_command(self, cmd: List[str], capture_output: bool = False) -> Optional[str]:
        """Execute a command and optionally capture output."""
        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
                if result.returncode != 0:
                    print(f"Error: {result.stderr}")
                    return None
                return result.stdout
            else:
                subprocess.run(cmd, cwd=self.base_dir)
                return None
        except Exception as e:
            print(f"Failed to execute command: {e}")
            return None
    
    def train(self, args):
        """Execute training pipeline."""
        cmd = ["python", "-m", "src.train_engine"]
        
        # Add common training arguments
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])
        if args.learning_rate:
            cmd.extend(["--learning_rate", str(args.learning_rate)])
        if args.use_transfer_learning:
            cmd.append("--use_transfer_learning")
        if args.base_model_name:
            cmd.extend(["--base_model_name", args.base_model_name])
        
        print("🚀 Starting training pipeline...")
        self.run_command(cmd)
    
    def predict(self, args):
        """Execute prediction pipeline."""
        if args.single_image:
            # Single image prediction with Grad-CAM
            cmd = ["python", "-m", "src.predict_utils",
                   "--model_path", args.model_path,
                   "--img_path", args.single_image]
            if args.output_path:
                cmd.extend(["--output_path", args.output_path])
        else:
            # Batch inference
            cmd = ["python", "-m", "src.inference",
                   "--model_path", args.model_path,
                   "--data_dir", args.data_dir,
                   "--output_csv", args.output_csv or "predictions.csv"]
        
        print("🔮 Running prediction...")
        self.run_command(cmd)
    
    def evaluate(self, args):
        """Execute evaluation pipeline."""
        cmd = ["python", "-m", "src.evaluate",
               "--pred_csv", args.pred_csv]
        
        if args.label_csv:
            cmd.extend(["--label_csv", args.label_csv])
        if args.output_png:
            cmd.extend(["--output_png", args.output_png])
        if args.normalize_cm:
            cmd.append("--normalize_cm")
        
        print("📊 Running evaluation...")
        self.run_command(cmd)
    
    def demo(self, args):
        """Run comprehensive system demo."""
        print("🎯 Running Comprehensive System Demo")
        print("=" * 50)
        
        demo_scripts = [
            ("Simple Demo", "simple_demo.py"),
            ("Robust Demo", "simple_robust_demo.py"),
            ("Research Demo V2", "research_demo_v2.py"),
            ("Performance Demo", "performance_demo.py")
        ]
        
        for name, script in demo_scripts:
            if not args.demo_type or args.demo_type.lower() in name.lower():
                print(f"\n🔹 Running {name}...")
                cmd = ["python", script]
                self.run_command(cmd)
                print(f"✅ {name} completed")
    
    def quality_gates(self, args):
        """Run comprehensive quality gates."""
        print("🛡️ Running Quality Gates...")
        
        scripts = [
            "run_quality_gates.py",
            "enhanced_quality_gates.py",
            "comprehensive_quality_gates.py"
        ]
        
        for script in scripts:
            if Path(self.base_dir / script).exists():
                print(f"\n🔹 Running {script}...")
                self.run_command(["python", script])
    
    def deployment(self, args):
        """Handle deployment operations."""
        print("🚀 Deployment Operations...")
        
        if args.environment == "production":
            cmd = ["python", "enhanced_production_deployment.py"]
        elif args.environment == "quantum":
            cmd = ["python", "quantum_enhanced_deployment.py"]
        else:
            cmd = ["python", "production_deployment_orchestrator.py"]
        
        self.run_command(cmd)
    
    def monitoring(self, args):
        """Start monitoring systems."""
        print("📡 Starting Monitoring Systems...")
        cmd = ["python", "intelligent_monitoring_system.py"]
        
        if args.comprehensive:
            cmd = ["python", "src/comprehensive_monitoring_system.py"]
        
        self.run_command(cmd)
    
    def status(self, args):
        """Show system status."""
        print("📋 System Status")
        print("=" * 30)
        
        # Check configuration
        print("\n🔧 Configuration:")
        env_info = config.get_env_info()
        for key, value in env_info.items():
            if key.endswith('_URI') or key.endswith('_PATH'):
                exists = Path(str(value)).exists() if value else False
                status = "✅" if exists else "❌"
                print(f"  {status} {key}: {value}")
            else:
                print(f"  ℹ️  {key}: {value}")
        
        # Check model files
        print("\n🤖 Models:")
        model_paths = [config.CHECKPOINT_PATH, config.SAVE_MODEL_PATH]
        for path in model_paths:
            exists = Path(path).exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {Path(path).name}: {path}")
        
        # Check recent logs
        print("\n📝 Recent Activity:")
        log_dirs = ["logs", "audit_logs", "deployment_artifacts"]
        for log_dir in log_dirs:
            log_path = self.base_dir / log_dir
            if log_path.exists():
                files = list(log_path.glob("*"))
                print(f"  📁 {log_dir}: {len(files)} files")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified CLI for Quantum-Enhanced Medical AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cxr_cli.py train --epochs 10 --use_transfer_learning
  python cxr_cli.py predict --single_image image.jpg --model_path model.keras
  python cxr_cli.py demo --demo_type simple
  python cxr_cli.py quality_gates
  python cxr_cli.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--use_transfer_learning', action='store_true', help='Use transfer learning')
    train_parser.add_argument('--base_model_name', default='MobileNetV2', help='Base model for transfer learning')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model_path', required=True, help='Path to trained model')
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument('--single_image', help='Single image path for prediction')
    predict_group.add_argument('--data_dir', help='Directory for batch prediction')
    predict_parser.add_argument('--output_csv', help='Output CSV for batch predictions')
    predict_parser.add_argument('--output_path', help='Output path for Grad-CAM visualization')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    eval_parser.add_argument('--pred_csv', required=True, help='Predictions CSV file')
    eval_parser.add_argument('--label_csv', help='Labels CSV file')
    eval_parser.add_argument('--output_png', help='Confusion matrix output path')
    eval_parser.add_argument('--normalize_cm', action='store_true', help='Normalize confusion matrix')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run system demonstrations')
    demo_parser.add_argument('--demo_type', choices=['simple', 'robust', 'research', 'performance', 'all'], 
                            default='all', help='Type of demo to run')
    
    # Quality gates command
    quality_parser = subparsers.add_parser('quality_gates', help='Run quality assurance checks')
    
    # Deployment command
    deploy_parser = subparsers.add_parser('deploy', help='Handle deployment operations')
    deploy_parser.add_argument('--environment', choices=['staging', 'production', 'quantum'], 
                              default='staging', help='Deployment environment')
    
    # Monitoring command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring systems')
    monitor_parser.add_argument('--comprehensive', action='store_true', help='Use comprehensive monitoring')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli_manager = CXRCLIManager()
    
    # Route to appropriate command handler
    command_handlers = {
        'train': cli_manager.train,
        'predict': cli_manager.predict,
        'evaluate': cli_manager.evaluate,
        'demo': cli_manager.demo,
        'quality_gates': cli_manager.quality_gates,
        'deploy': cli_manager.deployment,
        'monitor': cli_manager.monitoring,
        'status': cli_manager.status,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
        except Exception as e:
            print(f"❌ Error executing {args.command}: {e}")
    else:
        print(f"❌ Unknown command: {args.command}")


if __name__ == "__main__":
    main()