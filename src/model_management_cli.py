"""Command-line interface for model registry management.

This CLI provides easy access to model versioning and A/B testing functionality
for production deployment workflows. It integrates with the existing training
pipeline and provides operations for model promotion, rollback, and monitoring.

Usage Examples:
    # Register a newly trained model
    python -m src.model_management_cli register \
        --model-id pneumonia_detector \
        --version 1.2.0 \
        --model-path saved_models/pneumonia_cnn_v1.keras \
        --accuracy 0.92 \
        --f1-score 0.89 \
        --roc-auc 0.95 \
        --dataset-version chest_xray_v3

    # Promote model to production
    python -m src.model_management_cli promote \
        --model-id pneumonia_detector \
        --version 1.2.0

    # Start A/B test
    python -m src.model_management_cli ab-test start \
        --model-id pneumonia_detector \
        --control-version 1.1.0 \
        --treatment-version 1.2.0 \
        --traffic-split 0.2 \
        --duration-days 7

    # List models
    python -m src.model_management_cli list --model-id pneumonia_detector
"""

import argparse
import json
import sys
from datetime import datetime

from .model_registry import (
    ModelRegistry,
    ModelMetadata,
    ABTestConfig,
    ModelValidationError,
    ModelPromotionError,
)
from .config import config


def create_registry() -> ModelRegistry:
    """Create model registry instance with configuration."""
    registry_path = (
        config.MODEL_REGISTRY_PATH
        if hasattr(config, "MODEL_REGISTRY_PATH")
        else "./model_registry"
    )
    mlflow_uri = (
        config.MLFLOW_TRACKING_URI if hasattr(config, "MLFLOW_TRACKING_URI") else None
    )
    return ModelRegistry(registry_path=registry_path, mlflow_tracking_uri=mlflow_uri)


def cmd_register(args) -> None:
    """Register a new model version."""
    # Validate input paths for security
    from .input_validation import validate_model_path, ValidationError

    try:
        args.model_path = validate_model_path(args.model_path, must_exist=True)
    except ValidationError as e:
        print(f"❌ Input validation error: {e}")
        return

    registry = create_registry()

    # Create metadata from command line arguments
    metadata = ModelMetadata(
        model_id=args.model_id,
        version=args.version,
        accuracy=args.accuracy,
        f1_score=args.f1_score,
        roc_auc=args.roc_auc,
        training_date=datetime.now(),
        dataset_version=args.dataset_version,
        model_path=args.model_path,
        training_config=(
            json.loads(args.training_config) if args.training_config else {}
        ),
        description=args.description,
        tags=args.tags.split(",") if args.tags else [],
        mlflow_run_id=args.mlflow_run_id,
    )

    try:
        registry_path = registry.register_model(metadata)
        print(f"✅ Successfully registered model {args.model_id} v{args.version}")
        print(f"   Registry path: {registry_path}")
        print(f"   Accuracy: {args.accuracy:.3f}")
        print(f"   F1 Score: {args.f1_score:.3f}")
        print(f"   ROC AUC: {args.roc_auc:.3f}")
    except (ModelValidationError, FileNotFoundError) as e:
        print(f"❌ Registration failed: {e}")
        sys.exit(1)


def cmd_promote(args) -> None:
    """Promote a model version to production."""
    registry = create_registry()

    try:
        registry.promote_to_production(args.model_id, args.version)
        print(f"✅ Successfully promoted {args.model_id} v{args.version} to production")

        # Show current production model
        production_model = registry.get_production_model(args.model_id)
        if production_model:
            print(
                f"   Production model: {production_model.model_id} v{production_model.version}"
            )
            print(
                f"   Performance: Acc={production_model.accuracy:.3f}, F1={production_model.f1_score:.3f}"
            )
    except ModelPromotionError as e:
        print(f"❌ Promotion failed: {e}")
        sys.exit(1)


def cmd_rollback(args) -> None:
    """Rollback to a previous model version."""
    registry = create_registry()

    try:
        # Get current production model for confirmation
        current_prod = registry.get_production_model(args.model_id)
        if current_prod:
            print(f"⚠️  Rolling back from {current_prod.version} to {args.version}")

        registry.rollback_model(args.model_id, args.version)
        print(f"✅ Successfully rolled back {args.model_id} to v{args.version}")
    except ModelPromotionError as e:
        print(f"❌ Rollback failed: {e}")
        sys.exit(1)


def cmd_list(args) -> None:
    """List registered models."""
    registry = create_registry()

    models = registry.list_models(model_id=args.model_id)

    if not models:
        print("No models found.")
        return

    # Group by model_id for better display
    models_by_id = {}
    for model in models:
        if model.model_id not in models_by_id:
            models_by_id[model.model_id] = []
        models_by_id[model.model_id].append(model)

    for model_id, model_versions in models_by_id.items():
        print(f"\n📦 Model: {model_id}")
        print(f"   {'=' * 60}")

        for model in sorted(model_versions, key=lambda m: m.version, reverse=True):
            status_icon = (
                "🟢"
                if model.is_production
                else "🔵" if model.status == "staged" else "⚪"
            )
            print(f"   {status_icon} v{model.version} ({model.status})")
            print(
                f"      Accuracy: {model.accuracy:.3f} | F1: {model.f1_score:.3f} | ROC AUC: {model.roc_auc:.3f}"
            )
            print(f"      Trained: {model.training_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"      Dataset: {model.dataset_version}")
            if model.description:
                print(f"      Description: {model.description}")
            print()


def cmd_ab_test_start(args) -> None:
    """Start an A/B test."""
    registry = create_registry()

    # Create A/B test configuration
    config = ABTestConfig(
        experiment_name=args.experiment_name,
        control_model_version=args.control_version,
        treatment_model_version=args.treatment_version,
        traffic_split=args.traffic_split,
        success_metrics=(
            args.success_metrics.split(",")
            if args.success_metrics
            else ["accuracy", "f1_score"]
        ),
        duration_days=args.duration_days,
        minimum_sample_size=args.min_samples,
        confidence_level=args.confidence_level,
    )

    try:
        registry.start_ab_test(args.model_id, config)
        print(f"✅ Started A/B test: {args.experiment_name}")
        print(
            f"   Control: v{args.control_version} ({(1-args.traffic_split)*100:.1f}% traffic)"
        )
        print(
            f"   Treatment: v{args.treatment_version} ({args.traffic_split*100:.1f}% traffic)"
        )
        print(f"   Duration: {args.duration_days} days")
        print(f"   Success metrics: {config.success_metrics}")
    except ModelValidationError as e:
        print(f"❌ A/B test creation failed: {e}")
        sys.exit(1)


def cmd_ab_test_list(args) -> None:
    """List active A/B tests."""
    registry = create_registry()

    active_tests = registry.list_active_ab_tests()

    if not active_tests:
        print("No active A/B tests.")
        return

    print("🧪 Active A/B Tests")
    print("=" * 50)

    for test in active_tests:
        days_remaining = (test.end_date - datetime.now()).days
        print(f"\n📊 {test.experiment_name}")
        print(
            f"   Control: v{test.control_model_version} ({(1-test.traffic_split)*100:.1f}%)"
        )
        print(
            f"   Treatment: v{test.treatment_model_version} ({test.traffic_split*100:.1f}%)"
        )
        print(f"   Duration: {days_remaining} days remaining")
        print(f"   Metrics: {', '.join(test.success_metrics)}")


def cmd_performance(args) -> None:
    """Show model performance metrics."""
    registry = create_registry()

    summary = registry.get_model_performance_summary(args.model_id, args.version)

    if summary["total_inferences"] == 0:
        print(f"No performance data available for {args.model_id} v{args.version}")
        return

    print(f"📈 Performance Summary: {args.model_id} v{args.version}")
    print("=" * 50)
    print(f"Total inferences: {summary['total_inferences']:,}")
    print(f"Average confidence: {summary.get('avg_confidence', 0):.3f}")
    print(f"Average inference time: {summary.get('avg_inference_time_ms', 0):.1f} ms")

    if "accuracy" in summary:
        print(f"Runtime accuracy: {summary['accuracy']:.3f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model Registry CLI for production ML deployments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command
    register_parser = subparsers.add_parser(
        "register", help="Register a new model version"
    )
    register_parser.add_argument("--model-id", required=True, help="Model identifier")
    register_parser.add_argument(
        "--version", required=True, help="Model version (semantic versioning)"
    )
    register_parser.add_argument(
        "--model-path", required=True, help="Path to model file"
    )
    register_parser.add_argument(
        "--accuracy", type=float, required=True, help="Model accuracy (0-1)"
    )
    register_parser.add_argument(
        "--f1-score", type=float, required=True, help="F1 score (0-1)"
    )
    register_parser.add_argument(
        "--roc-auc", type=float, required=True, help="ROC AUC score (0-1)"
    )
    register_parser.add_argument(
        "--dataset-version", required=True, help="Training dataset version"
    )
    register_parser.add_argument(
        "--training-config", help="Training configuration as JSON string"
    )
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument("--tags", help="Comma-separated tags")
    register_parser.add_argument("--mlflow-run-id", help="Associated MLflow run ID")
    register_parser.set_defaults(func=cmd_register)

    # Promote command
    promote_parser = subparsers.add_parser(
        "promote", help="Promote model to production"
    )
    promote_parser.add_argument("--model-id", required=True, help="Model identifier")
    promote_parser.add_argument("--version", required=True, help="Version to promote")
    promote_parser.set_defaults(func=cmd_promote)

    # Rollback command
    rollback_parser = subparsers.add_parser(
        "rollback", help="Rollback to previous version"
    )
    rollback_parser.add_argument("--model-id", required=True, help="Model identifier")
    rollback_parser.add_argument(
        "--version", required=True, help="Version to rollback to"
    )
    rollback_parser.set_defaults(func=cmd_rollback)

    # List command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument("--model-id", help="Filter by model ID")
    list_parser.set_defaults(func=cmd_list)

    # A/B test commands
    ab_parser = subparsers.add_parser("ab-test", help="A/B testing commands")
    ab_subparsers = ab_parser.add_subparsers(
        dest="ab_command", help="A/B test operations"
    )

    # Start A/B test
    ab_start_parser = ab_subparsers.add_parser("start", help="Start A/B test")
    ab_start_parser.add_argument("--model-id", required=True, help="Model identifier")
    ab_start_parser.add_argument(
        "--experiment-name", required=True, help="Experiment name"
    )
    ab_start_parser.add_argument(
        "--control-version", required=True, help="Control model version"
    )
    ab_start_parser.add_argument(
        "--treatment-version", required=True, help="Treatment model version"
    )
    ab_start_parser.add_argument(
        "--traffic-split", type=float, required=True, help="Traffic to treatment (0-1)"
    )
    ab_start_parser.add_argument(
        "--duration-days", type=int, default=7, help="Test duration in days"
    )
    ab_start_parser.add_argument(
        "--success-metrics",
        default="accuracy,f1_score",
        help="Success metrics (comma-separated)",
    )
    ab_start_parser.add_argument(
        "--min-samples", type=int, default=1000, help="Minimum sample size"
    )
    ab_start_parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Statistical confidence level",
    )
    ab_start_parser.set_defaults(func=cmd_ab_test_start)

    # List A/B tests
    ab_list_parser = ab_subparsers.add_parser("list", help="List active A/B tests")
    ab_list_parser.set_defaults(func=cmd_ab_test_list)

    # Performance command
    perf_parser = subparsers.add_parser(
        "performance", help="Show model performance metrics"
    )
    perf_parser.add_argument("--model-id", required=True, help="Model identifier")
    perf_parser.add_argument("--version", required=True, help="Model version")
    perf_parser.set_defaults(func=cmd_performance)

    # Parse arguments and execute command
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ab-test" and not args.ab_command:
        ab_parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
