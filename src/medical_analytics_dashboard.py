#!/usr/bin/env python3
"""
Simple Medical Analytics Dashboard
Generation 1: Basic functionality for medical data visualization and analysis
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse


class SimpleMedicalAnalytics:
    """Basic medical analytics for chest X-ray predictions and model performance."""
    
    def __init__(self, data_dir: str = "analytics_data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create analytics data directory if it doesn't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def log_prediction(self, image_path: str, prediction: float, 
                      confidence: float, model_version: str = "v1.0"):
        """Log a single prediction for analytics tracking."""
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
            "predicted_class": "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        }
        
        log_file = os.path.join(self.data_dir, "predictions.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(prediction_data) + "\n")
        
        return prediction_data
    
    def get_prediction_stats(self) -> Dict:
        """Get basic statistics about predictions."""
        log_file = os.path.join(self.data_dir, "predictions.jsonl")
        
        if not os.path.exists(log_file):
            return {"total_predictions": 0, "pneumonia_detected": 0, "normal_detected": 0}
        
        total = 0
        pneumonia_count = 0
        normal_count = 0
        high_confidence_count = 0
        
        with open(log_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                total += 1
                if data["predicted_class"] == "PNEUMONIA":
                    pneumonia_count += 1
                else:
                    normal_count += 1
                
                if data["confidence"] > 0.8:
                    high_confidence_count += 1
        
        return {
            "total_predictions": total,
            "pneumonia_detected": pneumonia_count,
            "normal_detected": normal_count,
            "high_confidence_predictions": high_confidence_count,
            "pneumonia_rate": pneumonia_count / total if total > 0 else 0,
            "high_confidence_rate": high_confidence_count / total if total > 0 else 0
        }
    
    def generate_simple_report(self) -> str:
        """Generate a simple text report of analytics."""
        stats = self.get_prediction_stats()
        
        report = f"""
Medical Analytics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== Prediction Summary ===
Total Predictions: {stats['total_predictions']}
Pneumonia Detected: {stats['pneumonia_detected']}
Normal Cases: {stats['normal_detected']}
High Confidence Predictions: {stats['high_confidence_predictions']}

=== Rates ===
Pneumonia Detection Rate: {stats['pneumonia_rate']:.2%}
High Confidence Rate: {stats['high_confidence_rate']:.2%}

=== System Health ===
Status: Operational
Data Directory: {self.data_dir}
"""
        return report
    
    def export_analytics_csv(self, output_file: str = None) -> str:
        """Export analytics data to CSV format."""
        if output_file is None:
            output_file = os.path.join(self.data_dir, "analytics_export.csv")
        
        log_file = os.path.join(self.data_dir, "predictions.jsonl")
        
        if not os.path.exists(log_file):
            with open(output_file, "w") as f:
                f.write("timestamp,image_path,prediction,confidence,model_version,predicted_class\n")
            return output_file
        
        with open(output_file, "w") as out_f:
            out_f.write("timestamp,image_path,prediction,confidence,model_version,predicted_class\n")
            
            with open(log_file, "r") as in_f:
                for line in in_f:
                    data = json.loads(line.strip())
                    out_f.write(f"{data['timestamp']},{data['image_path']},{data['prediction']},{data['confidence']},{data['model_version']},{data['predicted_class']}\n")
        
        return output_file


def main():
    """CLI interface for medical analytics dashboard."""
    parser = argparse.ArgumentParser(description="Simple Medical Analytics Dashboard")
    parser.add_argument("--data-dir", default="analytics_data", help="Analytics data directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate analytics report")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export analytics to CSV")
    export_parser.add_argument("--output", help="Output CSV file path")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show quick statistics")
    
    # Log command for testing
    log_parser = subparsers.add_parser("log", help="Log a test prediction")
    log_parser.add_argument("--image", required=True, help="Image path")
    log_parser.add_argument("--prediction", type=float, required=True, help="Prediction value (0-1)")
    log_parser.add_argument("--confidence", type=float, required=True, help="Confidence value (0-1)")
    
    args = parser.parse_args()
    
    analytics = SimpleMedicalAnalytics(args.data_dir)
    
    if args.command == "report":
        print(analytics.generate_simple_report())
    
    elif args.command == "export":
        output_file = analytics.export_analytics_csv(args.output)
        print(f"Analytics exported to: {output_file}")
    
    elif args.command == "stats":
        stats = analytics.get_prediction_stats()
        print("Quick Stats:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}" if "rate" in key else f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    elif args.command == "log":
        prediction_data = analytics.log_prediction(
            args.image, args.prediction, args.confidence
        )
        print(f"Logged prediction: {prediction_data['predicted_class']} with {prediction_data['confidence']:.2%} confidence")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()