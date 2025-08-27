#!/usr/bin/env python3
"""
Adaptive UI Framework for Medical AI
Progressive Enhancement - Generation 1: Core Functionality
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio

class UIComponentType(Enum):
    """Types of adaptive UI components"""
    DASHBOARD = "dashboard"
    CHART = "chart" 
    FORM = "form"
    ALERT = "alert"
    METRIC = "metric"
    IMAGE_VIEWER = "image_viewer"

class AccessibilityLevel(Enum):
    """Accessibility compliance levels"""
    BASIC = "basic"
    AA = "aa"
    AAA = "aaa"

@dataclass
class UIComponent:
    """Adaptive UI component definition"""
    id: str
    type: UIComponentType
    title: str
    data: Dict[str, Any]
    accessibility: AccessibilityLevel = AccessibilityLevel.AA
    responsive: bool = True
    real_time: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class UserPreferences:
    """User interface preferences and accessibility needs"""
    theme: str = "light"
    font_size: str = "medium"
    high_contrast: bool = False
    screen_reader: bool = False
    language: str = "en"
    timezone: str = "UTC"
    reduced_motion: bool = False

class AdaptiveUIFramework:
    """
    Adaptive UI Framework for Medical AI Applications
    
    Features:
    - Real-time dashboard generation
    - Accessibility-first design
    - Multi-language support
    - Responsive layouts
    - Medical data visualization
    """
    
    def __init__(self, base_path: str = "ui_output"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.components: Dict[str, UIComponent] = {}
        self.user_preferences = UserPreferences()
        self.templates = {}
        self.real_time_callbacks: Dict[str, Callable] = {}
        
        self.logger = self._setup_logging()
        self._initialize_templates()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for UI framework"""
        logger = logging.getLogger("AdaptiveUI")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _initialize_templates(self):
        """Initialize UI templates"""
        self.templates = {
            "dashboard": self._dashboard_template(),
            "chart": self._chart_template(),
            "form": self._form_template(),
            "alert": self._alert_template(),
            "metric": self._metric_template(),
            "image_viewer": self._image_viewer_template()
        }
        
    def _dashboard_template(self) -> str:
        """Medical AI dashboard template"""
        return '''
<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --bg-primary: {bg_color};
            --text-primary: {text_color};
            --font-size: {font_size_px}px;
        }}
        
        * {{ 
            box-sizing: border-box; 
            font-family: 'Inter', system-ui, sans-serif;
        }}
        
        body {{ 
            margin: 0; 
            padding: 20px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: var(--font-size);
            line-height: 1.6;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .card {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
        }}
        
        .metric-card {{
            text-align: center;
            background: linear-gradient(135deg, var(--primary-color), #3b82f6);
            color: white;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        .alert {{
            padding: 12px 16px;
            border-radius: 6px;
            margin: 10px 0;
            font-weight: 500;
        }}
        
        .alert-success {{ background: #dcfce7; color: #166534; border-left: 4px solid var(--success-color); }}
        .alert-warning {{ background: #fef3c7; color: #92400e; border-left: 4px solid var(--warning-color); }}
        .alert-error {{ background: #fee2e2; color: #991b1b; border-left: 4px solid var(--error-color); }}
        
        .chart-container {{
            height: 300px;
            position: relative;
        }}
        
        /* Accessibility */
        @media (prefers-reduced-motion: reduce) {{
            * {{ animation-duration: 0.01ms !important; }}
        }}
        
        @media (prefers-contrast: high) {{
            .card {{ border: 2px solid var(--text-primary); }}
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .dashboard {{ grid-template-columns: 1fr; }}
            body {{ padding: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        {components}
    </div>
    
    <script>
        // Real-time updates
        {real_time_script}
        
        // Accessibility enhancements
        {accessibility_script}
    </script>
</body>
</html>
        '''
        
    def _chart_template(self) -> str:
        """Chart component template"""
        return '''
        <div class="card">
            <h3>{title}</h3>
            <div class="chart-container">
                <canvas id="chart-{id}" aria-label="{title} chart"></canvas>
            </div>
        </div>
        '''
        
    def _form_template(self) -> str:
        """Form component template"""
        return '''
        <div class="card">
            <h3>{title}</h3>
            <form id="form-{id}" aria-label="{title}">
                {form_fields}
                <button type="submit" class="btn-primary">Submit</button>
            </form>
        </div>
        '''
        
    def _alert_template(self) -> str:
        """Alert component template"""
        return '''
        <div class="alert alert-{type}" role="alert" aria-live="polite">
            <strong>{title}</strong>
            <p>{message}</p>
        </div>
        '''
        
    def _metric_template(self) -> str:
        """Metric display template"""
        return '''
        <div class="card metric-card">
            <div class="metric-value" aria-label="{label}">{value}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-trend">{trend}</div>
        </div>
        '''
        
    def _image_viewer_template(self) -> str:
        """Medical image viewer template"""
        return '''
        <div class="card">
            <h3>{title}</h3>
            <div class="image-viewer" role="img" aria-label="{alt_text}">
                <img src="{image_url}" alt="{alt_text}" style="max-width: 100%; height: auto;">
                <div class="image-metadata">
                    {metadata}
                </div>
            </div>
        </div>
        '''
        
    def create_component(self, component_id: str, component_type: UIComponentType, 
                        title: str, data: Dict[str, Any], **kwargs) -> UIComponent:
        """Create a new UI component"""
        component = UIComponent(
            id=component_id,
            type=component_type,
            title=title,
            data=data,
            **kwargs
        )
        
        self.components[component_id] = component
        self.logger.info(f"Created component: {component_id} ({component_type.value})")
        
        return component
        
    def create_medical_dashboard(self, metrics: Dict[str, Any]) -> str:
        """Create a comprehensive medical AI dashboard"""
        components_html = []
        
        # Health metrics
        if 'accuracy' in metrics:
            accuracy_component = self._metric_template().format(
                value=f"{metrics['accuracy']:.1%}",
                label="Model Accuracy",
                trend="↗ +2.3%" if metrics['accuracy'] > 0.9 else "→ Stable"
            )
            components_html.append(accuracy_component)
            
        # Processing statistics
        if 'processed_images' in metrics:
            processed_component = self._metric_template().format(
                value=f"{metrics['processed_images']:,}",
                label="Images Processed",
                trend=f"↗ +{metrics.get('daily_increase', 0):,} today"
            )
            components_html.append(processed_component)
            
        # System health alerts
        if 'alerts' in metrics:
            for alert in metrics['alerts']:
                alert_component = self._alert_template().format(
                    type=alert.get('type', 'info'),
                    title=alert.get('title', 'Alert'),
                    message=alert.get('message', '')
                )
                components_html.append(alert_component)
                
        # Recent predictions chart
        if 'recent_predictions' in metrics:
            chart_component = self._chart_template().format(
                id="predictions",
                title="Recent Predictions"
            )
            components_html.append(chart_component)
            
        # Generate dashboard
        font_sizes = {"small": 14, "medium": 16, "large": 18}
        
        dashboard_html = self.templates["dashboard"].format(
            language=self.user_preferences.language,
            title="Medical AI Dashboard",
            bg_color="#ffffff" if self.user_preferences.theme == "light" else "#1f2937",
            text_color="#111827" if self.user_preferences.theme == "light" else "#f9fafb",
            font_size_px=font_sizes.get(self.user_preferences.font_size, 16),
            components="\n".join(components_html),
            real_time_script=self._generate_realtime_script(),
            accessibility_script=self._generate_accessibility_script()
        )
        
        return dashboard_html
        
    def _generate_realtime_script(self) -> str:
        """Generate JavaScript for real-time updates"""
        return '''
        // Real-time dashboard updates
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update metrics
                    updateMetrics(data);
                })
                .catch(error => console.error('Update failed:', error));
        }
        
        function updateMetrics(data) {
            // Update metric values
            const metricElements = document.querySelectorAll('.metric-value');
            metricElements.forEach(element => {
                const label = element.getAttribute('aria-label');
                if (data[label]) {
                    element.textContent = data[label];
                    element.setAttribute('aria-live', 'polite');
                }
            });
        }
        
        // Update every 30 seconds
        setInterval(updateDashboard, 30000);
        '''
        
    def _generate_accessibility_script(self) -> str:
        """Generate JavaScript for accessibility enhancements"""
        return '''
        // Accessibility enhancements
        document.addEventListener('DOMContentLoaded', function() {
            // Add keyboard navigation
            const cards = document.querySelectorAll('.card');
            cards.forEach((card, index) => {
                card.setAttribute('tabindex', '0');
                card.setAttribute('role', 'region');
            });
            
            // Screen reader announcements
            function announceUpdate(message) {
                const announcement = document.createElement('div');
                announcement.setAttribute('aria-live', 'polite');
                announcement.setAttribute('aria-atomic', 'true');
                announcement.className = 'sr-only';
                announcement.textContent = message;
                document.body.appendChild(announcement);
                
                setTimeout(() => document.body.removeChild(announcement), 1000);
            }
            
            // High contrast mode detection
            if (window.matchMedia('(prefers-contrast: high)').matches) {
                document.body.classList.add('high-contrast');
            }
        });
        '''
        
    def render_to_file(self, filename: str, content: str):
        """Render UI content to HTML file"""
        output_path = self.base_path / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.logger.info(f"UI rendered to: {output_path}")
        return output_path
        
    def set_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        for key, value in preferences.items():
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)
                
        self.logger.info("User preferences updated")
        
    def add_real_time_callback(self, component_id: str, callback: Callable):
        """Add real-time update callback for component"""
        self.real_time_callbacks[component_id] = callback
        
    async def update_component_data(self, component_id: str, new_data: Dict[str, Any]):
        """Update component data with real-time callback"""
        if component_id in self.components:
            self.components[component_id].data.update(new_data)
            
            # Execute callback if registered
            if component_id in self.real_time_callbacks:
                await self.real_time_callbacks[component_id](new_data)
                
    def get_component_json(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get component data as JSON"""
        if component_id in self.components:
            return asdict(self.components[component_id])
        return None
        
    def export_dashboard_config(self) -> Dict[str, Any]:
        """Export complete dashboard configuration"""
        return {
            "components": {cid: asdict(comp) for cid, comp in self.components.items()},
            "user_preferences": asdict(self.user_preferences),
            "timestamp": datetime.now().isoformat()
        }


def demo_adaptive_ui():
    """Demonstrate the Adaptive UI Framework"""
    print("🎨 Adaptive UI Framework Demo")
    print("=" * 40)
    
    # Initialize framework
    ui = AdaptiveUIFramework("demo_ui_output")
    
    # Set user preferences
    ui.set_user_preferences({
        "theme": "light",
        "font_size": "medium",
        "high_contrast": False,
        "language": "en"
    })
    
    # Create sample medical AI metrics
    sample_metrics = {
        "accuracy": 0.94,
        "processed_images": 1250,
        "daily_increase": 180,
        "alerts": [
            {
                "type": "success",
                "title": "Model Update",
                "message": "Model v1.2 deployed successfully with 2% accuracy improvement"
            },
            {
                "type": "warning", 
                "title": "High Load",
                "message": "Processing queue depth is above normal threshold"
            }
        ],
        "recent_predictions": [
            {"time": "09:00", "normal": 12, "pneumonia": 3},
            {"time": "10:00", "normal": 15, "pneumonia": 5},
            {"time": "11:00", "normal": 18, "pneumonia": 2}
        ]
    }
    
    # Generate dashboard
    dashboard_html = ui.create_medical_dashboard(sample_metrics)
    
    # Save to file
    output_file = ui.render_to_file("medical_dashboard.html", dashboard_html)
    
    print(f"✅ Dashboard generated: {output_file}")
    print(f"📊 Components created: {len(ui.components)}")
    
    # Export configuration
    config = ui.export_dashboard_config()
    config_file = ui.base_path / "dashboard_config.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
        
    print(f"⚙️  Configuration saved: {config_file}")
    print("\n🎯 Demo complete - check the generated files!")


if __name__ == "__main__":
    demo_adaptive_ui()