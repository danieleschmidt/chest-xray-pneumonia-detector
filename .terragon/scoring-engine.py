#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Scoring Engine
Advanced scoring implementation combining WSJF, ICE, and Technical Debt methodologies
Repository: chest_xray_pneumonia_detector
"""

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ValueItem:
    """Represents a discovered value opportunity with comprehensive scoring."""
    id: str
    title: str
    category: str
    impact: str  # High/Medium/Low
    effort: int  # Hours estimate (1-8)
    risk: str    # Low/Medium/High
    medical_ai_priority: str
    description: str
    discovered_at: str
    scores: Dict[str, float]
    composite_score: float
    

class ValueScoringEngine:
    """Advanced value scoring engine implementing WSJF + ICE + Technical Debt."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        """Initialize scoring engine with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.maturity_level = self.config.get('repository', {}).get('maturity_level', 'advanced')
        self.weights = self.config.get('scoring', {}).get('weights', {}).get(self.maturity_level, {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Load value discovery configuration."""
        try:
            if yaml is None:
                return self._default_config()
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, ImportError):
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for advanced repositories."""
        return {
            'scoring': {
                'weights': {
                    'advanced': {
                        'wsjf': 0.5,
                        'ice': 0.1,
                        'technicalDebt': 0.3,
                        'security': 0.1
                    }
                },
                'thresholds': {
                    'minScore': 15,
                    'maxRisk': 0.75,
                    'securityBoost': 2.0,
                    'complianceBoost': 1.8,
                    'mlopsBoost': 1.5
                }
            }
        }
    
    def calculate_wsjf_score(self, item: Dict[str, Any]) -> float:
        """Calculate Weighted Shortest Job First (WSJF) score."""
        # Cost of Delay components
        user_business_value = self._score_user_impact(item)
        time_criticality = self._score_urgency(item)  
        risk_reduction = self._score_risk_mitigation(item)
        opportunity_enablement = self._score_opportunity(item)
        
        cost_of_delay = (
            user_business_value * 0.4 +
            time_criticality * 0.3 +
            risk_reduction * 0.2 +
            opportunity_enablement * 0.1
        )
        
        # Job Size (effort in hours, logarithmic scale for diminishing returns)
        job_size = max(1, math.log2(item.get('effort', 1)))
        
        return cost_of_delay / job_size
    
    def calculate_ice_score(self, item: Dict[str, Any]) -> float:
        """Calculate Impact/Confidence/Ease (ICE) score."""
        impact = self._score_business_impact(item)      # 1-10 scale
        confidence = self._score_execution_confidence(item)  # 1-10 scale  
        ease = self._score_implementation_ease(item)    # 1-10 scale
        
        return impact * confidence * ease
    
    def calculate_technical_debt_score(self, item: Dict[str, Any]) -> float:
        """Calculate technical debt impact score."""
        debt_impact = self._calculate_debt_cost(item)
        debt_interest = self._calculate_debt_growth(item)
        hotspot_multiplier = self._get_churn_complexity(item)
        
        return (debt_impact + debt_interest) * hotspot_multiplier
    
    def calculate_composite_score(self, item: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive composite score."""
        # Calculate individual scores
        wsjf_score = self.calculate_wsjf_score(item)
        ice_score = self.calculate_ice_score(item)
        tech_debt_score = self.calculate_technical_debt_score(item)
        
        # Normalize scores to 0-100 scale
        normalized_scores = {
            'wsjf': self._normalize_score(wsjf_score, max_value=50.0),
            'ice': self._normalize_score(ice_score, max_value=1000.0),
            'technicalDebt': self._normalize_score(tech_debt_score, max_value=200.0)
        }
        
        # Apply adaptive weighting
        composite_score = (
            self.weights.get('wsjf', 0.5) * normalized_scores['wsjf'] +
            self.weights.get('ice', 0.1) * normalized_scores['ice'] +
            self.weights.get('technicalDebt', 0.3) * normalized_scores['technicalDebt']
        )
        
        # Apply category-specific boosts
        composite_score = self._apply_boosts(composite_score, item)
        
        return {
            'wsjf': wsjf_score,
            'ice': ice_score,
            'technicalDebt': tech_debt_score,
            'composite': composite_score
        }
    
    def _score_user_impact(self, item: Dict[str, Any]) -> float:
        """Score user/business impact (0-10 scale)."""
        impact_map = {
            'High': 8.5,
            'Medium': 5.0, 
            'Low': 2.0
        }
        
        base_score = impact_map.get(item.get('impact', 'Medium'), 5.0)
        
        # Medical AI priority boost
        priority = item.get('medical_ai_priority', '').upper()
        if 'CRITICAL' in priority:
            base_score *= 1.5
        elif 'HIGH' in priority:
            base_score *= 1.3
        
        return min(10.0, base_score)
    
    def _score_urgency(self, item: Dict[str, Any]) -> float:
        """Score time criticality (0-10 scale)."""
        category = item.get('category', '').lower()
        
        # Security and compliance items are time-critical
        if category == 'security':
            return 9.0
        elif 'compliance' in item.get('description', '').lower():
            return 8.0
        elif category == 'technical_debt':
            return 4.0
        else:
            return 3.0
    
    def _score_risk_mitigation(self, item: Dict[str, Any]) -> float:
        """Score risk reduction value (0-10 scale)."""
        risk_level = item.get('risk', 'Medium').lower()
        category = item.get('category', '').lower()
        
        # Higher scores for items that mitigate high risks
        base_score = {
            'security': 9.0,
            'technical_debt': 6.0,
            'mlops': 5.0,
            'performance': 4.0,
            'quality': 3.0
        }.get(category, 3.0)
        
        # Adjust for implementation risk
        risk_adjustment = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.6
        }.get(risk_level, 0.8)
        
        return base_score * risk_adjustment
    
    def _score_opportunity(self, item: Dict[str, Any]) -> float:
        """Score opportunity enablement (0-10 scale)."""
        description = item.get('description', '').lower() 
        
        # Look for opportunity-enabling keywords
        enablers = {
            'optimization': 6.0,
            'automation': 7.0,
            'integration': 5.0,
            'scalability': 8.0,
            'monitoring': 4.0,
            'testing': 3.0
        }
        
        max_score = 0
        for keyword, score in enablers.items():
            if keyword in description:
                max_score = max(max_score, score)
        
        return max_score if max_score > 0 else 2.0
    
    def _score_business_impact(self, item: Dict[str, Any]) -> float:
        """Score business impact for ICE calculation (1-10 scale)."""
        impact_map = {
            'High': 9.0,
            'Medium': 6.0,
            'Low': 3.0
        }
        return impact_map.get(item.get('impact', 'Medium'), 6.0)
    
    def _score_execution_confidence(self, item: Dict[str, Any]) -> float:
        """Score execution confidence (1-10 scale)."""
        risk_confidence_map = {
            'Low': 9.0,     # Low risk = high confidence
            'Medium': 6.0,
            'High': 3.0     # High risk = low confidence
        }
        
        base_confidence = risk_confidence_map.get(item.get('risk', 'Medium'), 6.0)
        
        # Adjust for effort - smaller tasks are more predictable
        effort = item.get('effort', 4)
        if effort <= 2:
            base_confidence *= 1.2
        elif effort >= 7:
            base_confidence *= 0.8
        
        return min(10.0, base_confidence)
    
    def _score_implementation_ease(self, item: Dict[str, Any]) -> float:
        """Score implementation ease (1-10 scale)."""
        effort = item.get('effort', 4)
        
        # Inverse relationship between effort and ease
        if effort <= 2:
            ease = 9.0
        elif effort <= 4:
            ease = 7.0
        elif effort <= 6:
            ease = 5.0
        else:
            ease = 3.0
        
        # Adjust for risk
        risk_level = item.get('risk', 'Medium').lower()
        risk_adjustment = {
            'low': 1.0,
            'medium': 0.9,
            'high': 0.7
        }.get(risk_level, 0.9)
        
        return ease * risk_adjustment
    
    def _calculate_debt_cost(self, item: Dict[str, Any]) -> float:
        """Calculate technical debt maintenance cost."""
        category = item.get('category', '').lower()
        
        debt_costs = {
            'technical_debt': 8.0,
            'security': 6.0,  # Security debt is costly but often well-contained
            'performance': 7.0,
            'quality': 5.0,
            'mlops': 6.0
        }
        
        return debt_costs.get(category, 4.0)
    
    def _calculate_debt_growth(self, item: Dict[str, Any]) -> float:
        """Calculate debt interest/growth rate."""
        description = item.get('description', '').lower()
        
        # Debt that grows quickly over time
        growth_indicators = {
            'complexity': 3.0,
            'maintainability': 4.0,
            'scalability': 5.0,
            'performance': 3.0,
            'security': 6.0  # Security debt compounds rapidly
        }
        
        max_growth = 1.0
        for indicator, growth in growth_indicators.items():
            if indicator in description:
                max_growth = max(max_growth, growth)
        
        return max_growth
    
    def _get_churn_complexity(self, item: Dict[str, Any]) -> float:
        """Get churn/complexity multiplier for hotspots."""
        # Simplified hotspot detection based on description
        description = item.get('description', '').lower()
        
        if 'train_engine' in description or '1034' in description:
            return 2.5  # High churn file
        elif 'model_registry' in description or '940' in description:
            return 2.0  # Medium-high churn
        elif any(word in description for word in ['core', 'critical', 'main']):
            return 1.8
        else:
            return 1.0
    
    def _normalize_score(self, score: float, max_value: float) -> float:
        """Normalize score to 0-100 range."""
        return min(100.0, (score / max_value) * 100.0)
    
    def _apply_boosts(self, score: float, item: Dict[str, Any]) -> float:
        """Apply category-specific score boosts."""
        thresholds = self.config.get('scoring', {}).get('thresholds', {})
        
        # Security vulnerability boost
        if item.get('category') == 'security':
            score *= thresholds.get('securityBoost', 2.0)
        
        # Compliance boost for medical AI
        description = item.get('description', '').lower()
        if any(word in description for word in ['hipaa', 'compliance', 'regulatory']):
            score *= thresholds.get('complianceBoost', 1.8)
        
        # MLOps boost
        if item.get('category') == 'mlops':
            score *= thresholds.get('mlopsBoost', 1.5)
        
        return score


def create_value_items() -> List[ValueItem]:
    """Create value items from discovered opportunities."""
    opportunities = [
        {
            'id': 'HIPAA-001',
            'title': 'HIPAA Audit Logging Enhancement',
            'category': 'security',
            'impact': 'High',
            'effort': 5,
            'risk': 'Low',
            'medical_ai_priority': 'CRITICAL',
            'description': 'Implement comprehensive audit logging for all PHI access, model predictions, and user actions for HIPAA compliance.'
        },
        {
            'id': 'MLOPS-001', 
            'title': 'Model Drift Detection Implementation',
            'category': 'mlops',
            'impact': 'High',
            'effort': 7,
            'risk': 'Medium',
            'medical_ai_priority': 'CRITICAL',
            'description': 'Automated drift detection for medical model accuracy and performance monitoring.'
        },
        {
            'id': 'TD-001',
            'title': 'Train Engine Complexity Reduction',
            'category': 'technical_debt', 
            'impact': 'High',
            'effort': 8,
            'risk': 'Medium',
            'medical_ai_priority': 'HIGH',
            'description': 'Refactor 1034-line train_engine.py for improved maintainability and testing.'
        },
        {
            'id': 'SEC-001',
            'title': 'Medical Data Encryption at Rest',
            'category': 'security',
            'impact': 'High', 
            'effort': 6,
            'risk': 'Medium',
            'medical_ai_priority': 'HIGH',
            'description': 'AES-256 encryption for stored medical images and PHI data compliance.'
        },
        {
            'id': 'SEC-002',
            'title': 'Input Validation for Medical Images',
            'category': 'security',
            'impact': 'High',
            'effort': 5,
            'risk': 'Medium', 
            'medical_ai_priority': 'HIGH',
            'description': 'Medical image format validation, metadata sanitization, and DICOM header security.'
        }
    ]
    
    engine = ValueScoringEngine()
    value_items = []
    
    for opp in opportunities:
        scores = engine.calculate_composite_score(opp)
        
        item = ValueItem(
            id=opp['id'],
            title=opp['title'],
            category=opp['category'],
            impact=opp['impact'],
            effort=opp['effort'],
            risk=opp['risk'],
            medical_ai_priority=opp['medical_ai_priority'],
            description=opp['description'],
            discovered_at=datetime.now(timezone.utc).isoformat(),
            scores=scores,
            composite_score=scores['composite']
        )
        value_items.append(item)
    
    return sorted(value_items, key=lambda x: x.composite_score, reverse=True)


if __name__ == "__main__":
    # Generate scored value items
    items = create_value_items()
    
    # Save to JSON for consumption by other tools
    output_path = Path(".terragon/value-metrics.json")
    output_path.parent.mkdir(exist_ok=True)
    
    metrics = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'repository': 'chest_xray_pneumonia_detector',
        'maturity_level': 'advanced',
        'total_items': len(items),
        'items': [asdict(item) for item in items],
        'summary': {
            'average_composite_score': sum(item.composite_score for item in items) / len(items),
            'high_priority_items': len([item for item in items if item.composite_score > 100]),
            'categories': list(set(item.category for item in items))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Generated {len(items)} scored value items")
    print(f"Saved metrics to {output_path}")
    
    # Display top 3 items
    print("\nTop 3 Value Items:")
    for i, item in enumerate(items[:3], 1):
        print(f"{i}. {item.title} (Score: {item.composite_score:.1f})")