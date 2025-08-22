"""
Predictive Medical Analytics Engine
==================================

Advanced AI-driven predictive diagnostics system that provides:
- Multi-modal medical prediction
- Clinical risk assessment
- Treatment outcome prediction
- Population health analytics
- Personalized medicine recommendations

Key Features:
1. Temporal pattern analysis for disease progression
2. Multi-organ system correlation analysis
3. Risk stratification algorithms
4. Treatment response prediction
5. Population-level health trend analysis
6. Clinical decision support system
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PatientProfile:
    """Comprehensive patient profile for predictive analytics."""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_symptoms: List[str]
    vital_signs: Dict[str, float]
    lab_results: Dict[str, float]
    imaging_findings: Dict[str, Any]
    medications: List[str]
    genetic_markers: Dict[str, str] = field(default_factory=dict)
    lifestyle_factors: Dict[str, Any] = field(default_factory=dict)
    social_determinants: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictiveResult:
    """Comprehensive predictive analysis result."""
    patient_id: str
    primary_diagnosis_probability: Dict[str, float]
    risk_scores: Dict[str, float]
    disease_progression_timeline: Dict[str, List[Tuple[str, float]]]
    treatment_recommendations: List[Dict[str, Any]]
    clinical_alerts: List[Dict[str, Any]]
    confidence_score: float
    explanation: Dict[str, Any]
    population_comparison: Dict[str, Any]
    follow_up_recommendations: List[str]


@dataclass
class PopulationHealthMetrics:
    """Population-level health analytics metrics."""
    total_patients: int
    disease_prevalence: Dict[str, float]
    risk_distribution: Dict[str, Dict[str, int]]
    treatment_outcomes: Dict[str, Dict[str, float]]
    health_trends: Dict[str, List[Tuple[datetime, float]]]
    disparities_analysis: Dict[str, Dict[str, float]]
    cost_effectiveness: Dict[str, float]


class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in medical data for disease progression prediction."""
    
    def __init__(self):
        self.pattern_models = {}
        self.progression_profiles = {}
        
    def analyze_disease_progression(self, patient_history: List[Dict], 
                                  disease: str) -> Dict[str, Any]:
        """Analyze disease progression patterns for a patient."""
        
        # Extract temporal features
        timeline = self._extract_temporal_features(patient_history)
        
        # Predict progression stages
        progression_prediction = self._predict_progression_stages(timeline, disease)
        
        # Calculate progression velocity
        progression_velocity = self._calculate_progression_velocity(timeline)
        
        # Identify critical decision points
        decision_points = self._identify_decision_points(timeline, disease)
        
        return {
            'progression_stage': progression_prediction['current_stage'],
            'next_stage_probability': progression_prediction['next_stage_prob'],
            'time_to_progression': progression_prediction['time_estimate'],
            'progression_velocity': progression_velocity,
            'critical_decision_points': decision_points,
            'risk_factors': self._identify_progression_risk_factors(timeline)
        }
    
    def _extract_temporal_features(self, patient_history: List[Dict]) -> np.ndarray:
        """Extract temporal features from patient history."""
        
        features = []
        
        # Sort history by timestamp
        sorted_history = sorted(patient_history, key=lambda x: x.get('timestamp', 0))
        
        for i, record in enumerate(sorted_history):
            temporal_features = [
                record.get('symptom_severity', 0),
                record.get('lab_value_trend', 0),
                record.get('treatment_response', 0),
                i / len(sorted_history),  # Relative time position
                record.get('vital_sign_stability', 0)
            ]
            features.append(temporal_features)
        
        return np.array(features) if features else np.zeros((1, 5))
    
    def _predict_progression_stages(self, timeline: np.ndarray, 
                                  disease: str) -> Dict[str, Any]:
        """Predict disease progression stages."""
        
        # Disease-specific progression models
        disease_stages = {
            'pneumonia': ['early', 'consolidation', 'resolution', 'recovered'],
            'copd': ['mild', 'moderate', 'severe', 'very_severe'],
            'heart_failure': ['stage_a', 'stage_b', 'stage_c', 'stage_d'],
            'diabetes': ['prediabetes', 'early', 'established', 'complicated']
        }
        
        stages = disease_stages.get(disease, ['early', 'moderate', 'advanced', 'critical'])
        
        # Simulate progression prediction (in practice, use trained models)
        if len(timeline) > 0:
            current_severity = np.mean(timeline[-3:, 0]) if len(timeline) >= 3 else timeline[-1, 0]
            stage_index = min(len(stages) - 1, int(current_severity * len(stages)))
            current_stage = stages[stage_index]
            
            # Predict next stage probability
            next_stage_prob = min(0.95, current_severity + np.random.normal(0, 0.1))
            
            # Estimate time to progression
            progression_rate = np.mean(np.diff(timeline[:, 0])) if len(timeline) > 1 else 0.1
            time_estimate = max(1, int(30 / (progression_rate + 0.01)))  # Days
            
        else:
            current_stage = stages[0]
            next_stage_prob = 0.2
            time_estimate = 30
        
        return {
            'current_stage': current_stage,
            'next_stage_prob': next_stage_prob,
            'time_estimate': time_estimate
        }
    
    def _calculate_progression_velocity(self, timeline: np.ndarray) -> float:
        """Calculate the velocity of disease progression."""
        
        if len(timeline) < 2:
            return 0.0
        
        # Calculate rate of change in severity
        severity_changes = np.diff(timeline[:, 0])
        
        # Weight recent changes more heavily
        weights = np.exp(np.linspace(-1, 0, len(severity_changes)))
        weighted_velocity = np.average(severity_changes, weights=weights)
        
        return float(weighted_velocity)
    
    def _identify_decision_points(self, timeline: np.ndarray, 
                                disease: str) -> List[Dict[str, Any]]:
        """Identify critical decision points in disease progression."""
        
        decision_points = []
        
        if len(timeline) > 0:
            current_severity = timeline[-1, 0] if len(timeline) > 0 else 0
            
            # Disease-specific decision thresholds
            if disease == 'pneumonia':
                if current_severity > 0.7:
                    decision_points.append({
                        'point': 'hospitalization_consideration',
                        'urgency': 'high',
                        'recommendation': 'Consider immediate hospitalization',
                        'rationale': 'Severe pneumonia indicators present'
                    })
                elif current_severity > 0.4:
                    decision_points.append({
                        'point': 'antibiotic_escalation',
                        'urgency': 'medium',
                        'recommendation': 'Consider broad-spectrum antibiotics',
                        'rationale': 'Moderate severity with potential complications'
                    })
            
            elif disease == 'heart_failure':
                if current_severity > 0.6:
                    decision_points.append({
                        'point': 'device_therapy',
                        'urgency': 'medium',
                        'recommendation': 'Evaluate for device therapy',
                        'rationale': 'Advanced heart failure stage reached'
                    })
        
        return decision_points
    
    def _identify_progression_risk_factors(self, timeline: np.ndarray) -> List[str]:
        """Identify risk factors contributing to disease progression."""
        
        risk_factors = []
        
        if len(timeline) > 0:
            recent_trend = np.mean(np.diff(timeline[-5:, 0])) if len(timeline) >= 6 else 0
            
            if recent_trend > 0.1:
                risk_factors.extend(['rapid_progression', 'treatment_resistance'])
            
            stability = np.std(timeline[:, 4]) if len(timeline) > 1 else 0
            if stability > 0.3:
                risk_factors.append('vital_sign_instability')
        
        return risk_factors


class MultiOrganSystemAnalyzer:
    """Analyzes correlations across multiple organ systems."""
    
    def __init__(self):
        self.organ_interactions = self._initialize_organ_interactions()
        
    def _initialize_organ_interactions(self) -> Dict[str, Dict[str, float]]:
        """Initialize known organ system interaction weights."""
        
        return {
            'cardiovascular': {
                'respiratory': 0.8,
                'renal': 0.7,
                'endocrine': 0.6,
                'neurological': 0.5
            },
            'respiratory': {
                'cardiovascular': 0.8,
                'immune': 0.7,
                'renal': 0.4
            },
            'renal': {
                'cardiovascular': 0.7,
                'endocrine': 0.8,
                'immune': 0.5
            },
            'endocrine': {
                'cardiovascular': 0.6,
                'renal': 0.8,
                'immune': 0.5,
                'neurological': 0.4
            }
        }
    
    def analyze_system_correlations(self, patient_profile: PatientProfile) -> Dict[str, Any]:
        """Analyze correlations across organ systems for a patient."""
        
        # Extract system-specific indicators
        system_indicators = self._extract_system_indicators(patient_profile)
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_system_correlations(system_indicators)
        
        # Identify cascading effects
        cascading_effects = self._identify_cascading_effects(system_indicators)
        
        # Predict system-level outcomes
        system_outcomes = self._predict_system_outcomes(system_indicators)
        
        return {
            'system_indicators': system_indicators,
            'correlation_matrix': correlation_matrix,
            'cascading_effects': cascading_effects,
            'system_outcomes': system_outcomes,
            'primary_affected_system': self._identify_primary_system(system_indicators),
            'secondary_risk_systems': self._identify_secondary_risks(system_indicators)
        }
    
    def _extract_system_indicators(self, patient_profile: PatientProfile) -> Dict[str, float]:
        """Extract organ system-specific health indicators."""
        
        indicators = {}
        
        # Cardiovascular indicators
        cv_score = 0.0
        if 'heart_rate' in patient_profile.vital_signs:
            hr = patient_profile.vital_signs['heart_rate']
            cv_score += 1.0 if 60 <= hr <= 100 else max(0, 1.0 - abs(hr - 80) / 50)
        
        if 'blood_pressure_systolic' in patient_profile.vital_signs:
            sbp = patient_profile.vital_signs['blood_pressure_systolic']
            cv_score += 1.0 if 90 <= sbp <= 140 else max(0, 1.0 - abs(sbp - 120) / 50)
        
        indicators['cardiovascular'] = cv_score / 2 if cv_score > 0 else 0.5
        
        # Respiratory indicators
        resp_score = 0.0
        if 'respiratory_rate' in patient_profile.vital_signs:
            rr = patient_profile.vital_signs['respiratory_rate']
            resp_score += 1.0 if 12 <= rr <= 20 else max(0, 1.0 - abs(rr - 16) / 10)
        
        if 'oxygen_saturation' in patient_profile.vital_signs:
            o2sat = patient_profile.vital_signs['oxygen_saturation']
            resp_score += min(1.0, o2sat / 95) if o2sat > 0 else 0.5
        
        indicators['respiratory'] = resp_score / 2 if resp_score > 0 else 0.5
        
        # Renal indicators
        renal_score = 0.0
        if 'creatinine' in patient_profile.lab_results:
            creat = patient_profile.lab_results['creatinine']
            renal_score = max(0, 1.0 - abs(creat - 1.0) / 2.0)  # Normal ~1.0 mg/dL
        
        indicators['renal'] = renal_score if renal_score > 0 else 0.5
        
        # Endocrine indicators
        endo_score = 0.0
        if 'glucose' in patient_profile.lab_results:
            glucose = patient_profile.lab_results['glucose']
            endo_score = max(0, 1.0 - abs(glucose - 100) / 100)  # Normal ~100 mg/dL
        
        indicators['endocrine'] = endo_score if endo_score > 0 else 0.5
        
        return indicators
    
    def _calculate_system_correlations(self, system_indicators: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between organ systems."""
        
        correlation_matrix = {}
        
        for system1 in system_indicators:
            correlation_matrix[system1] = {}
            for system2 in system_indicators:
                if system1 == system2:
                    correlation_matrix[system1][system2] = 1.0
                else:
                    # Base correlation from known interactions
                    base_corr = self.organ_interactions.get(system1, {}).get(system2, 0.3)
                    
                    # Adjust based on current indicator values
                    indicator1 = system_indicators[system1]
                    indicator2 = system_indicators[system2]
                    
                    # Higher correlation if both systems are similarly affected
                    similarity = 1.0 - abs(indicator1 - indicator2)
                    adjusted_corr = base_corr * similarity
                    
                    correlation_matrix[system1][system2] = adjusted_corr
        
        return correlation_matrix
    
    def _identify_cascading_effects(self, system_indicators: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify potential cascading effects between organ systems."""
        
        cascading_effects = []
        
        # Find systems with low indicators (potential dysfunction)
        dysfunctional_systems = [
            system for system, indicator in system_indicators.items() 
            if indicator < 0.6
        ]
        
        for system in dysfunctional_systems:
            # Find systems likely to be affected
            affected_systems = []
            for target_system, interaction_strength in self.organ_interactions.get(system, {}).items():
                if interaction_strength > 0.6 and target_system not in dysfunctional_systems:
                    affected_systems.append({
                        'system': target_system,
                        'risk_level': interaction_strength * (0.6 - system_indicators[system]),
                        'timeline': 'immediate' if interaction_strength > 0.7 else 'delayed'
                    })
            
            if affected_systems:
                cascading_effects.append({
                    'primary_system': system,
                    'affected_systems': affected_systems,
                    'severity': system_indicators[system]
                })
        
        return cascading_effects
    
    def _predict_system_outcomes(self, system_indicators: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Predict outcomes for each organ system."""
        
        outcomes = {}
        
        for system, indicator in system_indicators.items():
            # Predict recovery probability
            recovery_prob = min(0.95, indicator + 0.1)
            
            # Predict deterioration risk
            deterioration_risk = max(0.05, 1.0 - indicator)
            
            # Predict intervention need
            intervention_need = max(0.0, 0.8 - indicator)
            
            outcomes[system] = {
                'recovery_probability': recovery_prob,
                'deterioration_risk': deterioration_risk,
                'intervention_need': intervention_need,
                'stability_score': indicator
            }
        
        return outcomes
    
    def _identify_primary_system(self, system_indicators: Dict[str, float]) -> str:
        """Identify the primary affected organ system."""
        
        return min(system_indicators.items(), key=lambda x: x[1])[0]
    
    def _identify_secondary_risks(self, system_indicators: Dict[str, float]) -> List[str]:
        """Identify organ systems at secondary risk."""
        
        # Sort systems by risk (lower indicators = higher risk)
        sorted_systems = sorted(system_indicators.items(), key=lambda x: x[1])
        
        # Return systems with indicators below threshold
        return [system for system, indicator in sorted_systems if indicator < 0.7]


class RiskStratificationEngine:
    """Advanced risk stratification for clinical decision support."""
    
    def __init__(self):
        self.risk_models = self._initialize_risk_models()
        
    def _initialize_risk_models(self) -> Dict[str, Any]:
        """Initialize risk assessment models."""
        
        # Simplified risk models (in practice, these would be trained ML models)
        return {
            'mortality': {'weights': [0.3, 0.25, 0.2, 0.15, 0.1], 'threshold': 0.7},
            'readmission': {'weights': [0.25, 0.2, 0.2, 0.2, 0.15], 'threshold': 0.6},
            'complication': {'weights': [0.2, 0.3, 0.25, 0.15, 0.1], 'threshold': 0.5},
            'treatment_response': {'weights': [0.2, 0.2, 0.2, 0.2, 0.2], 'threshold': 0.4}
        }
    
    def calculate_risk_scores(self, patient_profile: PatientProfile,
                            clinical_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive risk scores for a patient."""
        
        # Extract risk factors
        risk_factors = self._extract_risk_factors(patient_profile, clinical_context)
        
        risk_scores = {}
        
        for risk_type, model in self.risk_models.items():
            # Calculate weighted risk score
            score = np.sum([
                factor * weight for factor, weight in 
                zip(risk_factors, model['weights'])
            ])
            
            # Normalize to 0-1 range
            risk_scores[risk_type] = min(1.0, max(0.0, score))
        
        return risk_scores
    
    def _extract_risk_factors(self, patient_profile: PatientProfile,
                            clinical_context: Dict[str, Any]) -> List[float]:
        """Extract normalized risk factors from patient profile."""
        
        factors = []
        
        # Age factor
        age_factor = min(1.0, patient_profile.age / 100)
        factors.append(age_factor)
        
        # Comorbidity factor
        comorbidity_count = len(patient_profile.medical_history)
        comorbidity_factor = min(1.0, comorbidity_count / 10)
        factors.append(comorbidity_factor)
        
        # Severity factor from vital signs
        severity_factor = self._calculate_severity_factor(patient_profile.vital_signs)
        factors.append(severity_factor)
        
        # Lab abnormality factor
        lab_factor = self._calculate_lab_abnormality_factor(patient_profile.lab_results)
        factors.append(lab_factor)
        
        # Social risk factor
        social_factor = self._calculate_social_risk_factor(patient_profile.social_determinants)
        factors.append(social_factor)
        
        return factors
    
    def _calculate_severity_factor(self, vital_signs: Dict[str, float]) -> float:
        """Calculate severity factor from vital signs."""
        
        severity_score = 0.0
        factor_count = 0
        
        # Heart rate abnormality
        if 'heart_rate' in vital_signs:
            hr = vital_signs['heart_rate']
            if hr < 50 or hr > 120:
                severity_score += 0.3
            factor_count += 1
        
        # Blood pressure abnormality
        if 'blood_pressure_systolic' in vital_signs:
            sbp = vital_signs['blood_pressure_systolic']
            if sbp < 90 or sbp > 160:
                severity_score += 0.3
            factor_count += 1
        
        # Respiratory rate abnormality
        if 'respiratory_rate' in vital_signs:
            rr = vital_signs['respiratory_rate']
            if rr < 10 or rr > 25:
                severity_score += 0.2
            factor_count += 1
        
        # Temperature abnormality
        if 'temperature' in vital_signs:
            temp = vital_signs['temperature']
            if temp < 96 or temp > 101:
                severity_score += 0.2
            factor_count += 1
        
        return severity_score / max(1, factor_count)
    
    def _calculate_lab_abnormality_factor(self, lab_results: Dict[str, float]) -> float:
        """Calculate lab abnormality factor."""
        
        abnormality_score = 0.0
        factor_count = 0
        
        # Define normal ranges (simplified)
        normal_ranges = {
            'creatinine': (0.6, 1.3),
            'glucose': (70, 110),
            'white_blood_cell_count': (4.0, 11.0),
            'hemoglobin': (12.0, 16.0),
            'platelet_count': (150, 400)
        }
        
        for lab, value in lab_results.items():
            if lab in normal_ranges:
                min_val, max_val = normal_ranges[lab]
                if value < min_val or value > max_val:
                    # Calculate severity of abnormality
                    if value < min_val:
                        abnormality = (min_val - value) / min_val
                    else:
                        abnormality = (value - max_val) / max_val
                    
                    abnormality_score += min(0.5, abnormality)
                factor_count += 1
        
        return abnormality_score / max(1, factor_count)
    
    def _calculate_social_risk_factor(self, social_determinants: Dict[str, Any]) -> float:
        """Calculate social risk factor."""
        
        risk_score = 0.0
        
        # Insurance status
        if social_determinants.get('insurance_status') == 'uninsured':
            risk_score += 0.2
        
        # Housing stability
        if social_determinants.get('housing_status') == 'unstable':
            risk_score += 0.2
        
        # Transportation access
        if social_determinants.get('transportation_access') == 'limited':
            risk_score += 0.1
        
        # Social support
        if social_determinants.get('social_support') == 'limited':
            risk_score += 0.15
        
        # Employment status
        if social_determinants.get('employment_status') == 'unemployed':
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def generate_risk_alerts(self, risk_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate clinical alerts based on risk scores."""
        
        alerts = []
        
        for risk_type, score in risk_scores.items():
            model = self.risk_models[risk_type]
            threshold = model['threshold']
            
            if score >= threshold:
                severity = 'critical' if score >= 0.8 else 'high' if score >= 0.6 else 'moderate'
                
                alerts.append({
                    'type': f'{risk_type}_risk',
                    'severity': severity,
                    'score': score,
                    'message': f'High {risk_type} risk detected (score: {score:.3f})',
                    'recommendations': self._get_risk_specific_recommendations(risk_type, score)
                })
        
        return alerts
    
    def _get_risk_specific_recommendations(self, risk_type: str, score: float) -> List[str]:
        """Get risk-specific clinical recommendations."""
        
        recommendations = {
            'mortality': [
                'Consider intensive monitoring',
                'Evaluate for ICU transfer',
                'Optimize supportive care',
                'Discuss goals of care with family'
            ],
            'readmission': [
                'Arrange close follow-up within 72 hours',
                'Medication reconciliation',
                'Patient education on warning signs',
                'Consider home health services'
            ],
            'complication': [
                'Enhanced monitoring for complications',
                'Prophylactic interventions as appropriate',
                'Early intervention protocols',
                'Multidisciplinary team consultation'
            ],
            'treatment_response': [
                'Consider alternative treatment options',
                'Monitor for treatment failure',
                'Adjust therapy as needed',
                'Specialist consultation'
            ]
        }
        
        base_recommendations = recommendations.get(risk_type, ['Enhanced monitoring recommended'])
        
        # Add severity-specific recommendations
        if score >= 0.8:
            return base_recommendations + ['Immediate physician notification required']
        elif score >= 0.6:
            return base_recommendations[:2] + ['Frequent reassessment needed']
        else:
            return base_recommendations[:1]


class TreatmentOutcomePredictor:
    """Predicts treatment outcomes and response probabilities."""
    
    def __init__(self):
        self.treatment_models = self._initialize_treatment_models()
        
    def _initialize_treatment_models(self) -> Dict[str, Any]:
        """Initialize treatment outcome prediction models."""
        
        # Simplified models (in practice, these would be trained ML models)
        return {
            'antibiotic_response': {
                'factors': ['pathogen_susceptibility', 'patient_immune_status', 'comorbidities'],
                'weights': [0.4, 0.3, 0.3]
            },
            'surgical_outcome': {
                'factors': ['patient_fitness', 'procedure_complexity', 'surgeon_experience'],
                'weights': [0.4, 0.35, 0.25]
            },
            'medication_adherence': {
                'factors': ['medication_complexity', 'patient_education', 'social_support'],
                'weights': [0.3, 0.4, 0.3]
            }
        }
    
    def predict_treatment_outcomes(self, patient_profile: PatientProfile,
                                 proposed_treatments: List[str]) -> Dict[str, Dict[str, float]]:
        """Predict outcomes for proposed treatments."""
        
        treatment_outcomes = {}
        
        for treatment in proposed_treatments:
            # Extract treatment-specific factors
            factors = self._extract_treatment_factors(patient_profile, treatment)
            
            # Predict success probability
            success_prob = self._calculate_success_probability(treatment, factors)
            
            # Predict side effect risk
            side_effect_risk = self._calculate_side_effect_risk(patient_profile, treatment)
            
            # Predict time to improvement
            time_to_improvement = self._predict_time_to_improvement(treatment, factors)
            
            treatment_outcomes[treatment] = {
                'success_probability': success_prob,
                'side_effect_risk': side_effect_risk,
                'time_to_improvement': time_to_improvement,
                'confidence': min(0.95, success_prob * 0.8 + 0.2)
            }
        
        return treatment_outcomes
    
    def _extract_treatment_factors(self, patient_profile: PatientProfile,
                                 treatment: str) -> Dict[str, float]:
        """Extract factors relevant to treatment success."""
        
        factors = {}
        
        # Patient-specific factors
        factors['age_factor'] = 1.0 - (patient_profile.age / 100)
        factors['comorbidity_burden'] = min(1.0, len(patient_profile.medical_history) / 5)
        
        # Treatment-specific factors
        if 'antibiotic' in treatment.lower():
            factors['pathogen_susceptibility'] = np.random.uniform(0.6, 0.9)  # Simulated
            factors['immune_status'] = 1.0 - factors['comorbidity_burden'] * 0.5
            
        elif 'surgery' in treatment.lower():
            factors['surgical_risk'] = factors['age_factor'] * factors['comorbidity_burden']
            factors['anesthesia_risk'] = factors['age_factor'] * 0.8
            
        elif 'medication' in treatment.lower():
            factors['adherence_likelihood'] = 0.8 - factors['comorbidity_burden'] * 0.2
            factors['drug_interaction_risk'] = len(patient_profile.medications) / 10
        
        return factors
    
    def _calculate_success_probability(self, treatment: str, 
                                     factors: Dict[str, float]) -> float:
        """Calculate treatment success probability."""
        
        base_success_rates = {
            'antibiotic_therapy': 0.85,
            'surgical_intervention': 0.80,
            'medication_therapy': 0.75,
            'supportive_care': 0.70
        }
        
        # Find matching treatment type
        base_rate = 0.70  # Default
        for treatment_type, rate in base_success_rates.items():
            if any(word in treatment.lower() for word in treatment_type.split('_')):
                base_rate = rate
                break
        
        # Adjust based on patient factors
        age_adjustment = factors.get('age_factor', 0.8)
        comorbidity_adjustment = 1.0 - factors.get('comorbidity_burden', 0.0) * 0.3
        
        # Treatment-specific adjustments
        if 'antibiotic' in treatment.lower():
            pathogen_adj = factors.get('pathogen_susceptibility', 0.8)
            immune_adj = factors.get('immune_status', 0.8)
            adjusted_rate = base_rate * pathogen_adj * immune_adj * age_adjustment
            
        elif 'surgery' in treatment.lower():
            surgical_risk_adj = 1.0 - factors.get('surgical_risk', 0.2)
            adjusted_rate = base_rate * surgical_risk_adj * age_adjustment
            
        else:
            adjusted_rate = base_rate * age_adjustment * comorbidity_adjustment
        
        return min(0.95, max(0.05, adjusted_rate))
    
    def _calculate_side_effect_risk(self, patient_profile: PatientProfile,
                                  treatment: str) -> float:
        """Calculate side effect risk for treatment."""
        
        base_risk = 0.1  # 10% base risk
        
        # Age-related risk increase
        age_risk = min(0.3, patient_profile.age / 200)
        
        # Comorbidity-related risk
        comorbidity_risk = min(0.4, len(patient_profile.medical_history) / 10)
        
        # Medication interaction risk
        interaction_risk = min(0.3, len(patient_profile.medications) / 15)
        
        total_risk = base_risk + age_risk + comorbidity_risk + interaction_risk
        
        return min(0.8, total_risk)
    
    def _predict_time_to_improvement(self, treatment: str, 
                                   factors: Dict[str, float]) -> int:
        """Predict time to improvement in days."""
        
        base_times = {
            'antibiotic': 5,
            'surgery': 14,
            'medication': 21,
            'supportive': 10
        }
        
        # Find base time
        base_time = 10  # Default
        for treatment_type, time_days in base_times.items():
            if treatment_type in treatment.lower():
                base_time = time_days
                break
        
        # Adjust based on patient factors
        age_factor = factors.get('age_factor', 0.8)
        comorbidity_factor = 1.0 + factors.get('comorbidity_burden', 0.0)
        
        adjusted_time = base_time * comorbidity_factor / age_factor
        
        return max(1, int(adjusted_time))


class PredictiveMedicalAnalyticsEngine:
    """Main predictive medical analytics engine."""
    
    def __init__(self):
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.multi_organ_analyzer = MultiOrganSystemAnalyzer()
        self.risk_stratifier = RiskStratificationEngine()
        self.outcome_predictor = TreatmentOutcomePredictor()
        
        self.patient_database = {}
        self.population_metrics = None
        
        self.logger = logging.getLogger(__name__)
        
    def analyze_patient(self, patient_profile: PatientProfile,
                       clinical_context: Optional[Dict[str, Any]] = None) -> PredictiveResult:
        """Perform comprehensive predictive analysis for a patient."""
        
        clinical_context = clinical_context or {}
        
        # Multi-organ system analysis
        system_analysis = self.multi_organ_analyzer.analyze_system_correlations(patient_profile)
        
        # Risk stratification
        risk_scores = self.risk_stratifier.calculate_risk_scores(patient_profile, clinical_context)
        
        # Clinical alerts
        risk_alerts = self.risk_stratifier.generate_risk_alerts(risk_scores)
        
        # Disease progression analysis
        patient_history = clinical_context.get('patient_history', [])
        primary_diagnosis = clinical_context.get('suspected_diagnosis', 'pneumonia')
        
        progression_analysis = self.temporal_analyzer.analyze_disease_progression(
            patient_history, primary_diagnosis
        )
        
        # Treatment outcome predictions
        proposed_treatments = clinical_context.get('proposed_treatments', ['antibiotic_therapy'])
        treatment_outcomes = self.outcome_predictor.predict_treatment_outcomes(
            patient_profile, proposed_treatments
        )
        
        # Generate primary diagnosis probabilities
        diagnosis_probabilities = self._calculate_diagnosis_probabilities(
            patient_profile, system_analysis
        )
        
        # Generate treatment recommendations
        treatment_recommendations = self._generate_treatment_recommendations(
            patient_profile, risk_scores, treatment_outcomes
        )
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence_score(
            system_analysis, risk_scores, treatment_outcomes
        )
        
        # Generate explanations
        explanation = self._generate_explanation(
            patient_profile, system_analysis, risk_scores, progression_analysis
        )
        
        # Population comparison
        population_comparison = self._compare_to_population(patient_profile, risk_scores)
        
        # Follow-up recommendations
        follow_up_recommendations = self._generate_follow_up_recommendations(
            risk_scores, progression_analysis
        )
        
        result = PredictiveResult(
            patient_id=patient_profile.patient_id,
            primary_diagnosis_probability=diagnosis_probabilities,
            risk_scores=risk_scores,
            disease_progression_timeline={primary_diagnosis: [(
                progression_analysis['progression_stage'],
                progression_analysis['next_stage_probability']
            )]},
            treatment_recommendations=treatment_recommendations,
            clinical_alerts=risk_alerts,
            confidence_score=confidence_score,
            explanation=explanation,
            population_comparison=population_comparison,
            follow_up_recommendations=follow_up_recommendations
        )
        
        # Store patient in database for population analytics
        self.patient_database[patient_profile.patient_id] = {
            'profile': patient_profile,
            'result': result,
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"Completed predictive analysis for patient {patient_profile.patient_id}")
        
        return result
    
    def _calculate_diagnosis_probabilities(self, patient_profile: PatientProfile,
                                         system_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate probabilities for different diagnoses."""
        
        # Extract relevant indicators
        resp_indicator = system_analysis['system_indicators'].get('respiratory', 0.5)
        cv_indicator = system_analysis['system_indicators'].get('cardiovascular', 0.5)
        
        # Symptom analysis
        respiratory_symptoms = [s for s in patient_profile.current_symptoms 
                              if any(word in s.lower() for word in ['cough', 'shortness', 'chest'])]
        
        probabilities = {}
        
        # Pneumonia probability
        pneumonia_prob = 0.3  # Base probability
        if respiratory_symptoms:
            pneumonia_prob += 0.4
        if resp_indicator < 0.6:
            pneumonia_prob += 0.2
        if 'fever' in str(patient_profile.current_symptoms):
            pneumonia_prob += 0.1
        
        probabilities['pneumonia'] = min(0.95, pneumonia_prob)
        
        # Heart failure probability
        hf_prob = 0.1
        if cv_indicator < 0.6:
            hf_prob += 0.3
        if any('edema' in s.lower() for s in patient_profile.current_symptoms):
            hf_prob += 0.2
        
        probabilities['heart_failure'] = min(0.95, hf_prob)
        
        # COPD probability
        copd_prob = 0.15
        if 'smoking' in str(patient_profile.medical_history):
            copd_prob += 0.3
        if patient_profile.age > 50:
            copd_prob += 0.2
        
        probabilities['copd'] = min(0.95, copd_prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 1.0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    def _generate_treatment_recommendations(self, patient_profile: PatientProfile,
                                          risk_scores: Dict[str, float],
                                          treatment_outcomes: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate evidence-based treatment recommendations."""
        
        recommendations = []
        
        # Sort treatments by success probability
        sorted_treatments = sorted(
            treatment_outcomes.items(),
            key=lambda x: x[1]['success_probability'],
            reverse=True
        )
        
        for treatment, outcomes in sorted_treatments[:3]:  # Top 3 treatments
            recommendation = {
                'treatment': treatment,
                'success_probability': outcomes['success_probability'],
                'side_effect_risk': outcomes['side_effect_risk'],
                'time_to_improvement': outcomes['time_to_improvement'],
                'recommendation_strength': 'strong' if outcomes['success_probability'] > 0.8 else 'moderate',
                'rationale': f"Success probability: {outcomes['success_probability']:.1%}, "
                           f"Side effect risk: {outcomes['side_effect_risk']:.1%}",
                'monitoring_requirements': self._get_monitoring_requirements(treatment, risk_scores)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_monitoring_requirements(self, treatment: str, 
                                   risk_scores: Dict[str, float]) -> List[str]:
        """Get monitoring requirements for treatment."""
        
        monitoring = []
        
        # High-risk patients need more monitoring
        if max(risk_scores.values()) > 0.7:
            monitoring.append('Intensive monitoring required')
        
        # Treatment-specific monitoring
        if 'antibiotic' in treatment.lower():
            monitoring.extend(['Monitor for antibiotic resistance', 'Check liver function'])
        elif 'surgery' in treatment.lower():
            monitoring.extend(['Post-operative monitoring', 'Watch for complications'])
        
        return monitoring
    
    def _calculate_confidence_score(self, system_analysis: Dict[str, Any],
                                  risk_scores: Dict[str, float],
                                  treatment_outcomes: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall confidence in predictions."""
        
        # Base confidence
        confidence = 0.7
        
        # Higher confidence with consistent system indicators
        system_indicators = list(system_analysis['system_indicators'].values())
        if len(system_indicators) > 1:
            consistency = 1.0 - np.std(system_indicators)
            confidence += consistency * 0.2
        
        # Higher confidence with clear risk stratification
        risk_clarity = max(risk_scores.values()) - min(risk_scores.values())
        confidence += risk_clarity * 0.1
        
        # Higher confidence with high treatment success probabilities
        max_treatment_success = max(
            outcomes['success_probability'] for outcomes in treatment_outcomes.values()
        )
        confidence += (max_treatment_success - 0.5) * 0.1
        
        return min(0.95, max(0.3, confidence))
    
    def _generate_explanation(self, patient_profile: PatientProfile,
                            system_analysis: Dict[str, Any],
                            risk_scores: Dict[str, float],
                            progression_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation of predictions."""
        
        explanation = {
            'key_findings': [],
            'risk_factors': [],
            'protective_factors': [],
            'decision_rationale': ''
        }
        
        # Key findings
        primary_system = system_analysis['primary_affected_system']
        explanation['key_findings'].append(f"Primary affected system: {primary_system}")
        
        highest_risk = max(risk_scores.items(), key=lambda x: x[1])
        explanation['key_findings'].append(f"Highest risk: {highest_risk[0]} ({highest_risk[1]:.1%})")
        
        # Risk factors
        if patient_profile.age > 65:
            explanation['risk_factors'].append('Advanced age (>65 years)')
        
        if len(patient_profile.medical_history) > 3:
            explanation['risk_factors'].append('Multiple comorbidities')
        
        if progression_analysis.get('progression_velocity', 0) > 0.1:
            explanation['risk_factors'].append('Rapid disease progression')
        
        # Protective factors
        if patient_profile.age < 50:
            explanation['protective_factors'].append('Younger age (<50 years)')
        
        if len(patient_profile.medical_history) < 2:
            explanation['protective_factors'].append('Limited comorbidity burden')
        
        # Decision rationale
        explanation['decision_rationale'] = (
            f"Based on {primary_system} dysfunction, patient age, comorbidity profile, "
            f"and current clinical presentation, the recommended approach prioritizes "
            f"the highest-probability treatment while managing identified risks."
        )
        
        return explanation
    
    def _compare_to_population(self, patient_profile: PatientProfile,
                             risk_scores: Dict[str, float]) -> Dict[str, Any]:
        """Compare patient to population statistics."""
        
        # Simulate population comparison (in practice, use real population data)
        comparison = {
            'age_group_percentile': min(95, max(5, 50 + (patient_profile.age - 65) * 2)),
            'risk_percentile': {},
            'similar_patients_outcomes': {
                'recovery_rate': np.random.uniform(0.7, 0.9),
                'average_los': np.random.randint(3, 10),
                'readmission_rate': np.random.uniform(0.1, 0.3)
            }
        }
        
        # Risk percentiles
        for risk_type, score in risk_scores.items():
            # Simulate where patient ranks in population for each risk
            comparison['risk_percentile'][risk_type] = min(95, max(5, score * 100))
        
        return comparison
    
    def _generate_follow_up_recommendations(self, risk_scores: Dict[str, float],
                                          progression_analysis: Dict[str, Any]) -> List[str]:
        """Generate follow-up care recommendations."""
        
        recommendations = []
        
        # High-risk patients need closer follow-up
        if max(risk_scores.values()) > 0.7:
            recommendations.append('Close follow-up within 24-48 hours')
        else:
            recommendations.append('Follow-up within 1 week')
        
        # Disease progression-based recommendations
        if progression_analysis.get('progression_velocity', 0) > 0.1:
            recommendations.append('Monitor for disease progression')
        
        # Risk-specific recommendations
        if risk_scores.get('readmission', 0) > 0.6:
            recommendations.append('Arrange post-discharge care coordination')
        
        if risk_scores.get('complication', 0) > 0.5:
            recommendations.append('Patient education on warning signs')
        
        return recommendations
    
    def generate_population_health_analytics(self) -> PopulationHealthMetrics:
        """Generate population-level health analytics."""
        
        if not self.patient_database:
            return PopulationHealthMetrics(
                total_patients=0,
                disease_prevalence={},
                risk_distribution={},
                treatment_outcomes={},
                health_trends={},
                disparities_analysis={},
                cost_effectiveness={}
            )
        
        total_patients = len(self.patient_database)
        
        # Disease prevalence
        disease_counts = {}
        for patient_data in self.patient_database.values():
            result = patient_data['result']
            primary_diagnosis = max(
                result.primary_diagnosis_probability.items(),
                key=lambda x: x[1]
            )[0]
            disease_counts[primary_diagnosis] = disease_counts.get(primary_diagnosis, 0) + 1
        
        disease_prevalence = {
            disease: count / total_patients 
            for disease, count in disease_counts.items()
        }
        
        # Risk distribution
        risk_distribution = {}
        for risk_type in ['mortality', 'readmission', 'complication']:
            risk_distribution[risk_type] = {'low': 0, 'medium': 0, 'high': 0}
            
            for patient_data in self.patient_database.values():
                risk_score = patient_data['result'].risk_scores.get(risk_type, 0)
                if risk_score < 0.3:
                    risk_distribution[risk_type]['low'] += 1
                elif risk_score < 0.6:
                    risk_distribution[risk_type]['medium'] += 1
                else:
                    risk_distribution[risk_type]['high'] += 1
        
        # Simulate other metrics
        treatment_outcomes = {
            'antibiotic_therapy': {'success_rate': 0.85, 'complication_rate': 0.05},
            'surgical_intervention': {'success_rate': 0.80, 'complication_rate': 0.10}
        }
        
        health_trends = {
            'disease_incidence': [(datetime.now() - timedelta(days=30), 0.15),
                                (datetime.now(), 0.18)]
        }
        
        disparities_analysis = {
            'age_groups': {'<50': 0.12, '50-70': 0.18, '>70': 0.25},
            'gender': {'male': 0.16, 'female': 0.14}
        }
        
        cost_effectiveness = {
            'early_intervention': 0.85,
            'preventive_care': 0.92,
            'population_screening': 0.78
        }
        
        return PopulationHealthMetrics(
            total_patients=total_patients,
            disease_prevalence=disease_prevalence,
            risk_distribution=risk_distribution,
            treatment_outcomes=treatment_outcomes,
            health_trends=health_trends,
            disparities_analysis=disparities_analysis,
            cost_effectiveness=cost_effectiveness
        )


def create_demo_patient_profiles(num_patients: int = 5) -> List[PatientProfile]:
    """Create demo patient profiles for testing."""
    
    profiles = []
    
    for i in range(num_patients):
        profile = PatientProfile(
            patient_id=f"DEMO_{i:03d}",
            age=np.random.randint(25, 85),
            gender=np.random.choice(['male', 'female']),
            medical_history=np.random.choice([
                ['hypertension'], ['diabetes'], ['copd'], ['heart_disease'],
                ['hypertension', 'diabetes'], []
            ]).tolist(),
            current_symptoms=np.random.choice([
                ['cough', 'fever'], ['shortness_of_breath'], ['chest_pain'],
                ['cough', 'shortness_of_breath', 'fever'], ['fatigue']
            ]).tolist(),
            vital_signs={
                'heart_rate': np.random.normal(80, 15),
                'blood_pressure_systolic': np.random.normal(130, 20),
                'respiratory_rate': np.random.normal(16, 4),
                'temperature': np.random.normal(98.6, 1.5),
                'oxygen_saturation': np.random.normal(96, 3)
            },
            lab_results={
                'white_blood_cell_count': np.random.normal(7.5, 2),
                'creatinine': np.random.normal(1.0, 0.3),
                'glucose': np.random.normal(100, 20)
            },
            imaging_findings={
                'chest_xray': 'consolidation' if i % 3 == 0 else 'clear'
            },
            medications=np.random.choice([
                ['lisinopril'], ['metformin'], ['albuterol'],
                ['lisinopril', 'metformin'], []
            ]).tolist(),
            social_determinants={
                'insurance_status': np.random.choice(['insured', 'uninsured']),
                'housing_status': np.random.choice(['stable', 'unstable']),
                'transportation_access': np.random.choice(['good', 'limited']),
                'social_support': np.random.choice(['good', 'limited'])
            }
        )
        
        profiles.append(profile)
    
    return profiles


def main():
    """Demonstrate predictive medical analytics engine."""
    print("🧠 Predictive Medical Analytics Engine")
    print("=" * 45)
    
    # Initialize engine
    analytics_engine = PredictiveMedicalAnalyticsEngine()
    
    # Create demo patient profiles
    print("👥 Creating demo patient profiles...")
    patient_profiles = create_demo_patient_profiles(8)
    
    # Analyze each patient
    print("\n🔍 Analyzing patients...")
    for i, patient in enumerate(patient_profiles):
        print(f"\nPatient {i+1}: {patient.patient_id}")
        print(f"Age: {patient.age}, Gender: {patient.gender}")
        print(f"Symptoms: {', '.join(patient.current_symptoms)}")
        
        # Create clinical context
        clinical_context = {
            'suspected_diagnosis': 'pneumonia',
            'proposed_treatments': ['antibiotic_therapy', 'supportive_care'],
            'patient_history': [
                {'timestamp': time.time() - 86400, 'symptom_severity': 0.3},
                {'timestamp': time.time() - 43200, 'symptom_severity': 0.5},
                {'timestamp': time.time(), 'symptom_severity': 0.7}
            ]
        }
        
        # Perform analysis
        result = analytics_engine.analyze_patient(patient, clinical_context)
        
        # Display key results
        primary_diagnosis = max(result.primary_diagnosis_probability.items(), key=lambda x: x[1])
        print(f"Primary diagnosis: {primary_diagnosis[0]} ({primary_diagnosis[1]:.1%} probability)")
        
        highest_risk = max(result.risk_scores.items(), key=lambda x: x[1])
        print(f"Highest risk: {highest_risk[0]} ({highest_risk[1]:.1%})")
        
        if result.clinical_alerts:
            print(f"Alerts: {len(result.clinical_alerts)} clinical alerts")
        
        print(f"Confidence: {result.confidence_score:.1%}")
        
        if result.treatment_recommendations:
            best_treatment = result.treatment_recommendations[0]
            print(f"Recommended: {best_treatment['treatment']} "
                  f"({best_treatment['success_probability']:.1%} success rate)")
    
    # Generate population analytics
    print("\n📊 Generating population health analytics...")
    population_metrics = analytics_engine.generate_population_health_analytics()
    
    print(f"Total patients analyzed: {population_metrics.total_patients}")
    print("Disease prevalence:")
    for disease, prevalence in population_metrics.disease_prevalence.items():
        print(f"  • {disease}: {prevalence:.1%}")
    
    print("Risk distribution (high-risk patients):")
    for risk_type, distribution in population_metrics.risk_distribution.items():
        high_risk_count = distribution['high']
        print(f"  • {risk_type}: {high_risk_count} patients")
    
    print("\n✅ Predictive medical analytics demonstration complete!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()