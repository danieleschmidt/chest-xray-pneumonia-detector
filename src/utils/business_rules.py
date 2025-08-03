"""
Business rules engine for pneumonia detection system.
Implements medical guidelines and validation rules for clinical decision support.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import numpy as np


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Clinical risk levels for pneumonia assessment."""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"


class UrgencyLevel(Enum):
    """Clinical urgency levels for follow-up actions."""
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENT = "emergent"
    IMMEDIATE = "immediate"


@dataclass
class PatientContext:
    """Patient context for clinical decision making."""
    age: Optional[int] = None
    gender: Optional[str] = None
    symptoms: List[str] = None
    vital_signs: Dict[str, float] = None
    medical_history: List[str] = None
    current_medications: List[str] = None
    imaging_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.symptoms is None:
            self.symptoms = []
        if self.vital_signs is None:
            self.vital_signs = {}
        if self.medical_history is None:
            self.medical_history = []
        if self.current_medications is None:
            self.current_medications = []
        if self.imaging_history is None:
            self.imaging_history = []


@dataclass
class ClinicalRecommendation:
    """Clinical recommendation output from business rules."""
    primary_action: str
    urgency_level: UrgencyLevel
    risk_level: RiskLevel
    confidence_modifier: float  # Multiplier for AI confidence based on clinical context
    additional_tests: List[str]
    follow_up_timeline: str
    contraindications: List[str]
    clinical_notes: List[str]
    risk_factors: Dict[str, float]


class PneumoniaBusinessRules:
    """Business rules engine for pneumonia detection and clinical decision support."""
    
    def __init__(self):
        self.risk_factors = {
            "age_over_65": 1.5,
            "age_under_5": 1.8,
            "immunocompromised": 2.0,
            "chronic_lung_disease": 1.7,
            "heart_disease": 1.4,
            "diabetes": 1.3,
            "smoking_history": 1.6,
            "recent_hospitalization": 1.5,
            "poor_image_quality": 0.7
        }
        
        self.symptom_weights = {
            "fever": 1.4,
            "cough": 1.2,
            "shortness_of_breath": 1.6,
            "chest_pain": 1.3,
            "fatigue": 1.1,
            "confusion": 1.8,  # Especially in elderly
            "rapid_breathing": 1.5,
            "rapid_heart_rate": 1.3
        }
    
    def evaluate_clinical_context(
        self, 
        ai_prediction: str,
        ai_confidence: float,
        patient_context: Optional[PatientContext] = None,
        image_quality_score: float = 1.0
    ) -> ClinicalRecommendation:
        """
        Evaluate clinical context and generate comprehensive recommendations.
        
        Args:
            ai_prediction: AI model prediction (PNEUMONIA/NORMAL)
            ai_confidence: AI confidence score (0-1)
            patient_context: Patient clinical information
            image_quality_score: Image quality assessment (0-1)
            
        Returns:
            Clinical recommendation with risk assessment and actions
        """
        # Initialize risk assessment
        base_risk = self._calculate_base_risk(ai_prediction, ai_confidence)
        risk_factors = {}
        
        # Apply patient-specific risk factors
        if patient_context:
            patient_risk_multiplier, patient_risk_factors = self._assess_patient_risk(patient_context)
            base_risk *= patient_risk_multiplier
            risk_factors.update(patient_risk_factors)
        
        # Apply image quality factor
        if image_quality_score < 0.7:
            base_risk *= self.risk_factors["poor_image_quality"]
            risk_factors["poor_image_quality"] = self.risk_factors["poor_image_quality"]
        
        # Determine risk level
        risk_level = self._categorize_risk(base_risk)
        
        # Generate clinical recommendations
        return self._generate_recommendations(
            ai_prediction, ai_confidence, risk_level, risk_factors, 
            patient_context, image_quality_score
        )
    
    def _calculate_base_risk(self, prediction: str, confidence: float) -> float:
        """Calculate base risk score from AI prediction."""
        if prediction == "PNEUMONIA":
            # Higher confidence = higher risk for positive predictions
            return confidence * 2.0
        else:
            # Lower confidence = higher risk for negative predictions
            return (1 - confidence) * 1.5
    
    def _assess_patient_risk(self, patient_context: PatientContext) -> Tuple[float, Dict[str, float]]:
        """Assess patient-specific risk factors."""
        risk_multiplier = 1.0
        identified_factors = {}
        
        # Age-based risk assessment
        if patient_context.age:
            if patient_context.age >= 65:
                risk_multiplier *= self.risk_factors["age_over_65"]
                identified_factors["age_over_65"] = self.risk_factors["age_over_65"]
            elif patient_context.age <= 5:
                risk_multiplier *= self.risk_factors["age_under_5"]
                identified_factors["age_under_5"] = self.risk_factors["age_under_5"]
        
        # Medical history risk factors
        for condition in patient_context.medical_history:
            condition_lower = condition.lower()
            if any(term in condition_lower for term in ["immunocompromised", "hiv", "cancer", "transplant"]):
                risk_multiplier *= self.risk_factors["immunocompromised"]
                identified_factors["immunocompromised"] = self.risk_factors["immunocompromised"]
            elif any(term in condition_lower for term in ["copd", "asthma", "lung", "respiratory"]):
                risk_multiplier *= self.risk_factors["chronic_lung_disease"]
                identified_factors["chronic_lung_disease"] = self.risk_factors["chronic_lung_disease"]
            elif any(term in condition_lower for term in ["heart", "cardiac", "coronary"]):
                risk_multiplier *= self.risk_factors["heart_disease"]
                identified_factors["heart_disease"] = self.risk_factors["heart_disease"]
            elif "diabetes" in condition_lower:
                risk_multiplier *= self.risk_factors["diabetes"]
                identified_factors["diabetes"] = self.risk_factors["diabetes"]
            elif any(term in condition_lower for term in ["smoking", "tobacco"]):
                risk_multiplier *= self.risk_factors["smoking_history"]
                identified_factors["smoking_history"] = self.risk_factors["smoking_history"]
        
        # Symptom-based risk assessment
        symptom_risk = 1.0
        for symptom in patient_context.symptoms:
            symptom_lower = symptom.lower()
            for key_symptom, weight in self.symptom_weights.items():
                if key_symptom.replace("_", " ") in symptom_lower:
                    symptom_risk *= weight
                    identified_factors[f"symptom_{key_symptom}"] = weight
        
        risk_multiplier *= min(symptom_risk, 3.0)  # Cap symptom risk multiplier
        
        # Vital signs assessment
        if patient_context.vital_signs:
            vital_risk = self._assess_vital_signs(patient_context.vital_signs)
            risk_multiplier *= vital_risk
            if vital_risk > 1.0:
                identified_factors["abnormal_vitals"] = vital_risk
        
        return risk_multiplier, identified_factors
    
    def _assess_vital_signs(self, vital_signs: Dict[str, float]) -> float:
        """Assess risk based on vital signs."""
        risk_multiplier = 1.0
        
        # Temperature assessment
        if "temperature" in vital_signs:
            temp = vital_signs["temperature"]
            if temp >= 38.0:  # Fever in Celsius
                risk_multiplier *= 1.3
            elif temp >= 39.0:  # High fever
                risk_multiplier *= 1.6
        
        # Respiratory rate assessment
        if "respiratory_rate" in vital_signs:
            rr = vital_signs["respiratory_rate"]
            if rr > 20:  # Tachypnea
                risk_multiplier *= 1.4
            elif rr > 30:  # Severe tachypnea
                risk_multiplier *= 1.8
        
        # Heart rate assessment
        if "heart_rate" in vital_signs:
            hr = vital_signs["heart_rate"]
            if hr > 100:  # Tachycardia
                risk_multiplier *= 1.2
            elif hr > 120:  # Severe tachycardia
                risk_multiplier *= 1.5
        
        # Oxygen saturation assessment
        if "oxygen_saturation" in vital_signs:
            spo2 = vital_signs["oxygen_saturation"]
            if spo2 < 95:  # Hypoxemia
                risk_multiplier *= 1.6
            elif spo2 < 90:  # Severe hypoxemia
                risk_multiplier *= 2.0
        
        return risk_multiplier
    
    def _categorize_risk(self, risk_score: float) -> RiskLevel:
        """Categorize risk score into clinical risk levels."""
        if risk_score >= 3.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 2.0:
            return RiskLevel.HIGH
        elif risk_score >= 1.2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(
        self,
        prediction: str,
        confidence: float,
        risk_level: RiskLevel,
        risk_factors: Dict[str, float],
        patient_context: Optional[PatientContext],
        image_quality_score: float
    ) -> ClinicalRecommendation:
        """Generate comprehensive clinical recommendations."""
        
        # Determine primary action and urgency
        if prediction == "PNEUMONIA":
            if risk_level == RiskLevel.CRITICAL:
                primary_action = "Immediate clinical evaluation and treatment initiation"
                urgency = UrgencyLevel.IMMEDIATE
                follow_up = "Within 1 hour"
            elif risk_level == RiskLevel.HIGH:
                primary_action = "Urgent clinical evaluation required"
                urgency = UrgencyLevel.EMERGENT
                follow_up = "Within 4 hours"
            elif risk_level == RiskLevel.MODERATE:
                primary_action = "Clinical evaluation recommended"
                urgency = UrgencyLevel.URGENT
                follow_up = "Within 24 hours"
            else:
                primary_action = "Consider clinical correlation"
                urgency = UrgencyLevel.ROUTINE
                follow_up = "Within 1-2 days"
        else:  # NORMAL
            if risk_level == RiskLevel.HIGH and confidence < 0.8:
                primary_action = "Consider repeat imaging or additional evaluation"
                urgency = UrgencyLevel.URGENT
                follow_up = "Within 24 hours"
            elif risk_level == RiskLevel.MODERATE:
                primary_action = "Monitor symptoms, consider follow-up if worsening"
                urgency = UrgencyLevel.ROUTINE
                follow_up = "Within 3-5 days if symptoms persist"
            else:
                primary_action = "Continue routine care"
                urgency = UrgencyLevel.ROUTINE
                follow_up = "As clinically indicated"
        
        # Determine additional tests
        additional_tests = self._recommend_additional_tests(
            prediction, confidence, risk_level, patient_context
        )
        
        # Generate clinical notes
        clinical_notes = self._generate_clinical_notes(
            prediction, confidence, risk_level, risk_factors, image_quality_score
        )
        
        # Assess contraindications
        contraindications = self._assess_contraindications(patient_context)
        
        # Calculate confidence modifier
        confidence_modifier = self._calculate_confidence_modifier(
            risk_level, image_quality_score, patient_context
        )
        
        return ClinicalRecommendation(
            primary_action=primary_action,
            urgency_level=urgency,
            risk_level=risk_level,
            confidence_modifier=confidence_modifier,
            additional_tests=additional_tests,
            follow_up_timeline=follow_up,
            contraindications=contraindications,
            clinical_notes=clinical_notes,
            risk_factors=risk_factors
        )
    
    def _recommend_additional_tests(
        self,
        prediction: str,
        confidence: float,
        risk_level: RiskLevel,
        patient_context: Optional[PatientContext]
    ) -> List[str]:
        """Recommend additional diagnostic tests."""
        tests = []
        
        if prediction == "PNEUMONIA" or (prediction == "NORMAL" and confidence < 0.7):
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                tests.extend([
                    "Complete blood count with differential",
                    "Blood cultures",
                    "Arterial blood gas analysis",
                    "Procalcitonin level"
                ])
            
            if risk_level == RiskLevel.CRITICAL:
                tests.extend([
                    "Lactate level",
                    "Comprehensive metabolic panel",
                    "Consider CT chest if pneumonia severity unclear"
                ])
        
        if patient_context and patient_context.symptoms:
            if any("shortness" in s.lower() for s in patient_context.symptoms):
                tests.append("Pulse oximetry monitoring")
            
            if any("chest pain" in s.lower() for s in patient_context.symptoms):
                tests.append("Consider ECG to rule out cardiac etiology")
        
        return tests
    
    def _generate_clinical_notes(
        self,
        prediction: str,
        confidence: float,
        risk_level: RiskLevel,
        risk_factors: Dict[str, float],
        image_quality_score: float
    ) -> List[str]:
        """Generate clinical notes for documentation."""
        notes = []
        
        # AI prediction note
        notes.append(f"AI analysis suggests {prediction} with {confidence:.1%} confidence")
        
        # Risk assessment note
        notes.append(f"Clinical risk assessment: {risk_level.value.upper()}")
        
        # Risk factors
        if risk_factors:
            factor_list = [factor.replace("_", " ").title() for factor in risk_factors.keys()]
            notes.append(f"Identified risk factors: {', '.join(factor_list)}")
        
        # Image quality note
        if image_quality_score < 0.7:
            notes.append("Note: Image quality may limit diagnostic accuracy")
        
        # Confidence considerations
        if confidence < 0.6:
            notes.append("Low AI confidence - clinical correlation strongly recommended")
        elif confidence > 0.95:
            notes.append("High AI confidence - consistent with clinical findings")
        
        return notes
    
    def _assess_contraindications(self, patient_context: Optional[PatientContext]) -> List[str]:
        """Assess contraindications for standard treatments."""
        contraindications = []
        
        if not patient_context:
            return contraindications
        
        # Check medication allergies and interactions
        for medication in patient_context.current_medications:
            med_lower = medication.lower()
            if any(term in med_lower for term in ["warfarin", "anticoagulant"]):
                contraindications.append("Caution with antibiotics that may interact with anticoagulation")
            elif any(term in med_lower for term in ["steroid", "prednisone"]):
                contraindications.append("Consider steroid interaction with antibiotic selection")
        
        # Check medical history for contraindications
        for condition in patient_context.medical_history:
            condition_lower = condition.lower()
            if "kidney" in condition_lower or "renal" in condition_lower:
                contraindications.append("Adjust antibiotic dosing for renal function")
            elif "liver" in condition_lower or "hepatic" in condition_lower:
                contraindications.append("Consider hepatic dosing adjustments")
        
        return contraindications
    
    def _calculate_confidence_modifier(
        self,
        risk_level: RiskLevel,
        image_quality_score: float,
        patient_context: Optional[PatientContext]
    ) -> float:
        """Calculate modifier for AI confidence based on clinical context."""
        modifier = 1.0
        
        # Adjust based on risk level
        if risk_level == RiskLevel.HIGH:
            modifier *= 1.1  # Slightly increase confidence weight for high-risk cases
        elif risk_level == RiskLevel.LOW:
            modifier *= 0.9  # Slightly decrease confidence weight for low-risk cases
        
        # Adjust based on image quality
        modifier *= max(0.5, image_quality_score)
        
        # Adjust based on symptom consistency
        if patient_context and patient_context.symptoms:
            respiratory_symptoms = sum(1 for s in patient_context.symptoms 
                                     if any(term in s.lower() for term in 
                                           ["cough", "shortness", "chest", "breathing"]))
            if respiratory_symptoms >= 2:
                modifier *= 1.05  # Increase confidence if respiratory symptoms present
        
        return modifier