"""
Global Medical Compliance Framework
==================================

Comprehensive regulatory compliance system for medical AI deployment across
global jurisdictions including FDA, CE, Health Canada, TGA, and other
international medical device regulations.

Key Compliance Areas:
1. FDA 510(k) and De Novo pathway compliance
2. EU MDR (Medical Device Regulation) compliance  
3. ISO 13485 (Quality Management Systems)
4. ISO 14971 (Risk Management)
5. IEC 62304 (Medical Device Software)
6. HIPAA, GDPR, and data protection regulations
7. Clinical validation requirements
8. Post-market surveillance protocols
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import uuid


class RegulatoryJurisdiction(Enum):
    """Supported regulatory jurisdictions."""
    FDA_US = "FDA_US"
    CE_EU = "CE_EU"
    HEALTH_CANADA = "HEALTH_CANADA"
    TGA_AUSTRALIA = "TGA_AUSTRALIA"
    PMDA_JAPAN = "PMDA_JAPAN"
    NMPA_CHINA = "NMPA_CHINA"
    ANVISA_BRAZIL = "ANVISA_BRAZIL"
    ISO_INTERNATIONAL = "ISO_INTERNATIONAL"


class DeviceClassification(Enum):
    """Medical device classification levels."""
    CLASS_I = "CLASS_I"      # Low risk
    CLASS_II = "CLASS_II"    # Moderate risk
    CLASS_III = "CLASS_III"  # High risk


class ComplianceStatus(Enum):
    """Compliance verification status."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIAL = "PARTIAL"
    PENDING_REVIEW = "PENDING_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class RegulatoryRequirement:
    """Individual regulatory requirement."""
    requirement_id: str
    jurisdiction: RegulatoryJurisdiction
    title: str
    description: str
    applicable_device_classes: List[DeviceClassification]
    mandatory: bool
    verification_criteria: List[str]
    documentation_required: List[str]
    testing_required: List[str]
    review_frequency: str  # "annual", "biannual", "continuous"


@dataclass
class ComplianceEvidence:
    """Evidence supporting compliance with a requirement."""
    evidence_id: str
    requirement_id: str
    evidence_type: str  # "document", "test_result", "audit", "certification"
    title: str
    description: str
    file_path: Optional[str]
    verification_date: datetime
    expiry_date: Optional[datetime]
    verified_by: str
    digital_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAssessment:
    """Assessment result for a regulatory requirement."""
    assessment_id: str
    requirement_id: str
    status: ComplianceStatus
    compliance_score: float  # 0.0 to 1.0
    evidence_provided: List[str]  # Evidence IDs
    gaps_identified: List[str]
    recommendations: List[str]
    assessor: str
    assessment_date: datetime
    next_review_date: datetime
    confidence_level: float


@dataclass
class RegulatoryProfile:
    """Complete regulatory compliance profile."""
    profile_id: str
    device_name: str
    device_classification: DeviceClassification
    intended_use: str
    target_jurisdictions: List[RegulatoryJurisdiction]
    requirements: List[RegulatoryRequirement]
    evidence_database: List[ComplianceEvidence]
    assessments: List[ComplianceAssessment]
    overall_compliance_score: float
    certification_status: Dict[RegulatoryJurisdiction, ComplianceStatus]
    last_updated: datetime


class RegulatoryRequirementsDatabase:
    """Database of regulatory requirements across jurisdictions."""
    
    def __init__(self):
        self.requirements = self._initialize_requirements_database()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_requirements_database(self) -> List[RegulatoryRequirement]:
        """Initialize comprehensive regulatory requirements database."""
        
        requirements = []
        
        # FDA Requirements
        requirements.extend(self._get_fda_requirements())
        
        # EU MDR Requirements
        requirements.extend(self._get_eu_mdr_requirements())
        
        # ISO International Standards
        requirements.extend(self._get_iso_requirements())
        
        # Data Protection Requirements
        requirements.extend(self._get_data_protection_requirements())
        
        # Clinical Requirements
        requirements.extend(self._get_clinical_requirements())
        
        return requirements
    
    def _get_fda_requirements(self) -> List[RegulatoryRequirement]:
        """Get FDA-specific regulatory requirements."""
        
        return [
            RegulatoryRequirement(
                requirement_id="FDA_510K_001",
                jurisdiction=RegulatoryJurisdiction.FDA_US,
                title="510(k) Predicate Device Identification",
                description="Identification of substantially equivalent predicate device(s)",
                applicable_device_classes=[DeviceClassification.CLASS_II],
                mandatory=True,
                verification_criteria=[
                    "Predicate device identified with 510(k) number",
                    "Substantial equivalence demonstrated",
                    "Comparative analysis provided"
                ],
                documentation_required=[
                    "Predicate device comparison table",
                    "Substantial equivalence analysis",
                    "Clinical performance comparison"
                ],
                testing_required=[
                    "Performance testing vs predicate",
                    "Software validation testing"
                ],
                review_frequency="once"
            ),
            
            RegulatoryRequirement(
                requirement_id="FDA_QSR_001",
                jurisdiction=RegulatoryJurisdiction.FDA_US,
                title="Quality System Regulation (21 CFR 820)",
                description="Establishment and maintenance of quality system",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "QSR procedures documented",
                    "Design controls implemented",
                    "CAPA system operational"
                ],
                documentation_required=[
                    "Quality manual",
                    "Design control procedures",
                    "CAPA procedures"
                ],
                testing_required=[
                    "Design verification testing",
                    "Design validation testing"
                ],
                review_frequency="annual"
            ),
            
            RegulatoryRequirement(
                requirement_id="FDA_SOFTWARE_001",
                jurisdiction=RegulatoryJurisdiction.FDA_US,
                title="Software as Medical Device (SaMD) Guidance",
                description="Software lifecycle processes and documentation",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Software lifecycle plan documented",
                    "Software requirements specified",
                    "Software architecture documented",
                    "Software testing completed"
                ],
                documentation_required=[
                    "Software design specification",
                    "Software verification and validation plan",
                    "Software risk analysis"
                ],
                testing_required=[
                    "Software unit testing",
                    "Integration testing",
                    "System testing",
                    "Cybersecurity testing"
                ],
                review_frequency="continuous"
            )
        ]
    
    def _get_eu_mdr_requirements(self) -> List[RegulatoryRequirement]:
        """Get EU MDR regulatory requirements."""
        
        return [
            RegulatoryRequirement(
                requirement_id="EU_MDR_001",
                jurisdiction=RegulatoryJurisdiction.CE_EU,
                title="CE Marking Declaration of Conformity",
                description="Declaration of conformity with EU MDR requirements",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Declaration of conformity signed",
                    "CE marking affixed to device",
                    "Notified body approval (if required)"
                ],
                documentation_required=[
                    "Declaration of conformity",
                    "Technical documentation",
                    "Post-market surveillance plan"
                ],
                testing_required=[
                    "Conformity assessment",
                    "Clinical evaluation"
                ],
                review_frequency="annual"
            ),
            
            RegulatoryRequirement(
                requirement_id="EU_MDR_002",
                jurisdiction=RegulatoryJurisdiction.CE_EU,
                title="Unique Device Identification (UDI)",
                description="UDI system implementation for device traceability",
                applicable_device_classes=[DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "UDI assigned to device",
                    "UDI database registration completed",
                    "UDI carriers implemented"
                ],
                documentation_required=[
                    "UDI assignment documentation",
                    "Database registration confirmation"
                ],
                testing_required=[
                    "UDI readability testing",
                    "Traceability verification"
                ],
                review_frequency="continuous"
            ),
            
            RegulatoryRequirement(
                requirement_id="EU_MDR_003",
                jurisdiction=RegulatoryJurisdiction.CE_EU,
                title="Clinical Evidence Requirements",
                description="Clinical evaluation and post-market clinical follow-up",
                applicable_device_classes=[DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Clinical evaluation report completed",
                    "Clinical data supports intended use",
                    "PMCF plan implemented"
                ],
                documentation_required=[
                    "Clinical evaluation report",
                    "Clinical investigation plan",
                    "PMCF plan and reports"
                ],
                testing_required=[
                    "Clinical performance testing",
                    "Usability validation"
                ],
                review_frequency="annual"
            )
        ]
    
    def _get_iso_requirements(self) -> List[RegulatoryRequirement]:
        """Get ISO international standard requirements."""
        
        return [
            RegulatoryRequirement(
                requirement_id="ISO_13485_001",
                jurisdiction=RegulatoryJurisdiction.ISO_INTERNATIONAL,
                title="ISO 13485 Quality Management System",
                description="Quality management system for medical devices",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "QMS documented and implemented",
                    "Management responsibility defined",
                    "Resource management adequate"
                ],
                documentation_required=[
                    "Quality manual",
                    "Process procedures",
                    "Management review records"
                ],
                testing_required=[
                    "Internal audit",
                    "Management review",
                    "Third-party certification audit"
                ],
                review_frequency="annual"
            ),
            
            RegulatoryRequirement(
                requirement_id="ISO_14971_001",
                jurisdiction=RegulatoryJurisdiction.ISO_INTERNATIONAL,
                title="ISO 14971 Risk Management",
                description="Risk management processes for medical devices",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Risk management plan documented",
                    "Risk analysis completed",
                    "Risk control measures implemented"
                ],
                documentation_required=[
                    "Risk management file",
                    "Risk analysis report",
                    "Risk control verification"
                ],
                testing_required=[
                    "Risk control effectiveness testing",
                    "Residual risk evaluation"
                ],
                review_frequency="continuous"
            ),
            
            RegulatoryRequirement(
                requirement_id="IEC_62304_001",
                jurisdiction=RegulatoryJurisdiction.ISO_INTERNATIONAL,
                title="IEC 62304 Medical Device Software",
                description="Software lifecycle processes for medical device software",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Software lifecycle plan implemented",
                    "Software safety classification performed",
                    "Software architecture documented"
                ],
                documentation_required=[
                    "Software development plan",
                    "Software requirements specification",
                    "Software architecture document"
                ],
                testing_required=[
                    "Software unit testing",
                    "Software integration testing",
                    "Software system testing"
                ],
                review_frequency="continuous"
            )
        ]
    
    def _get_data_protection_requirements(self) -> List[RegulatoryRequirement]:
        """Get data protection regulatory requirements."""
        
        return [
            RegulatoryRequirement(
                requirement_id="GDPR_001",
                jurisdiction=RegulatoryJurisdiction.CE_EU,
                title="GDPR Data Protection Compliance",
                description="General Data Protection Regulation compliance",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Privacy policy documented",
                    "Data processing lawful basis identified",
                    "Data subject rights procedures implemented"
                ],
                documentation_required=[
                    "Privacy impact assessment",
                    "Data processing records",
                    "Consent management procedures"
                ],
                testing_required=[
                    "Data protection audit",
                    "Security vulnerability assessment"
                ],
                review_frequency="annual"
            ),
            
            RegulatoryRequirement(
                requirement_id="HIPAA_001",
                jurisdiction=RegulatoryJurisdiction.FDA_US,
                title="HIPAA Privacy and Security Rules",
                description="Health Insurance Portability and Accountability Act compliance",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Administrative safeguards implemented",
                    "Physical safeguards implemented",
                    "Technical safeguards implemented"
                ],
                documentation_required=[
                    "HIPAA risk assessment",
                    "Security policies and procedures",
                    "Business associate agreements"
                ],
                testing_required=[
                    "Security risk assessment",
                    "Penetration testing",
                    "Access control testing"
                ],
                review_frequency="annual"
            )
        ]
    
    def _get_clinical_requirements(self) -> List[RegulatoryRequirement]:
        """Get clinical validation requirements."""
        
        return [
            RegulatoryRequirement(
                requirement_id="CLINICAL_001",
                jurisdiction=RegulatoryJurisdiction.ISO_INTERNATIONAL,
                title="Clinical Validation Study",
                description="Clinical performance validation in intended use environment",
                applicable_device_classes=[DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Clinical study protocol approved",
                    "Clinical data collected and analyzed",
                    "Clinical performance meets specifications"
                ],
                documentation_required=[
                    "Clinical investigation plan",
                    "Clinical study report",
                    "Statistical analysis plan"
                ],
                testing_required=[
                    "Clinical performance testing",
                    "Usability validation",
                    "Real-world evidence collection"
                ],
                review_frequency="once"
            ),
            
            RegulatoryRequirement(
                requirement_id="CLINICAL_002",
                jurisdiction=RegulatoryJurisdiction.ISO_INTERNATIONAL,
                title="Post-Market Surveillance",
                description="Ongoing monitoring of device performance in clinical use",
                applicable_device_classes=[DeviceClassification.CLASS_I, DeviceClassification.CLASS_II, DeviceClassification.CLASS_III],
                mandatory=True,
                verification_criteria=[
                    "Surveillance plan implemented",
                    "Adverse events monitored",
                    "Performance data collected"
                ],
                documentation_required=[
                    "Post-market surveillance plan",
                    "Adverse event reports",
                    "Periodic safety update reports"
                ],
                testing_required=[
                    "Real-world performance monitoring",
                    "User feedback collection",
                    "Safety signal detection"
                ],
                review_frequency="continuous"
            )
        ]
    
    def get_requirements_for_jurisdiction(self, jurisdiction: RegulatoryJurisdiction) -> List[RegulatoryRequirement]:
        """Get all requirements for a specific jurisdiction."""
        
        return [req for req in self.requirements if req.jurisdiction == jurisdiction]
    
    def get_requirements_for_device_class(self, device_class: DeviceClassification) -> List[RegulatoryRequirement]:
        """Get all requirements applicable to a device class."""
        
        return [req for req in self.requirements if device_class in req.applicable_device_classes]


class ComplianceVerificationEngine:
    """Engine for verifying compliance with regulatory requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def assess_requirement_compliance(self, 
                                    requirement: RegulatoryRequirement,
                                    evidence_list: List[ComplianceEvidence],
                                    device_info: Dict[str, Any]) -> ComplianceAssessment:
        """Assess compliance with a specific regulatory requirement."""
        
        assessment_id = str(uuid.uuid4())
        assessment_date = datetime.now()
        
        # Find relevant evidence
        relevant_evidence = [
            evidence for evidence in evidence_list 
            if evidence.requirement_id == requirement.requirement_id
        ]
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(requirement, relevant_evidence)
        
        # Determine compliance status
        status = self._determine_compliance_status(compliance_score)
        
        # Identify gaps
        gaps = self._identify_compliance_gaps(requirement, relevant_evidence)
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(requirement, gaps, compliance_score)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(relevant_evidence, requirement)
        
        # Determine next review date
        next_review_date = self._calculate_next_review_date(requirement, assessment_date)
        
        return ComplianceAssessment(
            assessment_id=assessment_id,
            requirement_id=requirement.requirement_id,
            status=status,
            compliance_score=compliance_score,
            evidence_provided=[evidence.evidence_id for evidence in relevant_evidence],
            gaps_identified=gaps,
            recommendations=recommendations,
            assessor="Automated Compliance Engine",
            assessment_date=assessment_date,
            next_review_date=next_review_date,
            confidence_level=confidence_level
        )
    
    def _calculate_compliance_score(self, 
                                  requirement: RegulatoryRequirement,
                                  evidence_list: List[ComplianceEvidence]) -> float:
        """Calculate compliance score for a requirement."""
        
        if not requirement.verification_criteria:
            return 1.0
        
        total_criteria = len(requirement.verification_criteria)
        met_criteria = 0
        
        # Check each verification criterion
        for criterion in requirement.verification_criteria:
            # Look for evidence that addresses this criterion
            criterion_met = any(
                criterion.lower() in evidence.description.lower() or
                criterion.lower() in evidence.title.lower()
                for evidence in evidence_list
            )
            
            if criterion_met:
                met_criteria += 1
        
        # Check documentation requirements
        doc_score = 1.0
        if requirement.documentation_required:
            provided_docs = len([e for e in evidence_list if e.evidence_type == "document"])
            required_docs = len(requirement.documentation_required)
            doc_score = min(1.0, provided_docs / required_docs)
        
        # Check testing requirements
        test_score = 1.0
        if requirement.testing_required:
            provided_tests = len([e for e in evidence_list if e.evidence_type == "test_result"])
            required_tests = len(requirement.testing_required)
            test_score = min(1.0, provided_tests / required_tests)
        
        # Weighted average
        criteria_score = met_criteria / total_criteria
        overall_score = (criteria_score * 0.5 + doc_score * 0.3 + test_score * 0.2)
        
        return min(1.0, overall_score)
    
    def _determine_compliance_status(self, compliance_score: float) -> ComplianceStatus:
        """Determine compliance status based on score."""
        
        if compliance_score >= 0.95:
            return ComplianceStatus.COMPLIANT
        elif compliance_score >= 0.75:
            return ComplianceStatus.PARTIAL
        elif compliance_score >= 0.5:
            return ComplianceStatus.NON_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    def _identify_compliance_gaps(self, 
                                requirement: RegulatoryRequirement,
                                evidence_list: List[ComplianceEvidence]) -> List[str]:
        """Identify gaps in compliance evidence."""
        
        gaps = []
        
        # Check missing verification criteria
        for criterion in requirement.verification_criteria:
            criterion_addressed = any(
                criterion.lower() in evidence.description.lower() or
                criterion.lower() in evidence.title.lower()
                for evidence in evidence_list
            )
            
            if not criterion_addressed:
                gaps.append(f"Missing evidence for: {criterion}")
        
        # Check missing documentation
        provided_doc_types = set(e.title.lower() for e in evidence_list if e.evidence_type == "document")
        for required_doc in requirement.documentation_required:
            if not any(required_doc.lower() in doc_type for doc_type in provided_doc_types):
                gaps.append(f"Missing required documentation: {required_doc}")
        
        # Check missing testing
        provided_test_types = set(e.title.lower() for e in evidence_list if e.evidence_type == "test_result")
        for required_test in requirement.testing_required:
            if not any(required_test.lower() in test_type for test_type in provided_test_types):
                gaps.append(f"Missing required testing: {required_test}")
        
        # Check for expired evidence
        current_date = datetime.now()
        for evidence in evidence_list:
            if evidence.expiry_date and evidence.expiry_date < current_date:
                gaps.append(f"Expired evidence: {evidence.title}")
        
        return gaps
    
    def _generate_compliance_recommendations(self, 
                                           requirement: RegulatoryRequirement,
                                           gaps: List[str],
                                           compliance_score: float) -> List[str]:
        """Generate recommendations for improving compliance."""
        
        recommendations = []
        
        if compliance_score < 0.5:
            recommendations.append("Critical compliance gaps identified - immediate action required")
        elif compliance_score < 0.75:
            recommendations.append("Significant compliance improvements needed")
        elif compliance_score < 0.95:
            recommendations.append("Minor compliance improvements recommended")
        
        if gaps:
            recommendations.append(f"Address {len(gaps)} identified compliance gaps")
            
            # Prioritize critical gaps
            critical_gaps = [gap for gap in gaps if "missing required" in gap.lower()]
            if critical_gaps:
                recommendations.append("Priority: Address missing required documentation and testing")
        
        # Jurisdiction-specific recommendations
        if requirement.jurisdiction == RegulatoryJurisdiction.FDA_US:
            if compliance_score < 0.8:
                recommendations.append("Consider pre-submission meeting with FDA")
        elif requirement.jurisdiction == RegulatoryJurisdiction.CE_EU:
            if compliance_score < 0.8:
                recommendations.append("Consider consultation with notified body")
        
        return recommendations
    
    def _calculate_confidence_level(self, 
                                  evidence_list: List[ComplianceEvidence],
                                  requirement: RegulatoryRequirement) -> float:
        """Calculate confidence level in the compliance assessment."""
        
        confidence_factors = []
        
        # Evidence quality factor
        if evidence_list:
            verified_evidence = [e for e in evidence_list if e.verified_by]
            evidence_quality = len(verified_evidence) / len(evidence_list)
            confidence_factors.append(evidence_quality)
        else:
            confidence_factors.append(0.0)
        
        # Evidence currency factor
        current_date = datetime.now()
        recent_evidence = [
            e for e in evidence_list 
            if (current_date - e.verification_date).days <= 365
        ]
        
        if evidence_list:
            currency_factor = len(recent_evidence) / len(evidence_list)
            confidence_factors.append(currency_factor)
        else:
            confidence_factors.append(0.0)
        
        # Evidence completeness factor
        completeness = min(1.0, len(evidence_list) / max(1, len(requirement.verification_criteria)))
        confidence_factors.append(completeness)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_next_review_date(self, 
                                  requirement: RegulatoryRequirement,
                                  assessment_date: datetime) -> datetime:
        """Calculate the next review date based on requirement frequency."""
        
        frequency_map = {
            "continuous": timedelta(days=90),
            "quarterly": timedelta(days=90),
            "biannual": timedelta(days=180),
            "annual": timedelta(days=365),
            "once": timedelta(days=365 * 5)  # 5 years for one-time requirements
        }
        
        frequency = requirement.review_frequency.lower()
        time_delta = frequency_map.get(frequency, timedelta(days=365))
        
        return assessment_date + time_delta


class GlobalMedicalComplianceFramework:
    """Main framework for managing global medical device compliance."""
    
    def __init__(self):
        self.requirements_db = RegulatoryRequirementsDatabase()
        self.verification_engine = ComplianceVerificationEngine()
        self.compliance_profiles = {}
        self.logger = logging.getLogger(__name__)
        
    def create_regulatory_profile(self, 
                                device_name: str,
                                device_classification: DeviceClassification,
                                intended_use: str,
                                target_jurisdictions: List[RegulatoryJurisdiction]) -> str:
        """Create a new regulatory compliance profile."""
        
        profile_id = str(uuid.uuid4())
        
        # Get applicable requirements
        applicable_requirements = []
        for jurisdiction in target_jurisdictions:
            jurisdiction_requirements = self.requirements_db.get_requirements_for_jurisdiction(jurisdiction)
            applicable_requirements.extend(jurisdiction_requirements)
        
        # Filter by device classification
        filtered_requirements = [
            req for req in applicable_requirements 
            if device_classification in req.applicable_device_classes
        ]
        
        # Create profile
        profile = RegulatoryProfile(
            profile_id=profile_id,
            device_name=device_name,
            device_classification=device_classification,
            intended_use=intended_use,
            target_jurisdictions=target_jurisdictions,
            requirements=filtered_requirements,
            evidence_database=[],
            assessments=[],
            overall_compliance_score=0.0,
            certification_status={jurisdiction: ComplianceStatus.PENDING_REVIEW for jurisdiction in target_jurisdictions},
            last_updated=datetime.now()
        )
        
        self.compliance_profiles[profile_id] = profile
        
        self.logger.info(f"Created regulatory profile {profile_id} for {device_name}")
        
        return profile_id
    
    def add_compliance_evidence(self, 
                               profile_id: str,
                               requirement_id: str,
                               evidence_type: str,
                               title: str,
                               description: str,
                               file_path: Optional[str] = None,
                               verified_by: str = "System",
                               expiry_date: Optional[datetime] = None) -> str:
        """Add compliance evidence to a regulatory profile."""
        
        if profile_id not in self.compliance_profiles:
            raise ValueError(f"Profile {profile_id} not found")
        
        evidence_id = str(uuid.uuid4())
        
        # Generate digital signature
        evidence_content = f"{evidence_id}{requirement_id}{title}{description}{verified_by}"
        digital_signature = hashlib.sha256(evidence_content.encode()).hexdigest()
        
        evidence = ComplianceEvidence(
            evidence_id=evidence_id,
            requirement_id=requirement_id,
            evidence_type=evidence_type,
            title=title,
            description=description,
            file_path=file_path,
            verification_date=datetime.now(),
            expiry_date=expiry_date,
            verified_by=verified_by,
            digital_signature=digital_signature
        )
        
        self.compliance_profiles[profile_id].evidence_database.append(evidence)
        self.compliance_profiles[profile_id].last_updated = datetime.now()
        
        self.logger.info(f"Added evidence {evidence_id} for requirement {requirement_id}")
        
        return evidence_id
    
    def perform_compliance_assessment(self, profile_id: str) -> Dict[str, Any]:
        """Perform comprehensive compliance assessment for a profile."""
        
        if profile_id not in self.compliance_profiles:
            raise ValueError(f"Profile {profile_id} not found")
        
        profile = self.compliance_profiles[profile_id]
        
        self.logger.info(f"Starting compliance assessment for profile {profile_id}")
        
        # Assess each requirement
        new_assessments = []
        compliance_scores = []
        
        for requirement in profile.requirements:
            assessment = self.verification_engine.assess_requirement_compliance(
                requirement, profile.evidence_database, {}
            )
            
            new_assessments.append(assessment)
            compliance_scores.append(assessment.compliance_score)
        
        # Update profile with new assessments
        profile.assessments = new_assessments
        profile.overall_compliance_score = np.mean(compliance_scores) if compliance_scores else 0.0
        
        # Update certification status by jurisdiction
        jurisdiction_scores = {}
        for jurisdiction in profile.target_jurisdictions:
            jurisdiction_requirements = [req for req in profile.requirements if req.jurisdiction == jurisdiction]
            jurisdiction_assessments = [
                assessment for assessment in new_assessments
                if any(req.requirement_id == assessment.requirement_id for req in jurisdiction_requirements)
            ]
            
            if jurisdiction_assessments:
                jurisdiction_score = np.mean([a.compliance_score for a in jurisdiction_assessments])
                jurisdiction_scores[jurisdiction] = jurisdiction_score
                
                if jurisdiction_score >= 0.95:
                    profile.certification_status[jurisdiction] = ComplianceStatus.COMPLIANT
                elif jurisdiction_score >= 0.75:
                    profile.certification_status[jurisdiction] = ComplianceStatus.PARTIAL
                else:
                    profile.certification_status[jurisdiction] = ComplianceStatus.NON_COMPLIANT
        
        profile.last_updated = datetime.now()
        
        # Generate assessment summary
        assessment_summary = {
            'profile_id': profile_id,
            'assessment_date': datetime.now(),
            'overall_compliance_score': profile.overall_compliance_score,
            'total_requirements': len(profile.requirements),
            'compliant_requirements': len([a for a in new_assessments if a.status == ComplianceStatus.COMPLIANT]),
            'partial_requirements': len([a for a in new_assessments if a.status == ComplianceStatus.PARTIAL]),
            'non_compliant_requirements': len([a for a in new_assessments if a.status == ComplianceStatus.NON_COMPLIANT]),
            'jurisdiction_scores': jurisdiction_scores,
            'certification_status': profile.certification_status,
            'critical_gaps': [],
            'high_priority_recommendations': []
        }
        
        # Identify critical gaps and recommendations
        for assessment in new_assessments:
            if assessment.status == ComplianceStatus.NON_COMPLIANT:
                assessment_summary['critical_gaps'].extend(assessment.gaps_identified)
                assessment_summary['high_priority_recommendations'].extend(assessment.recommendations)
        
        self.logger.info(f"Compliance assessment completed for profile {profile_id}")
        
        return assessment_summary
    
    def generate_compliance_report(self, profile_id: str, 
                                 report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate detailed compliance report."""
        
        if profile_id not in self.compliance_profiles:
            raise ValueError(f"Profile {profile_id} not found")
        
        profile = self.compliance_profiles[profile_id]
        
        report = {
            'report_metadata': {
                'profile_id': profile_id,
                'device_name': profile.device_name,
                'device_classification': profile.device_classification.value,
                'intended_use': profile.intended_use,
                'target_jurisdictions': [j.value for j in profile.target_jurisdictions],
                'report_type': report_type,
                'generation_date': datetime.now().isoformat(),
                'overall_compliance_score': profile.overall_compliance_score
            },
            'compliance_summary': {
                'certification_status': {j.value: s.value for j, s in profile.certification_status.items()},
                'total_requirements': len(profile.requirements),
                'evidence_items': len(profile.evidence_database),
                'recent_assessments': len(profile.assessments)
            },
            'jurisdiction_analysis': {},
            'requirement_details': [],
            'evidence_inventory': [],
            'recommendations': [],
            'next_actions': []
        }
        
        # Jurisdiction-specific analysis
        for jurisdiction in profile.target_jurisdictions:
            jurisdiction_requirements = [req for req in profile.requirements if req.jurisdiction == jurisdiction]
            jurisdiction_assessments = [
                assessment for assessment in profile.assessments
                if any(req.requirement_id == assessment.requirement_id for req in jurisdiction_requirements)
            ]
            
            jurisdiction_score = np.mean([a.compliance_score for a in jurisdiction_assessments]) if jurisdiction_assessments else 0.0
            
            report['jurisdiction_analysis'][jurisdiction.value] = {
                'compliance_score': jurisdiction_score,
                'certification_status': profile.certification_status[jurisdiction].value,
                'requirements_count': len(jurisdiction_requirements),
                'compliant_count': len([a for a in jurisdiction_assessments if a.status == ComplianceStatus.COMPLIANT]),
                'gaps_count': len([a for a in jurisdiction_assessments if a.status == ComplianceStatus.NON_COMPLIANT])
            }
        
        # Requirement details
        for requirement in profile.requirements:
            # Find corresponding assessment
            assessment = next(
                (a for a in profile.assessments if a.requirement_id == requirement.requirement_id),
                None
            )
            
            requirement_detail = {
                'requirement_id': requirement.requirement_id,
                'title': requirement.title,
                'jurisdiction': requirement.jurisdiction.value,
                'mandatory': requirement.mandatory,
                'compliance_status': assessment.status.value if assessment else ComplianceStatus.PENDING_REVIEW.value,
                'compliance_score': assessment.compliance_score if assessment else 0.0,
                'gaps_identified': assessment.gaps_identified if assessment else [],
                'recommendations': assessment.recommendations if assessment else []
            }
            
            report['requirement_details'].append(requirement_detail)
        
        # Evidence inventory
        for evidence in profile.evidence_database:
            evidence_detail = {
                'evidence_id': evidence.evidence_id,
                'requirement_id': evidence.requirement_id,
                'title': evidence.title,
                'evidence_type': evidence.evidence_type,
                'verification_date': evidence.verification_date.isoformat(),
                'verified_by': evidence.verified_by,
                'expiry_date': evidence.expiry_date.isoformat() if evidence.expiry_date else None,
                'digital_signature': evidence.digital_signature
            }
            
            report['evidence_inventory'].append(evidence_detail)
        
        # Overall recommendations
        critical_requirements = [a for a in profile.assessments if a.status == ComplianceStatus.NON_COMPLIANT]
        
        if critical_requirements:
            report['recommendations'].append(f"Address {len(critical_requirements)} critical compliance gaps")
        
        partial_requirements = [a for a in profile.assessments if a.status == ComplianceStatus.PARTIAL]
        if partial_requirements:
            report['recommendations'].append(f"Improve {len(partial_requirements)} partially compliant requirements")
        
        # Next actions
        upcoming_reviews = [
            a for a in profile.assessments
            if (a.next_review_date - datetime.now()).days <= 90
        ]
        
        if upcoming_reviews:
            report['next_actions'].append(f"Schedule reviews for {len(upcoming_reviews)} requirements due within 90 days")
        
        return report
    
    def export_regulatory_documentation(self, profile_id: str, 
                                      output_directory: Path) -> Dict[str, str]:
        """Export regulatory documentation for submission."""
        
        if profile_id not in self.compliance_profiles:
            raise ValueError(f"Profile {profile_id} not found")
        
        profile = self.compliance_profiles[profile_id]
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Generate compliance report
        compliance_report = self.generate_compliance_report(profile_id, "comprehensive")
        report_file = output_directory / f"{profile.device_name}_compliance_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(compliance_report, f, indent=2, default=str)
        
        exported_files['compliance_report'] = str(report_file)
        
        # Generate regulatory submission package
        submission_package = {
            'device_information': {
                'device_name': profile.device_name,
                'device_classification': profile.device_classification.value,
                'intended_use': profile.intended_use,
                'target_jurisdictions': [j.value for j in profile.target_jurisdictions]
            },
            'compliance_evidence': [],
            'assessments': [],
            'certification_status': {j.value: s.value for j, s in profile.certification_status.items()}
        }
        
        # Add evidence details
        for evidence in profile.evidence_database:
            submission_package['compliance_evidence'].append({
                'evidence_id': evidence.evidence_id,
                'requirement_id': evidence.requirement_id,
                'title': evidence.title,
                'description': evidence.description,
                'evidence_type': evidence.evidence_type,
                'verification_date': evidence.verification_date.isoformat(),
                'verified_by': evidence.verified_by,
                'digital_signature': evidence.digital_signature
            })
        
        # Add assessment details
        for assessment in profile.assessments:
            submission_package['assessments'].append({
                'assessment_id': assessment.assessment_id,
                'requirement_id': assessment.requirement_id,
                'status': assessment.status.value,
                'compliance_score': assessment.compliance_score,
                'assessment_date': assessment.assessment_date.isoformat(),
                'confidence_level': assessment.confidence_level
            })
        
        submission_file = output_directory / f"{profile.device_name}_regulatory_submission.json"
        
        with open(submission_file, 'w') as f:
            json.dump(submission_package, f, indent=2, default=str)
        
        exported_files['submission_package'] = str(submission_file)
        
        self.logger.info(f"Exported regulatory documentation for profile {profile_id}")
        
        return exported_files


def create_demo_medical_device_profile() -> Tuple[str, GlobalMedicalComplianceFramework]:
    """Create a demo medical device compliance profile."""
    
    framework = GlobalMedicalComplianceFramework()
    
    # Create regulatory profile for AI-based pneumonia detection system
    profile_id = framework.create_regulatory_profile(
        device_name="AI Pneumonia Detection System",
        device_classification=DeviceClassification.CLASS_II,
        intended_use="Computer-aided detection of pneumonia in chest X-ray images for diagnostic support",
        target_jurisdictions=[
            RegulatoryJurisdiction.FDA_US,
            RegulatoryJurisdiction.CE_EU,
            RegulatoryJurisdiction.ISO_INTERNATIONAL
        ]
    )
    
    # Add some demo compliance evidence
    evidence_items = [
        {
            'requirement_id': 'FDA_510K_001',
            'evidence_type': 'document',
            'title': 'Predicate Device Analysis',
            'description': 'Comparison with FDA-cleared predicate device for pneumonia detection'
        },
        {
            'requirement_id': 'FDA_QSR_001',
            'evidence_type': 'document',
            'title': 'Quality System Manual',
            'description': 'ISO 13485 compliant quality management system documentation'
        },
        {
            'requirement_id': 'FDA_SOFTWARE_001',
            'evidence_type': 'test_result',
            'title': 'Software Validation Testing',
            'description': 'Comprehensive software testing including unit, integration, and system tests'
        },
        {
            'requirement_id': 'EU_MDR_001',
            'evidence_type': 'certification',
            'title': 'CE Marking Declaration',
            'description': 'Declaration of conformity with EU MDR requirements'
        },
        {
            'requirement_id': 'ISO_14971_001',
            'evidence_type': 'document',
            'title': 'Risk Management File',
            'description': 'Comprehensive risk analysis and risk control measures for AI system'
        },
        {
            'requirement_id': 'CLINICAL_001',
            'evidence_type': 'test_result',
            'title': 'Clinical Validation Study',
            'description': 'Multi-site clinical study demonstrating diagnostic performance'
        }
    ]
    
    for evidence_item in evidence_items:
        framework.add_compliance_evidence(
            profile_id=profile_id,
            requirement_id=evidence_item['requirement_id'],
            evidence_type=evidence_item['evidence_type'],
            title=evidence_item['title'],
            description=evidence_item['description'],
            verified_by="Regulatory Affairs Team"
        )
    
    return profile_id, framework


def main():
    """Demonstrate global medical compliance framework."""
    print("📊 Global Medical Compliance Framework")
    print("=" * 45)
    
    # Create demo profile
    print("🏥 Creating demo medical device compliance profile...")
    profile_id, framework = create_demo_medical_device_profile()
    
    print(f"✅ Created profile: {profile_id}")
    
    # Perform compliance assessment
    print("\n🔍 Performing comprehensive compliance assessment...")
    assessment_summary = framework.perform_compliance_assessment(profile_id)
    
    print(f"📋 Assessment Results:")
    print(f"   Overall compliance score: {assessment_summary['overall_compliance_score']:.3f}")
    print(f"   Total requirements: {assessment_summary['total_requirements']}")
    print(f"   Compliant requirements: {assessment_summary['compliant_requirements']}")
    print(f"   Partial compliance: {assessment_summary['partial_requirements']}")
    print(f"   Non-compliant requirements: {assessment_summary['non_compliant_requirements']}")
    
    # Show jurisdiction-specific results
    print(f"\n🌍 Jurisdiction-Specific Compliance:")
    for jurisdiction, score in assessment_summary['jurisdiction_scores'].items():
        status = assessment_summary['certification_status'][jurisdiction].value
        print(f"   {jurisdiction.value}: {score:.3f} ({status})")
    
    # Show critical gaps
    if assessment_summary['critical_gaps']:
        print(f"\n⚠️  Critical Gaps Identified:")
        for gap in assessment_summary['critical_gaps'][:5]:  # Show first 5
            print(f"   • {gap}")
    
    # Generate comprehensive report
    print(f"\n📄 Generating compliance report...")
    compliance_report = framework.generate_compliance_report(profile_id)
    
    print(f"   Report generated for: {compliance_report['report_metadata']['device_name']}")
    print(f"   Target jurisdictions: {len(compliance_report['report_metadata']['target_jurisdictions'])}")
    print(f"   Evidence items documented: {compliance_report['compliance_summary']['evidence_items']}")
    
    # Export regulatory documentation
    print(f"\n📤 Exporting regulatory documentation...")
    output_dir = Path("regulatory_exports")
    exported_files = framework.export_regulatory_documentation(profile_id, output_dir)
    
    print(f"   Exported files:")
    for file_type, file_path in exported_files.items():
        print(f"     {file_type}: {file_path}")
    
    # Show recommendations
    print(f"\n💡 Key Recommendations:")
    for recommendation in compliance_report['recommendations']:
        print(f"   • {recommendation}")
    
    print(f"\n📅 Next Actions:")
    for action in compliance_report['next_actions']:
        print(f"   • {action}")
    
    print(f"\n✅ Global medical compliance framework demonstration complete!")
    print(f"📁 Regulatory documentation exported to: {output_dir}")
    
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np  # Import numpy for calculations
    main()