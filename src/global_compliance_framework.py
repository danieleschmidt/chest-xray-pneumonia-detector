"""Global Compliance Framework for Medical AI Systems.

This module implements comprehensive compliance with international medical AI
regulations including FDA (US), CE-MDR (EU), Health Canada, PMDA (Japan),
NMPA (China), TGA (Australia), and other global regulatory frameworks.
"""

import logging
import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from datetime import datetime, timezone
import hashlib
import uuid


class RegulatoryRegion(Enum):
    """Global regulatory regions for medical AI."""
    US_FDA = "us_fda"                    # United States - FDA
    EU_CE_MDR = "eu_ce_mdr"             # European Union - CE-MDR
    CANADA_HC = "canada_hc"             # Health Canada
    JAPAN_PMDA = "japan_pmda"           # Japan - PMDA
    CHINA_NMPA = "china_nmpa"           # China - NMPA
    AUSTRALIA_TGA = "australia_tga"     # Australia - TGA
    UK_MHRA = "uk_mhra"                 # United Kingdom - MHRA
    BRAZIL_ANVISA = "brazil_anvisa"     # Brazil - ANVISA
    INDIA_CDSCO = "india_cdsco"         # India - CDSCO
    SINGAPORE_HSA = "singapore_hsa"     # Singapore - HSA


class ComplianceLevel(Enum):
    """Levels of regulatory compliance."""
    CLASS_I = "class_i"                 # Low risk
    CLASS_II = "class_ii"               # Moderate risk
    CLASS_III = "class_iii"             # High risk
    SOFTWARE_MEDICAL_DEVICE = "smd"     # Software as Medical Device
    AI_ML_ENABLED = "ai_ml_enabled"     # AI/ML Enabled Medical Device


class DataProtectionRegulation(Enum):
    """Data protection and privacy regulations."""
    GDPR = "gdpr"                       # EU General Data Protection Regulation
    HIPAA = "hipaa"                     # US Health Insurance Portability
    PIPEDA = "pipeda"                   # Canada Personal Information Protection
    PDPA_SINGAPORE = "pdpa_sg"          # Singapore Personal Data Protection
    LGPD = "lgpd"                       # Brazil Lei Geral de Proteção de Dados
    PDPA_THAILAND = "pdpa_th"           # Thailand Personal Data Protection
    APPI = "appi"                       # Japan Act on Protection of Personal Information


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""
    requirement_id: str
    regulation: str
    region: RegulatoryRegion
    compliance_level: ComplianceLevel
    title: str
    description: str
    mandatory: bool = True
    evidence_required: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    
    @property
    def is_overdue(self) -> bool:
        """Check if requirement is overdue."""
        if self.deadline is None:
            return False
        return datetime.now(timezone.utc) > self.deadline


@dataclass
class ComplianceEvidence:
    """Evidence for compliance requirement."""
    evidence_id: str
    requirement_id: str
    evidence_type: str  # document, test_result, certification, etc.
    title: str
    description: str
    file_path: Optional[str] = None
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expiry_date: Optional[datetime] = None
    verification_status: str = "pending"  # pending, verified, rejected
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAuditTrail:
    """Audit trail entry for compliance activities."""
    audit_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action: str = ""
    actor: str = ""
    requirement_id: Optional[str] = None
    evidence_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    session_id: str = ""


class GlobalComplianceManager:
    """Manages global regulatory compliance for medical AI systems."""
    
    def __init__(self, target_regions: List[RegulatoryRegion] = None):
        self.target_regions = target_regions or [
            RegulatoryRegion.US_FDA,
            RegulatoryRegion.EU_CE_MDR,
            RegulatoryRegion.CANADA_HC
        ]
        
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.evidence: Dict[str, ComplianceEvidence] = {}
        self.audit_trail: List[ComplianceAuditTrail] = []
        
        # Initialize requirements for target regions
        self._initialize_compliance_requirements()
        
    def _initialize_compliance_requirements(self) -> None:
        """Initialize compliance requirements for target regions."""
        
        for region in self.target_regions:
            if region == RegulatoryRegion.US_FDA:
                self._add_fda_requirements()
            elif region == RegulatoryRegion.EU_CE_MDR:
                self._add_eu_ce_mdr_requirements()
            elif region == RegulatoryRegion.CANADA_HC:
                self._add_health_canada_requirements()
            elif region == RegulatoryRegion.JAPAN_PMDA:
                self._add_japan_pmda_requirements()
            elif region == RegulatoryRegion.CHINA_NMPA:
                self._add_china_nmpa_requirements()
            elif region == RegulatoryRegion.AUSTRALIA_TGA:
                self._add_australia_tga_requirements()
                
    def _add_fda_requirements(self) -> None:
        """Add FDA compliance requirements."""
        
        requirements = [
            ComplianceRequirement(
                requirement_id="FDA-001",
                regulation="21 CFR Part 820",
                region=RegulatoryRegion.US_FDA,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Quality Management System",
                description="Implement and maintain a quality management system compliant with FDA QSR",
                evidence_required=[
                    "Quality manual",
                    "Process documentation",
                    "Training records",
                    "Audit reports"
                ],
                validation_criteria=[
                    "ISO 13485 certification",
                    "Internal audit completion",
                    "Management review records"
                ]
            ),
            ComplianceRequirement(
                requirement_id="FDA-002",
                regulation="21 CFR Part 11",
                region=RegulatoryRegion.US_FDA,
                compliance_level=ComplianceLevel.SOFTWARE_MEDICAL_DEVICE,
                title="Electronic Records and Electronic Signatures",
                description="Ensure electronic records and signatures are compliant with FDA requirements",
                evidence_required=[
                    "System validation documentation",
                    "User access controls",
                    "Audit trail configuration",
                    "Digital signature implementation"
                ],
                validation_criteria=[
                    "Complete audit trail",
                    "Secure user authentication",
                    "Data integrity validation"
                ]
            ),
            ComplianceRequirement(
                requirement_id="FDA-003",
                regulation="FDA AI/ML Guidance",
                region=RegulatoryRegion.US_FDA,
                compliance_level=ComplianceLevel.AI_ML_ENABLED,
                title="AI/ML Algorithm Change Control",
                description="Implement change control for AI/ML algorithms including locked and adaptive algorithms",
                evidence_required=[
                    "Algorithm change control plan",
                    "Pre-specified change protocol",
                    "Real-world performance monitoring",
                    "Risk management documentation"
                ],
                validation_criteria=[
                    "Predetermined change control plan",
                    "Clinical performance monitoring",
                    "Risk mitigation strategies"
                ]
            ),
            ComplianceRequirement(
                requirement_id="FDA-004",
                regulation="510(k) Premarket Notification",
                region=RegulatoryRegion.US_FDA,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Premarket Notification",
                description="Submit 510(k) premarket notification demonstrating substantial equivalence",
                evidence_required=[
                    "510(k) submission",
                    "Predicate device comparison",
                    "Clinical data",
                    "Software documentation"
                ],
                validation_criteria=[
                    "FDA clearance letter",
                    "Substantial equivalence demonstration"
                ]
            ),
            ComplianceRequirement(
                requirement_id="FDA-005",
                regulation="Clinical Evaluation",
                region=RegulatoryRegion.US_FDA,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Clinical Validation",
                description="Provide clinical evidence of safety and effectiveness",
                evidence_required=[
                    "Clinical study protocol",
                    "Clinical study report",
                    "Statistical analysis plan",
                    "Clinical data"
                ],
                validation_criteria=[
                    "Statistically significant results",
                    "Clinical endpoint achievement",
                    "Safety profile demonstration"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.requirement_id] = req
            
    def _add_eu_ce_mdr_requirements(self) -> None:
        """Add EU CE-MDR compliance requirements."""
        
        requirements = [
            ComplianceRequirement(
                requirement_id="EU-001",
                regulation="MDR Article 10",
                region=RegulatoryRegion.EU_CE_MDR,
                compliance_level=ComplianceLevel.CLASS_II,
                title="CE Marking and Declaration of Conformity",
                description="Affix CE marking and prepare EU declaration of conformity",
                evidence_required=[
                    "EU declaration of conformity",
                    "Technical documentation",
                    "Notified body certificate",
                    "CE marking placement"
                ],
                validation_criteria=[
                    "Notified body approval",
                    "Conformity assessment completion",
                    "Technical documentation review"
                ]
            ),
            ComplianceRequirement(
                requirement_id="EU-002",
                regulation="MDR Annex II",
                region=RegulatoryRegion.EU_CE_MDR,
                compliance_level=ComplianceLevel.SOFTWARE_MEDICAL_DEVICE,
                title="Software Lifecycle Documentation",
                description="Document software development lifecycle according to MDR requirements",
                evidence_required=[
                    "Software lifecycle process",
                    "Risk management file",
                    "Software safety classification",
                    "Verification and validation"
                ],
                validation_criteria=[
                    "IEC 62304 compliance",
                    "Risk analysis completion",
                    "V&V documentation"
                ]
            ),
            ComplianceRequirement(
                requirement_id="EU-003",
                regulation="MDR Article 61",
                region=RegulatoryRegion.EU_CE_MDR,
                compliance_level=ComplianceLevel.AI_ML_ENABLED,
                title="Post-Market Surveillance",
                description="Establish post-market surveillance system for AI/ML devices",
                evidence_required=[
                    "Post-market surveillance plan",
                    "Periodic safety update reports",
                    "Incident reporting procedures",
                    "Performance monitoring data"
                ],
                validation_criteria=[
                    "PMS system implementation",
                    "Regular PSUR submissions",
                    "Incident investigation records"
                ]
            ),
            ComplianceRequirement(
                requirement_id="EU-004",
                regulation="GDPR Compliance",
                region=RegulatoryRegion.EU_CE_MDR,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Data Protection Compliance",
                description="Ensure GDPR compliance for personal health data processing",
                evidence_required=[
                    "Data protection impact assessment",
                    "Privacy policy",
                    "Consent management system",
                    "Data processing agreements"
                ],
                validation_criteria=[
                    "DPIA completion",
                    "Legal basis establishment",
                    "Data subject rights implementation"
                ]
            ),
            ComplianceRequirement(
                requirement_id="EU-005",
                regulation="MDR Article 120",
                region=RegulatoryRegion.EU_CE_MDR,
                compliance_level=ComplianceLevel.AI_ML_ENABLED,
                title="AI Algorithm Transparency",
                description="Provide transparency and explainability for AI algorithms",
                evidence_required=[
                    "Algorithm transparency document",
                    "Explainability implementation",
                    "Bias assessment report",
                    "Performance characteristics"
                ],
                validation_criteria=[
                    "Algorithm explainability",
                    "Bias mitigation measures",
                    "Performance validation"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.requirement_id] = req
            
    def _add_health_canada_requirements(self) -> None:
        """Add Health Canada compliance requirements."""
        
        requirements = [
            ComplianceRequirement(
                requirement_id="HC-001",
                regulation="Medical Device Regulations",
                region=RegulatoryRegion.CANADA_HC,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Medical Device License",
                description="Obtain Medical Device License from Health Canada",
                evidence_required=[
                    "Medical device license application",
                    "Quality system certificate",
                    "Clinical evidence",
                    "Risk management documentation"
                ],
                validation_criteria=[
                    "Health Canada license approval",
                    "Quality system certification",
                    "Clinical data acceptance"
                ]
            ),
            ComplianceRequirement(
                requirement_id="HC-002",
                regulation="AI/ML Guidance",
                region=RegulatoryRegion.CANADA_HC,
                compliance_level=ComplianceLevel.AI_ML_ENABLED,
                title="AI/ML Medical Device Requirements",
                description="Meet Health Canada requirements for AI/ML medical devices",
                evidence_required=[
                    "AI/ML development documentation",
                    "Algorithm validation report",
                    "Real-world evidence plan",
                    "Change control procedures"
                ],
                validation_criteria=[
                    "Algorithm validation completion",
                    "Change control implementation",
                    "Performance monitoring plan"
                ]
            ),
            ComplianceRequirement(
                requirement_id="HC-003",
                regulation="PIPEDA Compliance",
                region=RegulatoryRegion.CANADA_HC,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Privacy Protection",
                description="Ensure compliance with Personal Information Protection and Electronic Documents Act",
                evidence_required=[
                    "Privacy impact assessment",
                    "Privacy policy",
                    "Consent procedures",
                    "Data security measures"
                ],
                validation_criteria=[
                    "Privacy protection implementation",
                    "Consent mechanism validation",
                    "Security measure verification"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.requirement_id] = req
            
    def _add_japan_pmda_requirements(self) -> None:
        """Add Japan PMDA compliance requirements."""
        
        requirements = [
            ComplianceRequirement(
                requirement_id="JP-001",
                regulation="Pharmaceutical and Medical Device Act",
                region=RegulatoryRegion.JAPAN_PMDA,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Marketing Authorization",
                description="Obtain marketing authorization from PMDA",
                evidence_required=[
                    "Marketing authorization application",
                    "Clinical trial data",
                    "Quality management documentation",
                    "Risk benefit analysis"
                ],
                validation_criteria=[
                    "PMDA approval",
                    "Clinical trial completion",
                    "QMS verification"
                ]
            ),
            ComplianceRequirement(
                requirement_id="JP-002",
                regulation="AI/ML Medical Device Guidelines",
                region=RegulatoryRegion.JAPAN_PMDA,
                compliance_level=ComplianceLevel.AI_ML_ENABLED,
                title="AI/ML Device Validation",
                description="Validate AI/ML medical devices according to PMDA guidelines",
                evidence_required=[
                    "Algorithm development documentation",
                    "Training data validation",
                    "Performance evaluation",
                    "Clinical validation study"
                ],
                validation_criteria=[
                    "Algorithm performance validation",
                    "Clinical effectiveness demonstration",
                    "Safety profile establishment"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.requirement_id] = req
            
    def _add_china_nmpa_requirements(self) -> None:
        """Add China NMPA compliance requirements."""
        
        requirements = [
            ComplianceRequirement(
                requirement_id="CN-001",
                regulation="Medical Device Regulations",
                region=RegulatoryRegion.CHINA_NMPA,
                compliance_level=ComplianceLevel.CLASS_II,
                title="Medical Device Registration",
                description="Register medical device with NMPA",
                evidence_required=[
                    "Registration application",
                    "Clinical trial approval",
                    "Manufacturing license",
                    "Quality management certificate"
                ],
                validation_criteria=[
                    "NMPA registration certificate",
                    "Clinical trial completion",
                    "GMP compliance"
                ]
            ),
            ComplianceRequirement(
                requirement_id="CN-002",
                regulation="AI Medical Device Guidelines",
                region=RegulatoryRegion.CHINA_NMPA,
                compliance_level=ComplianceLevel.AI_ML_ENABLED,
                title="AI Medical Device Approval",
                description="Obtain approval for AI-based medical devices",
                evidence_required=[
                    "AI algorithm documentation",
                    "Training dataset validation",
                    "Clinical performance study",
                    "Risk management plan"
                ],
                validation_criteria=[
                    "Algorithm validation completion",
                    "Clinical study approval",
                    "Risk assessment acceptance"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.requirement_id] = req
            
    def _add_australia_tga_requirements(self) -> None:
        """Add Australia TGA compliance requirements."""
        
        requirements = [
            ComplianceRequirement(
                requirement_id="AU-001",
                regulation="Therapeutic Goods Act",
                region=RegulatoryRegion.AUSTRALIA_TGA,
                compliance_level=ComplianceLevel.CLASS_II,
                title="TGA Registration",
                description="Register medical device with TGA",
                evidence_required=[
                    "TGA registration application",
                    "Conformity assessment",
                    "Clinical evidence",
                    "Quality management system"
                ],
                validation_criteria=[
                    "TGA registration approval",
                    "Conformity assessment completion",
                    "Clinical evidence acceptance"
                ]
            ),
            ComplianceRequirement(
                requirement_id="AU-002",
                regulation="AI/ML Device Guidelines",
                region=RegulatoryRegion.AUSTRALIA_TGA,
                compliance_level=ComplianceLevel.AI_ML_ENABLED,
                title="AI/ML Device Requirements",
                description="Meet TGA requirements for AI/ML medical devices",
                evidence_required=[
                    "AI/ML development documentation",
                    "Validation and verification",
                    "Post-market monitoring plan",
                    "Risk management file"
                ],
                validation_criteria=[
                    "V&V completion",
                    "Risk management approval",
                    "Post-market plan acceptance"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.requirement_id] = req
            
    def add_evidence(self, evidence: ComplianceEvidence) -> None:
        """Add evidence for a compliance requirement."""
        self.evidence[evidence.evidence_id] = evidence
        
        # Create audit trail entry
        audit = ComplianceAuditTrail(
            audit_id=str(uuid.uuid4()),
            action="evidence_added",
            actor="system",
            requirement_id=evidence.requirement_id,
            evidence_id=evidence.evidence_id,
            details={
                "evidence_type": evidence.evidence_type,
                "title": evidence.title
            }
        )
        self.audit_trail.append(audit)
        
    def verify_evidence(self, evidence_id: str, verifier: str, status: str) -> None:
        """Verify evidence for compliance."""
        if evidence_id in self.evidence:
            self.evidence[evidence_id].verification_status = status
            
            # Create audit trail entry
            audit = ComplianceAuditTrail(
                audit_id=str(uuid.uuid4()),
                action="evidence_verified",
                actor=verifier,
                evidence_id=evidence_id,
                details={"verification_status": status}
            )
            self.audit_trail.append(audit)
            
    def get_compliance_status(self, region: Optional[RegulatoryRegion] = None) -> Dict[str, Any]:
        """Get overall compliance status."""
        
        # Filter requirements by region if specified
        requirements = self.requirements.values()
        if region:
            requirements = [req for req in requirements if req.region == region]
            
        total_requirements = len(list(requirements))
        if total_requirements == 0:
            return {"status": "no_requirements", "percentage": 0}
            
        # Calculate compliance metrics
        requirements_with_evidence = 0
        verified_requirements = 0
        overdue_requirements = 0
        
        requirement_details = {}
        
        for req in requirements:
            # Find evidence for this requirement
            req_evidence = [ev for ev in self.evidence.values() if ev.requirement_id == req.requirement_id]
            
            has_evidence = len(req_evidence) > 0
            is_verified = any(ev.verification_status == "verified" for ev in req_evidence)
            is_overdue = req.is_overdue
            
            if has_evidence:
                requirements_with_evidence += 1
            if is_verified:
                verified_requirements += 1
            if is_overdue:
                overdue_requirements += 1
                
            requirement_details[req.requirement_id] = {
                "title": req.title,
                "regulation": req.regulation,
                "region": req.region.value,
                "compliance_level": req.compliance_level.value,
                "has_evidence": has_evidence,
                "is_verified": is_verified,
                "is_overdue": is_overdue,
                "evidence_count": len(req_evidence),
                "mandatory": req.mandatory
            }
            
        # Calculate percentages
        evidence_percentage = (requirements_with_evidence / total_requirements) * 100
        verification_percentage = (verified_requirements / total_requirements) * 100
        
        # Determine overall status
        if verification_percentage >= 95:
            overall_status = "compliant"
        elif verification_percentage >= 80:
            overall_status = "mostly_compliant"
        elif evidence_percentage >= 60:
            overall_status = "in_progress"
        else:
            overall_status = "non_compliant"
            
        return {
            "status": overall_status,
            "total_requirements": total_requirements,
            "requirements_with_evidence": requirements_with_evidence,
            "verified_requirements": verified_requirements,
            "overdue_requirements": overdue_requirements,
            "evidence_percentage": evidence_percentage,
            "verification_percentage": verification_percentage,
            "requirement_details": requirement_details,
            "regions_covered": list(set(req.region.value for req in requirements))
        }
        
    def generate_compliance_report(self, format: str = "json") -> str:
        """Generate comprehensive compliance report."""
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "target_regions": [region.value for region in self.target_regions],
                "report_version": "1.0"
            },
            "overall_status": self.get_compliance_status(),
            "regional_status": {},
            "critical_gaps": [],
            "upcoming_deadlines": [],
            "evidence_summary": {}
        }
        
        # Regional compliance status
        for region in self.target_regions:
            report_data["regional_status"][region.value] = self.get_compliance_status(region)
            
        # Identify critical gaps
        for req in self.requirements.values():
            req_evidence = [ev for ev in self.evidence.values() if ev.requirement_id == req.requirement_id]
            
            if req.mandatory and not req_evidence:
                report_data["critical_gaps"].append({
                    "requirement_id": req.requirement_id,
                    "title": req.title,
                    "region": req.region.value,
                    "regulation": req.regulation,
                    "reason": "no_evidence_provided"
                })
            elif req.mandatory and not any(ev.verification_status == "verified" for ev in req_evidence):
                report_data["critical_gaps"].append({
                    "requirement_id": req.requirement_id,
                    "title": req.title,
                    "region": req.region.value,
                    "regulation": req.regulation,
                    "reason": "evidence_not_verified"
                })
                
        # Upcoming deadlines
        upcoming_deadlines = []
        current_time = datetime.now(timezone.utc)
        
        for req in self.requirements.values():
            if req.deadline and req.deadline > current_time:
                days_until_deadline = (req.deadline - current_time).days
                if days_until_deadline <= 90:  # Within 90 days
                    upcoming_deadlines.append({
                        "requirement_id": req.requirement_id,
                        "title": req.title,
                        "deadline": req.deadline.isoformat(),
                        "days_remaining": days_until_deadline,
                        "region": req.region.value
                    })
                    
        report_data["upcoming_deadlines"] = sorted(upcoming_deadlines, key=lambda x: x["days_remaining"])
        
        # Evidence summary
        evidence_by_type = {}
        for evidence in self.evidence.values():
            if evidence.evidence_type not in evidence_by_type:
                evidence_by_type[evidence.evidence_type] = {
                    "count": 0,
                    "verified": 0,
                    "pending": 0,
                    "rejected": 0
                }
                
            evidence_by_type[evidence.evidence_type]["count"] += 1
            if evidence.verification_status == "verified":
                evidence_by_type[evidence.evidence_type]["verified"] += 1
            elif evidence.verification_status == "pending":
                evidence_by_type[evidence.evidence_type]["pending"] += 1
            elif evidence.verification_status == "rejected":
                evidence_by_type[evidence.evidence_type]["rejected"] += 1
                
        report_data["evidence_summary"] = evidence_by_type
        
        if format == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif format == "markdown":
            return self._generate_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown compliance report."""
        
        md = f"""# Global Compliance Report

**Generated:** {report_data['report_metadata']['generated_at']}  
**Target Regions:** {', '.join(report_data['report_metadata']['target_regions'])}

## Overall Compliance Status

- **Status:** {report_data['overall_status']['status'].upper()}
- **Verification Progress:** {report_data['overall_status']['verification_percentage']:.1f}%
- **Evidence Collection:** {report_data['overall_status']['evidence_percentage']:.1f}%
- **Total Requirements:** {report_data['overall_status']['total_requirements']}
- **Overdue Requirements:** {report_data['overall_status']['overdue_requirements']}

## Regional Compliance Status

"""
        
        for region, status in report_data["regional_status"].items():
            md += f"""### {region.upper()}

- **Status:** {status['status'].upper()}
- **Verification:** {status['verification_percentage']:.1f}%
- **Requirements:** {status['verified_requirements']}/{status['total_requirements']}
- **Overdue:** {status['overdue_requirements']}

"""
        
        # Critical gaps
        if report_data["critical_gaps"]:
            md += f"""## 🚨 Critical Gaps

"""
            for gap in report_data["critical_gaps"]:
                md += f"""- **{gap['requirement_id']}**: {gap['title']} ({gap['region']})
  - Reason: {gap['reason']}
  - Regulation: {gap['regulation']}

"""
        
        # Upcoming deadlines
        if report_data["upcoming_deadlines"]:
            md += f"""## ⏰ Upcoming Deadlines

"""
            for deadline in report_data["upcoming_deadlines"]:
                md += f"""- **{deadline['requirement_id']}**: {deadline['title']}
  - Days Remaining: {deadline['days_remaining']}
  - Region: {deadline['region']}
  - Deadline: {deadline['deadline']}

"""
        
        # Evidence summary
        md += f"""## 📋 Evidence Summary

"""
        for evidence_type, summary in report_data["evidence_summary"].items():
            md += f"""### {evidence_type.title()}

- Total: {summary['count']}
- Verified: {summary['verified']}
- Pending: {summary['pending']}
- Rejected: {summary['rejected']}

"""
        
        return md
        
    def export_audit_trail(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export audit trail for compliance audits."""
        
        filtered_trail = self.audit_trail
        
        if start_date:
            filtered_trail = [entry for entry in filtered_trail if entry.timestamp >= start_date]
            
        if end_date:
            filtered_trail = [entry for entry in filtered_trail if entry.timestamp <= end_date]
            
        return [
            {
                "audit_id": entry.audit_id,
                "timestamp": entry.timestamp.isoformat(),
                "action": entry.action,
                "actor": entry.actor,
                "requirement_id": entry.requirement_id,
                "evidence_id": entry.evidence_id,
                "details": entry.details,
                "ip_address": entry.ip_address,
                "session_id": entry.session_id
            }
            for entry in filtered_trail
        ]


class DataProtectionComplianceManager:
    """Manages data protection and privacy compliance across regions."""
    
    def __init__(self, target_regulations: List[DataProtectionRegulation] = None):
        self.target_regulations = target_regulations or [
            DataProtectionRegulation.GDPR,
            DataProtectionRegulation.HIPAA,
            DataProtectionRegulation.PIPEDA
        ]
        
        self.compliance_measures: Dict[str, Any] = {}
        self._initialize_data_protection_measures()
        
    def _initialize_data_protection_measures(self) -> None:
        """Initialize data protection compliance measures."""
        
        for regulation in self.target_regulations:
            if regulation == DataProtectionRegulation.GDPR:
                self._setup_gdpr_compliance()
            elif regulation == DataProtectionRegulation.HIPAA:
                self._setup_hipaa_compliance()
            elif regulation == DataProtectionRegulation.PIPEDA:
                self._setup_pipeda_compliance()
                
    def _setup_gdpr_compliance(self) -> None:
        """Setup GDPR compliance measures."""
        
        self.compliance_measures["gdpr"] = {
            "legal_basis": "legitimate_interest",  # or consent, contract, etc.
            "data_minimization": True,
            "purpose_limitation": True,
            "consent_management": True,
            "right_to_be_forgotten": True,
            "data_portability": True,
            "privacy_by_design": True,
            "dpia_required": True,
            "dpo_appointed": True,
            "breach_notification": "72_hours",
            "data_retention_policy": True,
            "international_transfers": "adequacy_decision"
        }
        
    def _setup_hipaa_compliance(self) -> None:
        """Setup HIPAA compliance measures."""
        
        self.compliance_measures["hipaa"] = {
            "phi_encryption": True,
            "access_controls": True,
            "audit_logging": True,
            "business_associate_agreements": True,
            "minimum_necessary": True,
            "breach_notification": "60_days",
            "risk_assessment": True,
            "workforce_training": True,
            "incident_response": True,
            "physical_safeguards": True
        }
        
    def _setup_pipeda_compliance(self) -> None:
        """Setup PIPEDA compliance measures."""
        
        self.compliance_measures["pipeda"] = {
            "consent_required": True,
            "purpose_identification": True,
            "data_minimization": True,
            "retention_limits": True,
            "accuracy_maintenance": True,
            "security_safeguards": True,
            "openness_policy": True,
            "individual_access": True,
            "challenge_compliance": True,
            "accountability": True
        }
        
    def validate_data_processing(self, processing_activity: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data processing activity against applicable regulations."""
        
        validation_results = {}
        
        for regulation in self.target_regulations:
            regulation_key = regulation.value
            measures = self.compliance_measures.get(regulation_key, {})
            
            validation_result = {
                "regulation": regulation_key,
                "compliant": True,
                "issues": [],
                "recommendations": []
            }
            
            # Validate against specific regulation requirements
            if regulation == DataProtectionRegulation.GDPR:
                validation_result = self._validate_gdpr_processing(processing_activity, measures)
            elif regulation == DataProtectionRegulation.HIPAA:
                validation_result = self._validate_hipaa_processing(processing_activity, measures)
            elif regulation == DataProtectionRegulation.PIPEDA:
                validation_result = self._validate_pipeda_processing(processing_activity, measures)
                
            validation_results[regulation_key] = validation_result
            
        return validation_results
        
    def _validate_gdpr_processing(self, activity: Dict[str, Any], measures: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data processing against GDPR requirements."""
        
        result = {"regulation": "gdpr", "compliant": True, "issues": [], "recommendations": []}
        
        # Check legal basis
        if not activity.get("legal_basis"):
            result["issues"].append("No legal basis specified for data processing")
            result["compliant"] = False
            
        # Check consent management
        if activity.get("requires_consent") and not measures.get("consent_management"):
            result["issues"].append("Consent management system not implemented")
            result["compliant"] = False
            
        # Check data minimization
        if not measures.get("data_minimization"):
            result["issues"].append("Data minimization principle not implemented")
            result["compliant"] = False
            
        # Check DPIA requirement
        if activity.get("high_risk") and not measures.get("dpia_required"):
            result["issues"].append("Data Protection Impact Assessment required for high-risk processing")
            result["compliant"] = False
            
        return result
        
    def _validate_hipaa_processing(self, activity: Dict[str, Any], measures: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data processing against HIPAA requirements."""
        
        result = {"regulation": "hipaa", "compliant": True, "issues": [], "recommendations": []}
        
        # Check PHI encryption
        if activity.get("contains_phi") and not measures.get("phi_encryption"):
            result["issues"].append("PHI must be encrypted at rest and in transit")
            result["compliant"] = False
            
        # Check access controls
        if not measures.get("access_controls"):
            result["issues"].append("Access controls not properly implemented")
            result["compliant"] = False
            
        # Check audit logging
        if not measures.get("audit_logging"):
            result["issues"].append("Comprehensive audit logging required")
            result["compliant"] = False
            
        return result
        
    def _validate_pipeda_processing(self, activity: Dict[str, Any], measures: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data processing against PIPEDA requirements."""
        
        result = {"regulation": "pipeda", "compliant": True, "issues": [], "recommendations": []}
        
        # Check consent requirement
        if not measures.get("consent_required"):
            result["issues"].append("Consent required for personal information collection")
            result["compliant"] = False
            
        # Check purpose identification
        if not activity.get("purpose_identified"):
            result["issues"].append("Purpose for collection must be clearly identified")
            result["compliant"] = False
            
        return result


class InternationalizationManager:
    """Manages internationalization and localization for global deployment."""
    
    def __init__(self):
        self.supported_locales = [
            "en-US",  # English (United States)
            "en-GB",  # English (United Kingdom)
            "fr-FR",  # French (France)
            "fr-CA",  # French (Canada)
            "de-DE",  # German (Germany)
            "es-ES",  # Spanish (Spain)
            "es-MX",  # Spanish (Mexico)
            "pt-BR",  # Portuguese (Brazil)
            "it-IT",  # Italian (Italy)
            "ja-JP",  # Japanese (Japan)
            "ko-KR",  # Korean (South Korea)
            "zh-CN",  # Chinese (Simplified)
            "zh-TW",  # Chinese (Traditional)
            "ar-SA",  # Arabic (Saudi Arabia)
            "hi-IN",  # Hindi (India)
            "th-TH",  # Thai (Thailand)
            "vi-VN",  # Vietnamese (Vietnam)
            "ru-RU",  # Russian (Russia)
            "pl-PL",  # Polish (Poland)
            "nl-NL",  # Dutch (Netherlands)
            "sv-SE",  # Swedish (Sweden)
            "da-DK",  # Danish (Denmark)
            "no-NO",  # Norwegian (Norway)
            "fi-FI",  # Finnish (Finland)
            "he-IL",  # Hebrew (Israel)
            "tr-TR",  # Turkish (Turkey)
        ]
        
        self.medical_terminology_standards = {
            "ICD-11": "International Classification of Diseases 11th Revision",
            "SNOMED-CT": "Systematized Nomenclature of Medicine Clinical Terms",
            "LOINC": "Logical Observation Identifiers Names and Codes",
            "CPT": "Current Procedural Terminology",
            "MESH": "Medical Subject Headings",
            "ICD-10": "International Classification of Diseases 10th Revision"
        }
        
    def get_locale_specific_requirements(self, locale: str) -> Dict[str, Any]:
        """Get locale-specific regulatory and cultural requirements."""
        
        requirements = {
            "locale": locale,
            "date_format": self._get_date_format(locale),
            "number_format": self._get_number_format(locale),
            "currency": self._get_currency(locale),
            "measurement_units": self._get_measurement_units(locale),
            "medical_terminology": self._get_medical_terminology(locale),
            "regulatory_authority": self._get_regulatory_authority(locale),
            "data_protection_law": self._get_data_protection_law(locale),
            "language_direction": self._get_language_direction(locale),
            "cultural_considerations": self._get_cultural_considerations(locale)
        }
        
        return requirements
        
    def _get_date_format(self, locale: str) -> str:
        """Get date format for locale."""
        formats = {
            "en-US": "MM/DD/YYYY",
            "en-GB": "DD/MM/YYYY",
            "fr-FR": "DD/MM/YYYY",
            "de-DE": "DD.MM.YYYY",
            "ja-JP": "YYYY/MM/DD",
            "zh-CN": "YYYY-MM-DD",
            "ar-SA": "DD/MM/YYYY"
        }
        return formats.get(locale, "YYYY-MM-DD")
        
    def _get_number_format(self, locale: str) -> Dict[str, str]:
        """Get number formatting for locale."""
        formats = {
            "en-US": {"decimal": ".", "thousand": ","},
            "fr-FR": {"decimal": ",", "thousand": " "},
            "de-DE": {"decimal": ",", "thousand": "."},
            "ja-JP": {"decimal": ".", "thousand": ","},
            "ar-SA": {"decimal": "٫", "thousand": "٬"}
        }
        return formats.get(locale, {"decimal": ".", "thousand": ","})
        
    def _get_currency(self, locale: str) -> str:
        """Get currency for locale."""
        currencies = {
            "en-US": "USD",
            "en-GB": "GBP",
            "fr-FR": "EUR",
            "de-DE": "EUR",
            "ja-JP": "JPY",
            "zh-CN": "CNY",
            "pt-BR": "BRL"
        }
        return currencies.get(locale, "USD")
        
    def _get_measurement_units(self, locale: str) -> Dict[str, str]:
        """Get measurement units for locale."""
        
        metric_system = {
            "length": "meter",
            "weight": "kilogram",
            "temperature": "celsius",
            "volume": "liter"
        }
        
        imperial_system = {
            "length": "foot",
            "weight": "pound",
            "temperature": "fahrenheit",
            "volume": "gallon"
        }
        
        if locale in ["en-US"]:
            return imperial_system
        else:
            return metric_system
            
    def _get_medical_terminology(self, locale: str) -> List[str]:
        """Get preferred medical terminology standards for locale."""
        
        terminologies = {
            "en-US": ["ICD-10", "CPT", "LOINC", "SNOMED-CT"],
            "en-GB": ["ICD-11", "SNOMED-CT", "LOINC"],
            "fr-FR": ["ICD-11", "SNOMED-CT"],
            "de-DE": ["ICD-11", "SNOMED-CT"],
            "ja-JP": ["ICD-11", "SNOMED-CT"],
            "zh-CN": ["ICD-11"],
            "pt-BR": ["ICD-11", "SNOMED-CT"]
        }
        
        return terminologies.get(locale, ["ICD-11", "SNOMED-CT"])
        
    def _get_regulatory_authority(self, locale: str) -> str:
        """Get primary regulatory authority for locale."""
        
        authorities = {
            "en-US": "FDA",
            "en-GB": "MHRA",
            "fr-FR": "ANSM",
            "de-DE": "BfArM",
            "ja-JP": "PMDA",
            "zh-CN": "NMPA",
            "pt-BR": "ANVISA",
            "en-CA": "Health Canada",
            "en-AU": "TGA"
        }
        
        return authorities.get(locale, "Unknown")
        
    def _get_data_protection_law(self, locale: str) -> str:
        """Get applicable data protection law for locale."""
        
        laws = {
            "en-US": "HIPAA",
            "en-GB": "GDPR",
            "fr-FR": "GDPR",
            "de-DE": "GDPR",
            "ja-JP": "APPI",
            "zh-CN": "PIPL",
            "pt-BR": "LGPD",
            "en-CA": "PIPEDA",
            "en-AU": "Privacy Act",
            "en-SG": "PDPA"
        }
        
        return laws.get(locale, "GDPR")
        
    def _get_language_direction(self, locale: str) -> str:
        """Get text direction for locale."""
        
        rtl_locales = ["ar-SA", "he-IL", "fa-IR", "ur-PK"]
        
        return "rtl" if locale in rtl_locales else "ltr"
        
    def _get_cultural_considerations(self, locale: str) -> List[str]:
        """Get cultural considerations for locale."""
        
        considerations = {
            "ar-SA": [
                "Gender-specific medical care preferences",
                "Religious considerations for medical imagery",
                "Family involvement in medical decisions"
            ],
            "ja-JP": [
                "Respect for hierarchy in medical settings",
                "Indirect communication style",
                "Privacy concerns with medical data"
            ],
            "zh-CN": [
                "Traditional medicine integration",
                "Family-centered decision making",
                "Data sovereignty requirements"
            ],
            "hi-IN": [
                "Multi-language support within region",
                "Traditional and modern medicine coexistence",
                "Economic sensitivity in healthcare"
            ]
        }
        
        return considerations.get(locale, [])


if __name__ == "__main__":
    # Demonstration of global compliance framework
    
    print("🌍 Global Compliance Framework for Medical AI Systems\\n")
    
    # Initialize compliance manager for multiple regions
    target_regions = [
        RegulatoryRegion.US_FDA,
        RegulatoryRegion.EU_CE_MDR,
        RegulatoryRegion.CANADA_HC,
        RegulatoryRegion.JAPAN_PMDA
    ]
    
    compliance_manager = GlobalComplianceManager(target_regions)
    
    # Add some sample evidence
    evidence1 = ComplianceEvidence(
        evidence_id="EV-001",
        requirement_id="FDA-001",
        evidence_type="certification",
        title="ISO 13485 Quality Management Certificate",
        description="Quality management system certification for medical devices",
        verification_status="verified"
    )
    
    evidence2 = ComplianceEvidence(
        evidence_id="EV-002",
        requirement_id="EU-001",
        evidence_type="document",
        title="CE Marking Technical Documentation",
        description="Complete technical documentation for CE marking",
        verification_status="pending"
    )
    
    compliance_manager.add_evidence(evidence1)
    compliance_manager.add_evidence(evidence2)
    
    # Get compliance status
    print("📊 Overall Compliance Status:")
    overall_status = compliance_manager.get_compliance_status()
    print(f"Status: {overall_status['status'].upper()}")
    print(f"Verification Progress: {overall_status['verification_percentage']:.1f}%")
    print(f"Evidence Collection: {overall_status['evidence_percentage']:.1f}%")
    print(f"Total Requirements: {overall_status['total_requirements']}")
    
    # Regional compliance status
    print(f"\\n🗺️ Regional Compliance Status:")
    for region in target_regions:
        regional_status = compliance_manager.get_compliance_status(region)
        print(f"{region.value.upper()}: {regional_status['verification_percentage']:.1f}% verified")
        
    # Generate compliance report
    print(f"\\n📋 Generating Compliance Report...")
    report = compliance_manager.generate_compliance_report("markdown")
    
    # Save report to file
    with open("global_compliance_report.md", "w") as f:
        f.write(report)
    print("✅ Compliance report saved to global_compliance_report.md")
    
    # Initialize data protection compliance
    print(f"\\n🔒 Data Protection Compliance:")
    data_protection = DataProtectionComplianceManager([
        DataProtectionRegulation.GDPR,
        DataProtectionRegulation.HIPAA,
        DataProtectionRegulation.PIPEDA
    ])
    
    # Validate a sample data processing activity
    processing_activity = {
        "contains_phi": True,
        "requires_consent": True,
        "purpose_identified": True,
        "legal_basis": "legitimate_interest",
        "high_risk": False
    }
    
    validation_results = data_protection.validate_data_processing(processing_activity)
    
    for regulation, result in validation_results.items():
        status = "✅ COMPLIANT" if result["compliant"] else "❌ NON-COMPLIANT"
        print(f"{regulation.upper()}: {status}")
        
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  🚨 Issue: {issue}")
                
    # Initialize internationalization
    print(f"\\n🌐 Internationalization Support:")
    i18n = InternationalizationManager()
    
    sample_locales = ["en-US", "fr-FR", "ja-JP", "ar-SA", "zh-CN"]
    
    for locale in sample_locales:
        requirements = i18n.get_locale_specific_requirements(locale)
        print(f"{locale}: {requirements['regulatory_authority']} | {requirements['data_protection_law']} | {requirements['language_direction']}")
        
    print(f"\\n🎯 Global Compliance Framework Demonstration Complete!")
    print(f"✅ {len(compliance_manager.requirements)} compliance requirements loaded")
    print(f"✅ {len(compliance_manager.evidence)} evidence items tracked")
    print(f"✅ {len(i18n.supported_locales)} locales supported")
    print(f"✅ {len(target_regions)} regulatory regions covered")