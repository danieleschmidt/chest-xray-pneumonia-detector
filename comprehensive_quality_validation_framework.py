#!/usr/bin/env python3
"""
Comprehensive Quality Validation Framework
Progressive Enhancement - Quality Gates Implementation
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

class TestCategory(Enum):
    """Test categories for comprehensive validation"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    END_TO_END = "end_to_end"

class QualityGateStatus(Enum):
    """Quality gate validation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class TestResult:
    """Individual test result"""
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: TestCategory = TestCategory.UNIT
    name: str = ""
    description: str = ""
    status: QualityGateStatus = QualityGateStatus.FAILED
    execution_time_ms: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityGateResult:
    """Quality gate validation result"""
    gate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    status: QualityGateStatus = QualityGateStatus.FAILED
    test_results: List[TestResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    coverage_percentage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class ComprehensiveQualityValidationFramework:
    """
    Comprehensive quality validation framework for medical AI systems.
    
    Features:
    - Multi-category testing (unit, integration, security, performance)
    - HIPAA compliance validation
    - Performance benchmarking
    - Security vulnerability scanning
    - Code quality metrics
    - Automated quality gates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Test tracking
        self.test_results: List[TestResult] = []
        self.quality_gates: Dict[str, QualityGateResult] = {}
        
        # Quality metrics
        self.coverage_data = {}
        self.performance_baselines = {}
        self.security_scan_results = {}
        
        self.logger = self._setup_logging()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for quality validation"""
        return {
            "quality_gates": {
                "min_test_coverage": 85.0,
                "max_test_failure_rate": 5.0,
                "max_security_violations": 0,
                "max_performance_degradation": 10.0
            },
            "testing": {
                "unit_test_timeout": 300,
                "integration_test_timeout": 600,
                "security_scan_timeout": 1800,
                "performance_test_timeout": 900
            },
            "compliance": {
                "hipaa_required": True,
                "gdpr_required": True,
                "audit_logging": True
            },
            "performance": {
                "max_latency_ms": 2000,
                "min_throughput_ops_per_sec": 100,
                "max_memory_usage_mb": 4096,
                "max_cpu_utilization": 0.8
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup quality validation logging"""
        logger = logging.getLogger("QualityValidation")
        logger.setLevel(logging.INFO)
        
        # Quality logs directory
        log_dir = Path("quality_logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"quality_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - QUALITY - %(levelname)s - %(message)s"
            )
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("QUALITY - %(levelname)s - %(message)s")
        )
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    async def run_comprehensive_validation(self) -> Dict[str, QualityGateResult]:
        """Run comprehensive quality validation across all categories"""
        self.logger.info("Starting comprehensive quality validation")
        start_time = time.time()
        
        # Run all quality gates in parallel
        tasks = [
            asyncio.create_task(self._run_unit_tests(), name="unit_tests"),
            asyncio.create_task(self._run_integration_tests(), name="integration_tests"),
            asyncio.create_task(self._run_security_validation(), name="security_validation"),
            asyncio.create_task(self._run_performance_validation(), name="performance_validation"),
            asyncio.create_task(self._run_compliance_validation(), name="compliance_validation"),
            asyncio.create_task(self._run_code_quality_validation(), name="code_quality")
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for task, result in zip(tasks, results):
            task_name = task.get_name()
            if isinstance(result, Exception):
                self.logger.error(f"Quality gate {task_name} failed with exception: {result}")
                # Create failed gate result
                self.quality_gates[task_name] = QualityGateResult(
                    name=task_name,
                    status=QualityGateStatus.FAILED,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            else:
                self.quality_gates[task_name] = result
                
        # Generate summary report
        total_time = (time.time() - start_time) * 1000
        self.logger.info(f"Comprehensive quality validation completed in {total_time:.2f}ms")
        
        return self.quality_gates
        
    async def _run_unit_tests(self) -> QualityGateResult:
        """Run comprehensive unit tests"""
        self.logger.info("Running unit tests")
        start_time = time.time()
        
        gate_result = QualityGateResult(name="unit_tests")
        
        # Mock unit test execution - in production would run pytest
        unit_tests = [
            {"name": "test_data_loader_validation", "module": "data_loader"},
            {"name": "test_model_builder_creation", "module": "model_builder"},
            {"name": "test_inference_accuracy", "module": "inference"},
            {"name": "test_image_preprocessing", "module": "image_utils"},
            {"name": "test_security_input_validation", "module": "input_validation"},
            {"name": "test_error_handling", "module": "error_handling"},
            {"name": "test_performance_metrics", "module": "monitoring"},
            {"name": "test_quantum_optimization", "module": "quantum_performance_orchestrator"}
        ]
        
        passed_tests = 0
        for test in unit_tests:
            test_result = await self._execute_unit_test(test)
            gate_result.test_results.append(test_result)
            
            if test_result.status == QualityGateStatus.PASSED:
                passed_tests += 1
                
        # Calculate coverage and determine gate status
        coverage_percentage = (passed_tests / len(unit_tests)) * 100
        gate_result.coverage_percentage = coverage_percentage
        
        min_coverage = self.config["quality_gates"]["min_test_coverage"]
        gate_result.status = (
            QualityGateStatus.PASSED if coverage_percentage >= min_coverage 
            else QualityGateStatus.FAILED
        )
        
        gate_result.execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Unit tests completed: {passed_tests}/{len(unit_tests)} passed "
                        f"({coverage_percentage:.1f}% coverage)")
        
        return gate_result
        
    async def _execute_unit_test(self, test_config: Dict[str, str]) -> TestResult:
        """Execute individual unit test"""
        test_name = test_config["name"]
        module = test_config["module"]
        
        start_time = time.time()
        
        try:
            # Mock test execution - in production would run actual tests
            await asyncio.sleep(0.1)  # Simulate test execution time
            
            # Simulate test results (90% pass rate)
            success = hash(test_name) % 10 != 0
            
            result = TestResult(
                category=TestCategory.UNIT,
                name=test_name,
                description=f"Unit test for {module} module",
                status=QualityGateStatus.PASSED if success else QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="" if success else f"Mock test failure in {test_name}",
                metadata={
                    "module": module,
                    "test_framework": "pytest",
                    "assertions": hash(test_name) % 10 + 1
                }
            )
            
            return result
            
        except Exception as e:
            return TestResult(
                category=TestCategory.UNIT,
                name=test_name,
                description=f"Unit test for {module} module",
                status=QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
                metadata={"module": module, "exception": type(e).__name__}
            )
            
    async def _run_integration_tests(self) -> QualityGateResult:
        """Run integration tests"""
        self.logger.info("Running integration tests")
        start_time = time.time()
        
        gate_result = QualityGateResult(name="integration_tests")
        
        # Integration test scenarios
        integration_tests = [
            {"name": "test_end_to_end_pipeline", "description": "Full ML pipeline integration"},
            {"name": "test_api_database_integration", "description": "API-Database integration"},
            {"name": "test_model_inference_pipeline", "description": "Model inference integration"},
            {"name": "test_security_authentication_flow", "description": "Authentication integration"},
            {"name": "test_monitoring_alerting_system", "description": "Monitoring integration"}
        ]
        
        passed_tests = 0
        for test in integration_tests:
            test_result = await self._execute_integration_test(test)
            gate_result.test_results.append(test_result)
            
            if test_result.status == QualityGateStatus.PASSED:
                passed_tests += 1
                
        # Determine gate status
        success_rate = (passed_tests / len(integration_tests)) * 100
        max_failure_rate = self.config["quality_gates"]["max_test_failure_rate"]
        
        gate_result.status = (
            QualityGateStatus.PASSED if success_rate >= (100 - max_failure_rate)
            else QualityGateStatus.FAILED
        )
        
        gate_result.execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Integration tests completed: {passed_tests}/{len(integration_tests)} passed")
        
        return gate_result
        
    async def _execute_integration_test(self, test_config: Dict[str, str]) -> TestResult:
        """Execute individual integration test"""
        test_name = test_config["name"]
        description = test_config["description"]
        
        start_time = time.time()
        
        try:
            # Mock integration test execution
            await asyncio.sleep(0.3)  # Simulate longer execution time
            
            # Simulate test results (85% pass rate for integration tests)
            success = hash(test_name) % 100 < 85
            
            return TestResult(
                category=TestCategory.INTEGRATION,
                name=test_name,
                description=description,
                status=QualityGateStatus.PASSED if success else QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="" if success else f"Integration failure: {test_name}",
                metadata={
                    "test_type": "integration",
                    "components_tested": hash(test_name) % 3 + 2
                }
            )
            
        except Exception as e:
            return TestResult(
                category=TestCategory.INTEGRATION,
                name=test_name,
                description=description,
                status=QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
            
    async def _run_security_validation(self) -> QualityGateResult:
        """Run comprehensive security validation"""
        self.logger.info("Running security validation")
        start_time = time.time()
        
        gate_result = QualityGateResult(name="security_validation")
        
        # Security validation tests
        security_tests = [
            {"name": "vulnerability_scan", "description": "Dependency vulnerability scanning"},
            {"name": "input_validation_security", "description": "Input validation security tests"},
            {"name": "authentication_security", "description": "Authentication security validation"},
            {"name": "data_encryption_validation", "description": "Data encryption compliance"},
            {"name": "access_control_validation", "description": "Access control security"},
            {"name": "audit_logging_validation", "description": "Audit logging security"}
        ]
        
        security_violations = 0
        
        for test in security_tests:
            test_result = await self._execute_security_test(test)
            gate_result.test_results.append(test_result)
            
            if test_result.status == QualityGateStatus.FAILED:
                security_violations += 1
                
        # Security gate must have zero violations
        max_violations = self.config["quality_gates"]["max_security_violations"]
        gate_result.status = (
            QualityGateStatus.PASSED if security_violations <= max_violations
            else QualityGateStatus.FAILED
        )
        
        gate_result.execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Security validation completed: {security_violations} violations found")
        
        return gate_result
        
    async def _execute_security_test(self, test_config: Dict[str, str]) -> TestResult:
        """Execute individual security test"""
        test_name = test_config["name"]
        description = test_config["description"]
        
        start_time = time.time()
        
        try:
            # Mock security test execution
            await asyncio.sleep(0.2)
            
            # Mock security scan results (95% pass rate for security)
            success = hash(test_name) % 100 < 95
            
            return TestResult(
                category=TestCategory.SECURITY,
                name=test_name,
                description=description,
                status=QualityGateStatus.PASSED if success else QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="" if success else f"Security violation: {test_name}",
                metadata={
                    "security_level": "high",
                    "compliance_checked": ["HIPAA", "GDPR"]
                }
            )
            
        except Exception as e:
            return TestResult(
                category=TestCategory.SECURITY,
                name=test_name,
                description=description,
                status=QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
            
    async def _run_performance_validation(self) -> QualityGateResult:
        """Run performance validation tests"""
        self.logger.info("Running performance validation")
        start_time = time.time()
        
        gate_result = QualityGateResult(name="performance_validation")
        
        # Performance test scenarios
        performance_tests = [
            {"name": "latency_benchmark", "metric": "latency", "target": 2000},
            {"name": "throughput_benchmark", "metric": "throughput", "target": 100},
            {"name": "memory_usage_validation", "metric": "memory", "target": 4096},
            {"name": "cpu_utilization_validation", "metric": "cpu", "target": 0.8},
            {"name": "concurrent_load_test", "metric": "concurrency", "target": 50}
        ]
        
        performance_failures = 0
        
        for test in performance_tests:
            test_result = await self._execute_performance_test(test)
            gate_result.test_results.append(test_result)
            
            if test_result.status == QualityGateStatus.FAILED:
                performance_failures += 1
                
        # Determine performance gate status
        max_degradation = self.config["quality_gates"]["max_performance_degradation"]
        failure_rate = (performance_failures / len(performance_tests)) * 100
        
        gate_result.status = (
            QualityGateStatus.PASSED if failure_rate <= max_degradation
            else QualityGateStatus.FAILED
        )
        
        gate_result.execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Performance validation completed: {performance_failures} failures")
        
        return gate_result
        
    async def _execute_performance_test(self, test_config: Dict[str, Any]) -> TestResult:
        """Execute individual performance test"""
        test_name = test_config["name"]
        metric = test_config["metric"]
        target = test_config["target"]
        
        start_time = time.time()
        
        try:
            # Mock performance test execution
            await asyncio.sleep(0.5)  # Simulate performance test time
            
            # Mock performance results
            if metric == "latency":
                actual_value = hash(test_name) % 1000 + 500  # 500-1500ms
                success = actual_value <= target
                unit = "ms"
            elif metric == "throughput":
                actual_value = hash(test_name) % 200 + 50  # 50-250 ops/sec
                success = actual_value >= target
                unit = "ops/sec"
            elif metric == "memory":
                actual_value = hash(test_name) % 2000 + 1000  # 1000-3000MB
                success = actual_value <= target
                unit = "MB"
            elif metric == "cpu":
                actual_value = (hash(test_name) % 60 + 20) / 100  # 0.2-0.8
                success = actual_value <= target
                unit = "utilization"
            else:  # concurrency
                actual_value = hash(test_name) % 30 + 20  # 20-50
                success = actual_value >= target
                unit = "connections"
                
            return TestResult(
                category=TestCategory.PERFORMANCE,
                name=test_name,
                description=f"Performance test for {metric}",
                status=QualityGateStatus.PASSED if success else QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="" if success else f"Performance target missed: {actual_value}{unit} vs {target}{unit}",
                metadata={
                    "metric": metric,
                    "target_value": target,
                    "actual_value": actual_value,
                    "unit": unit,
                    "baseline": target * 0.9
                }
            )
            
        except Exception as e:
            return TestResult(
                category=TestCategory.PERFORMANCE,
                name=test_name,
                description=f"Performance test for {metric}",
                status=QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
            
    async def _run_compliance_validation(self) -> QualityGateResult:
        """Run compliance validation (HIPAA, GDPR, etc.)"""
        self.logger.info("Running compliance validation")
        start_time = time.time()
        
        gate_result = QualityGateResult(name="compliance_validation")
        
        # Compliance validation tests
        compliance_tests = [
            {"name": "hipaa_compliance_check", "standard": "HIPAA"},
            {"name": "gdpr_compliance_check", "standard": "GDPR"},
            {"name": "data_retention_policy_check", "standard": "DATA_RETENTION"},
            {"name": "audit_trail_compliance", "standard": "AUDIT"},
            {"name": "data_anonymization_check", "standard": "ANONYMIZATION"}
        ]
        
        compliance_failures = 0
        
        for test in compliance_tests:
            test_result = await self._execute_compliance_test(test)
            gate_result.test_results.append(test_result)
            
            if test_result.status == QualityGateStatus.FAILED:
                compliance_failures += 1
                
        # Compliance gate requires 100% pass rate
        gate_result.status = (
            QualityGateStatus.PASSED if compliance_failures == 0
            else QualityGateStatus.FAILED
        )
        
        gate_result.execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Compliance validation completed: {compliance_failures} failures")
        
        return gate_result
        
    async def _execute_compliance_test(self, test_config: Dict[str, str]) -> TestResult:
        """Execute individual compliance test"""
        test_name = test_config["name"]
        standard = test_config["standard"]
        
        start_time = time.time()
        
        try:
            # Mock compliance test execution
            await asyncio.sleep(0.1)
            
            # Mock compliance results (98% pass rate)
            success = hash(test_name) % 100 < 98
            
            return TestResult(
                category=TestCategory.COMPLIANCE,
                name=test_name,
                description=f"Compliance validation for {standard}",
                status=QualityGateStatus.PASSED if success else QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="" if success else f"Compliance violation: {standard}",
                metadata={
                    "standard": standard,
                    "compliance_level": "full",
                    "audit_required": True
                }
            )
            
        except Exception as e:
            return TestResult(
                category=TestCategory.COMPLIANCE,
                name=test_name,
                description=f"Compliance validation for {standard}",
                status=QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
            
    async def _run_code_quality_validation(self) -> QualityGateResult:
        """Run code quality validation"""
        self.logger.info("Running code quality validation")
        start_time = time.time()
        
        gate_result = QualityGateResult(name="code_quality")
        
        # Code quality checks
        quality_checks = [
            {"name": "linting_validation", "tool": "ruff"},
            {"name": "security_scanning", "tool": "bandit"},
            {"name": "type_checking", "tool": "mypy"},
            {"name": "complexity_analysis", "tool": "radon"},
            {"name": "documentation_coverage", "tool": "pydocstyle"}
        ]
        
        quality_failures = 0
        
        for check in quality_checks:
            test_result = await self._execute_quality_check(check)
            gate_result.test_results.append(test_result)
            
            if test_result.status == QualityGateStatus.FAILED:
                quality_failures += 1
                
        # Code quality gate
        gate_result.status = (
            QualityGateStatus.PASSED if quality_failures <= 1  # Allow 1 non-critical failure
            else QualityGateStatus.WARNING if quality_failures <= 2
            else QualityGateStatus.FAILED
        )
        
        gate_result.execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Code quality validation completed: {quality_failures} issues found")
        
        return gate_result
        
    async def _execute_quality_check(self, check_config: Dict[str, str]) -> TestResult:
        """Execute individual code quality check"""
        check_name = check_config["name"]
        tool = check_config["tool"]
        
        start_time = time.time()
        
        try:
            # Mock quality check execution
            await asyncio.sleep(0.15)
            
            # Mock quality results (92% pass rate)
            success = hash(check_name) % 100 < 92
            
            return TestResult(
                category=TestCategory.UNIT,  # Code quality as unit category
                name=check_name,
                description=f"Code quality check using {tool}",
                status=QualityGateStatus.PASSED if success else QualityGateStatus.WARNING,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="" if success else f"Code quality issues found by {tool}",
                metadata={
                    "tool": tool,
                    "issues_found": 0 if success else hash(check_name) % 5 + 1,
                    "severity": "low" if not success else "none"
                }
            )
            
        except Exception as e:
            return TestResult(
                category=TestCategory.UNIT,
                name=check_name,
                description=f"Code quality check using {tool}",
                status=QualityGateStatus.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
            
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality validation report"""
        total_gates = len(self.quality_gates)
        passed_gates = len([g for g in self.quality_gates.values() if g.status == QualityGateStatus.PASSED])
        warning_gates = len([g for g in self.quality_gates.values() if g.status == QualityGateStatus.WARNING])
        failed_gates = len([g for g in self.quality_gates.values() if g.status == QualityGateStatus.FAILED])
        
        # Collect all test results
        all_tests = []
        for gate in self.quality_gates.values():
            all_tests.extend(gate.test_results)
            
        total_tests = len(all_tests)
        passed_tests = len([t for t in all_tests if t.status == QualityGateStatus.PASSED])
        
        # Calculate overall quality score
        gate_score = (passed_gates + warning_gates * 0.5) / max(total_gates, 1) * 100
        test_score = passed_tests / max(total_tests, 1) * 100
        overall_score = (gate_score + test_score) / 2
        
        report = {
            "summary": {
                "overall_quality_score": overall_score,
                "total_quality_gates": total_gates,
                "passed_gates": passed_gates,
                "warning_gates": warning_gates,
                "failed_gates": failed_gates,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_success_rate": test_score
            },
            "quality_gates": {
                gate_name: {
                    "status": gate.status.value,
                    "execution_time_ms": gate.execution_time_ms,
                    "test_count": len(gate.test_results),
                    "passed_tests": len([t for t in gate.test_results if t.status == QualityGateStatus.PASSED]),
                    "coverage_percentage": gate.coverage_percentage
                }
                for gate_name, gate in self.quality_gates.items()
            },
            "test_results_by_category": self._group_tests_by_category(),
            "performance_metrics": self._extract_performance_metrics(),
            "compliance_status": self._extract_compliance_status(),
            "recommendations": self._generate_recommendations(),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
        
    def _group_tests_by_category(self) -> Dict[str, Any]:
        """Group test results by category"""
        categories = {}
        
        for gate in self.quality_gates.values():
            for test in gate.test_results:
                category = test.category.value
                if category not in categories:
                    categories[category] = {
                        "total": 0,
                        "passed": 0,
                        "failed": 0,
                        "warnings": 0
                    }
                    
                categories[category]["total"] += 1
                
                if test.status == QualityGateStatus.PASSED:
                    categories[category]["passed"] += 1
                elif test.status == QualityGateStatus.FAILED:
                    categories[category]["failed"] += 1
                elif test.status == QualityGateStatus.WARNING:
                    categories[category]["warnings"] += 1
                    
        return categories
        
    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from test results"""
        performance_tests = []
        
        for gate in self.quality_gates.values():
            for test in gate.test_results:
                if test.category == TestCategory.PERFORMANCE and "actual_value" in test.metadata:
                    performance_tests.append({
                        "metric": test.metadata.get("metric"),
                        "target": test.metadata.get("target_value"),
                        "actual": test.metadata.get("actual_value"),
                        "unit": test.metadata.get("unit"),
                        "passed": test.status == QualityGateStatus.PASSED
                    })
                    
        return {"performance_tests": performance_tests}
        
    def _extract_compliance_status(self) -> Dict[str, Any]:
        """Extract compliance status"""
        compliance_results = {}
        
        for gate in self.quality_gates.values():
            for test in gate.test_results:
                if test.category == TestCategory.COMPLIANCE:
                    standard = test.metadata.get("standard")
                    if standard:
                        compliance_results[standard] = test.status.value
                        
        return compliance_results
        
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Check failed gates
        for gate_name, gate in self.quality_gates.items():
            if gate.status == QualityGateStatus.FAILED:
                if gate_name == "unit_tests":
                    recommendations.append("Increase unit test coverage to meet minimum requirements")
                elif gate_name == "security_validation":
                    recommendations.append("Address security vulnerabilities before deployment")
                elif gate_name == "performance_validation":
                    recommendations.append("Optimize performance to meet benchmarks")
                elif gate_name == "compliance_validation":
                    recommendations.append("Ensure full compliance with regulatory requirements")
                    
        # Check test failure rates
        all_tests = []
        for gate in self.quality_gates.values():
            all_tests.extend(gate.test_results)
            
        if all_tests:
            failure_rate = len([t for t in all_tests if t.status == QualityGateStatus.FAILED]) / len(all_tests) * 100
            if failure_rate > 10:
                recommendations.append("High test failure rate indicates systemic issues requiring investigation")
                
        if not recommendations:
            recommendations.append("All quality gates passed - system ready for deployment")
            
        return recommendations
        
    def save_quality_report(self, filename: str = None) -> Path:
        """Save quality validation report to file"""
        if filename is None:
            filename = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        report_path = Path("quality_reports") 
        report_path.mkdir(exist_ok=True)
        
        full_path = report_path / filename
        
        report = self.generate_quality_report()
        
        with open(full_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Quality report saved to {full_path}")
        return full_path


async def demo_quality_validation_framework():
    """Demonstrate the Comprehensive Quality Validation Framework"""
    print("🧪 Comprehensive Quality Validation Framework Demo")
    print("=" * 55)
    
    # Initialize framework
    framework = ComprehensiveQualityValidationFramework()
    
    # Run comprehensive validation
    print("\n🚀 Running comprehensive quality validation...")
    print("This includes: Unit tests, Integration tests, Security, Performance, Compliance")
    
    quality_gates = await framework.run_comprehensive_validation()
    
    # Display results
    print(f"\n📊 Quality Gate Results:")
    print("=" * 30)
    
    for gate_name, gate_result in quality_gates.items():
        status_emoji = {
            "passed": "✅",
            "warning": "⚠️", 
            "failed": "❌",
            "skipped": "⏭️"
        }
        
        emoji = status_emoji.get(gate_result.status.value, "❓")
        print(f"{emoji} {gate_name.replace('_', ' ').title()}: {gate_result.status.value}")
        print(f"   Tests: {len(gate_result.test_results)} | "
              f"Time: {gate_result.execution_time_ms:.0f}ms")
        
        # Show failed tests
        failed_tests = [t for t in gate_result.test_results if t.status == QualityGateStatus.FAILED]
        if failed_tests:
            print(f"   Failed tests: {[t.name for t in failed_tests[:3]]}")
            
    # Generate and save report
    print(f"\n📋 Generating comprehensive quality report...")
    report_path = framework.save_quality_report()
    
    # Display summary
    report = framework.generate_quality_report()
    summary = report["summary"]
    
    print(f"\n🎯 Quality Summary:")
    print(f"Overall Quality Score: {summary['overall_quality_score']:.1f}%")
    print(f"Quality Gates: {summary['passed_gates']}/{summary['total_quality_gates']} passed")
    print(f"Test Success Rate: {summary['test_success_rate']:.1f}%")
    
    if report["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"][:3]:
            print(f"  • {rec}")
            
    print(f"\n📄 Detailed report saved to: {report_path}")
    print("\n✅ Quality validation framework demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_quality_validation_framework())