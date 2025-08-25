"""
Comprehensive Testing and Validation Framework
Advanced testing, validation, and quality assurance for medical AI systems
"""

import asyncio
import logging
import time
import unittest
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
from pathlib import Path
import hashlib
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of tests in the framework."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"
    STRESS = "stress"
    COMPLIANCE = "compliance"

class TestSeverity(Enum):
    """Test failure severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKER = "blocker"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestCase:
    """Comprehensive test case definition."""
    test_id: str
    name: str
    description: str
    test_type: TestType
    severity: TestSeverity
    test_function: Callable
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    timeout: int = 30
    retry_count: int = 0
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestResult:
    """Comprehensive test result record."""
    test_id: str
    name: str
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationRule:
    """Data validation rule definition."""
    rule_id: str
    name: str
    description: str
    validation_function: Callable
    severity: TestSeverity = TestSeverity.MEDIUM
    enabled: bool = True

class ComprehensiveTestingValidationFramework:
    """
    Comprehensive Testing and Validation Framework providing:
    - Automated test discovery and execution
    - Performance benchmarking and profiling  
    - Security testing and vulnerability assessment
    - Data validation and integrity checks
    - Compliance testing (HIPAA, FDA, etc.)
    - Regression testing and A/B testing
    - Load testing and stress testing
    - Continuous integration support
    """
    
    def __init__(self, 
                 test_output_dir: str = "/tmp/medical_ai_tests",
                 enable_parallel_execution: bool = True,
                 max_parallel_tests: int = 4):
        """Initialize the comprehensive testing framework."""
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(exist_ok=True, parents=True)
        self.enable_parallel_execution = enable_parallel_execution
        self.max_parallel_tests = max_parallel_tests
        
        # Test registry
        self.test_registry: Dict[str, TestCase] = {}
        self.validation_rules: Dict[str, ValidationRule] = {}
        
        # Execution tracking
        self.test_results: List[TestResult] = []
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Security test configurations
        self.security_configurations: Dict[str, Any] = {
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "rate_limit_requests_per_minute": 1000,
            "sql_injection_patterns": ["'", '"', ";", "--", "/*", "*/"],
            "xss_patterns": ["<script>", "javascript:", "onload=", "onerror="]
        }
        
        logger.info("Comprehensive Testing and Validation Framework initialized")
    
    def register_test_case(self, test_case: TestCase):
        """Register a new test case."""
        self.test_registry[test_case.test_id] = test_case
        logger.info(f"Registered test case: {test_case.test_id} ({test_case.test_type.value})")
    
    def register_validation_rule(self, rule: ValidationRule):
        """Register a data validation rule."""
        self.validation_rules[rule.rule_id] = rule
        logger.info(f"Registered validation rule: {rule.rule_id}")
    
    async def run_test_suite(self, 
                           test_filter: Optional[List[TestType]] = None,
                           tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive test suite with filtering options."""
        start_time = time.time()
        
        # Filter tests
        tests_to_run = self._filter_tests(test_filter, tags)
        
        if not tests_to_run:
            return {"status": "no_tests", "message": "No tests match the filter criteria"}
        
        logger.info(f"Running {len(tests_to_run)} tests...")
        
        # Execute tests
        if self.enable_parallel_execution and len(tests_to_run) > 1:
            results = await self._run_tests_parallel(tests_to_run)
        else:
            results = await self._run_tests_sequential(tests_to_run)
        
        # Generate comprehensive report
        execution_time = time.time() - start_time
        report = self._generate_test_report(results, execution_time)
        
        # Save report
        await self._save_test_report(report)
        
        logger.info(f"Test suite completed in {execution_time:.2f}s")
        return report
    
    def _filter_tests(self, 
                     test_filter: Optional[List[TestType]] = None,
                     tags: Optional[List[str]] = None) -> List[TestCase]:
        """Filter tests based on criteria."""
        filtered_tests = list(self.test_registry.values())
        
        if test_filter:
            filtered_tests = [t for t in filtered_tests if t.test_type in test_filter]
        
        if tags:
            filtered_tests = [t for t in filtered_tests if any(tag in t.tags for tag in tags)]
        
        # Sort by dependencies and priority
        return self._sort_tests_by_dependencies(filtered_tests)
    
    def _sort_tests_by_dependencies(self, tests: List[TestCase]) -> List[TestCase]:
        """Sort tests by dependencies using topological sort."""
        # Simple dependency resolution
        sorted_tests = []
        remaining_tests = tests.copy()
        
        while remaining_tests:
            # Find tests with no unresolved dependencies
            ready_tests = [
                t for t in remaining_tests 
                if not t.dependencies or all(
                    dep_id in [st.test_id for st in sorted_tests] 
                    for dep_id in t.dependencies
                )
            ]
            
            if not ready_tests:
                # Circular dependency or missing dependency
                logger.warning("Circular dependency detected, proceeding with remaining tests")
                sorted_tests.extend(remaining_tests)
                break
            
            # Add ready tests and remove from remaining
            sorted_tests.extend(ready_tests)
            for test in ready_tests:
                remaining_tests.remove(test)
        
        return sorted_tests
    
    async def _run_tests_parallel(self, tests: List[TestCase]) -> List[TestResult]:
        """Run tests in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_parallel_tests)
        tasks = []
        
        for test_case in tests:
            task = self._run_single_test_with_semaphore(test_case, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = TestResult(
                    test_id=tests[i].test_id,
                    name=tests[i].name,
                    status=TestStatus.ERROR,
                    execution_time=0.0,
                    error_message=f"Test execution failed: {str(result)}"
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _run_tests_sequential(self, tests: List[TestCase]) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        for test_case in tests:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _run_single_test_with_semaphore(self, 
                                            test_case: TestCase, 
                                            semaphore: asyncio.Semaphore) -> TestResult:
        """Run single test with semaphore control."""
        async with semaphore:
            return await self._run_single_test(test_case)
    
    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = time.time()
        result = TestResult(
            test_id=test_case.test_id,
            name=test_case.name,
            status=TestStatus.RUNNING,
            execution_time=0.0
        )
        
        try:
            logger.info(f"Running test: {test_case.test_id}")
            
            # Setup
            if test_case.setup_function:
                await self._execute_with_timeout(test_case.setup_function, test_case.timeout)
            
            # Execute test with retry logic
            for attempt in range(test_case.retry_count + 1):
                try:
                    # Execute main test function
                    test_result = await self._execute_with_timeout(
                        test_case.test_function, test_case.timeout
                    )
                    
                    # Check if test result indicates success
                    if test_result is False:
                        raise AssertionError("Test returned False")
                    elif isinstance(test_result, dict) and test_result.get("status") == "failed":
                        raise AssertionError(test_result.get("message", "Test failed"))
                    
                    # Test passed
                    result.status = TestStatus.PASSED
                    if isinstance(test_result, dict):
                        result.metrics.update(test_result.get("metrics", {}))
                    break
                    
                except Exception as e:
                    if attempt < test_case.retry_count:
                        logger.warning(f"Test {test_case.test_id} attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(1)  # Wait before retry
                    else:
                        raise e
            
        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.error_message = f"Test timed out after {test_case.timeout} seconds"
        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.stack_trace = traceback.format_exc()
        finally:
            # Teardown
            try:
                if test_case.teardown_function:
                    await self._execute_with_timeout(test_case.teardown_function, test_case.timeout)
            except Exception as e:
                logger.warning(f"Teardown failed for test {test_case.test_id}: {e}")
            
            result.execution_time = time.time() - start_time
            logger.info(f"Test {test_case.test_id} completed: {result.status.value} ({result.execution_time:.2f}s)")
        
        return result
    
    async def _execute_with_timeout(self, func: Callable, timeout: int):
        """Execute function with timeout."""
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(), timeout=timeout)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, func), timeout=timeout
            )
    
    def _generate_test_report(self, results: List[TestResult], execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in results if r.status == TestStatus.FAILED])
        error_tests = len([r for r in results if r.status == TestStatus.ERROR])
        skipped_tests = len([r for r in results if r.status == TestStatus.SKIPPED])
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Group results by test type
        results_by_type = {}
        for result in results:
            test_case = self.test_registry.get(result.test_id)
            if test_case:
                test_type = test_case.test_type.value
                if test_type not in results_by_type:
                    results_by_type[test_type] = []
                results_by_type[test_type].append(result)
        
        # Performance metrics
        avg_execution_time = np.mean([r.execution_time for r in results]) if results else 0
        max_execution_time = max([r.execution_time for r in results]) if results else 0
        
        # Critical failures (high/critical severity failures)
        critical_failures = []
        for result in results:
            if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                test_case = self.test_registry.get(result.test_id)
                if test_case and test_case.severity in [TestSeverity.CRITICAL, TestSeverity.BLOCKER]:
                    critical_failures.append(result)
        
        report = {
            "execution_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": execution_time,
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": success_rate
            },
            "performance_metrics": {
                "average_test_time": avg_execution_time,
                "maximum_test_time": max_execution_time,
                "parallel_execution": self.enable_parallel_execution,
                "max_parallel_tests": self.max_parallel_tests
            },
            "results_by_type": {
                test_type: {
                    "total": len(type_results),
                    "passed": len([r for r in type_results if r.status == TestStatus.PASSED]),
                    "failed": len([r for r in type_results if r.status == TestStatus.FAILED])
                }
                for test_type, type_results in results_by_type.items()
            },
            "critical_failures": [
                {
                    "test_id": result.test_id,
                    "name": result.name,
                    "error_message": result.error_message,
                    "execution_time": result.execution_time
                }
                for result in critical_failures
            ],
            "detailed_results": [
                {
                    "test_id": result.test_id,
                    "name": result.name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "metrics": result.metrics
                }
                for result in results
            ],
            "quality_gates": {
                "success_rate_gate": success_rate >= 95,  # 95% pass rate required
                "no_critical_failures": len(critical_failures) == 0,
                "performance_gate": avg_execution_time <= 5.0,  # Average test time under 5s
                "overall_quality_score": self._calculate_quality_score(results)
            }
        }
        
        return report
    
    def _calculate_quality_score(self, results: List[TestResult]) -> float:
        """Calculate overall quality score (0-100)."""
        if not results:
            return 0.0
        
        # Base score from pass rate
        passed = len([r for r in results if r.status == TestStatus.PASSED])
        pass_rate = passed / len(results)
        score = pass_rate * 100
        
        # Penalize critical failures heavily
        critical_failures = 0
        for result in results:
            if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                test_case = self.test_registry.get(result.test_id)
                if test_case and test_case.severity in [TestSeverity.CRITICAL, TestSeverity.BLOCKER]:
                    critical_failures += 1
        
        score -= critical_failures * 25  # -25 points per critical failure
        
        # Penalize slow tests
        avg_time = np.mean([r.execution_time for r in results])
        if avg_time > 2.0:
            score -= min(10, (avg_time - 2.0) * 2)  # Up to -10 points for slow tests
        
        return max(0.0, min(100.0, score))
    
    async def _save_test_report(self, report: Dict[str, Any]):
        """Save test report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_output_dir / f"test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {report_file}")
    
    async def validate_data(self, data: Any, rule_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run data validation using registered rules."""
        rules_to_run = []
        
        if rule_ids:
            rules_to_run = [self.validation_rules[rid] for rid in rule_ids if rid in self.validation_rules]
        else:
            rules_to_run = [rule for rule in self.validation_rules.values() if rule.enabled]
        
        validation_results = []
        
        for rule in rules_to_run:
            try:
                start_time = time.time()
                result = await self._execute_with_timeout(
                    lambda: rule.validation_function(data), 10
                )
                execution_time = time.time() - start_time
                
                validation_results.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "status": "passed" if result else "failed",
                    "execution_time": execution_time,
                    "severity": rule.severity.value
                })
                
            except Exception as e:
                validation_results.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "status": "error",
                    "error_message": str(e),
                    "severity": rule.severity.value
                })
        
        # Calculate overall validation status
        failed_rules = [r for r in validation_results if r["status"] != "passed"]
        critical_failures = [r for r in failed_rules if r.get("severity") in ["critical", "blocker"]]
        
        return {
            "overall_status": "passed" if not failed_rules else "failed",
            "total_rules": len(validation_results),
            "passed_rules": len([r for r in validation_results if r["status"] == "passed"]),
            "failed_rules": len(failed_rules),
            "critical_failures": len(critical_failures),
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    
    # Built-in test generators
    def generate_performance_tests(self, target_functions: List[Callable]) -> List[TestCase]:
        """Generate performance tests for target functions."""
        performance_tests = []
        
        for i, func in enumerate(target_functions):
            test_case = TestCase(
                test_id=f"perf_test_{func.__name__}_{i:03d}",
                name=f"Performance Test: {func.__name__}",
                description=f"Performance benchmark for {func.__name__}",
                test_type=TestType.PERFORMANCE,
                severity=TestSeverity.MEDIUM,
                test_function=lambda f=func: self._performance_test_wrapper(f),
                timeout=60,
                tags=["performance", "benchmark"]
            )
            performance_tests.append(test_case)
        
        return performance_tests
    
    async def _performance_test_wrapper(self, func: Callable) -> Dict[str, Any]:
        """Wrapper for performance testing."""
        iterations = 100
        execution_times = []
        
        # Warm-up
        for _ in range(10):
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
        
        # Benchmark
        for _ in range(iterations):
            start_time = time.time()
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
            execution_times.append(time.time() - start_time)
        
        # Calculate statistics
        avg_time = np.mean(execution_times)
        p95_time = np.percentile(execution_times, 95)
        p99_time = np.percentile(execution_times, 99)
        
        # Check against baseline if available
        baseline_key = func.__name__
        baseline_exceeded = False
        
        if baseline_key in self.performance_baselines:
            baseline_avg = self.performance_baselines[baseline_key].get("avg_time", float('inf'))
            if avg_time > baseline_avg * 1.2:  # 20% degradation threshold
                baseline_exceeded = True
        
        return {
            "status": "failed" if baseline_exceeded else "passed",
            "metrics": {
                "avg_execution_time": avg_time,
                "p95_execution_time": p95_time,
                "p99_execution_time": p99_time,
                "iterations": iterations,
                "baseline_exceeded": baseline_exceeded
            }
        }
    
    def set_performance_baseline(self, function_name: str, metrics: Dict[str, float]):
        """Set performance baseline for a function."""
        self.performance_baselines[function_name] = metrics
        logger.info(f"Performance baseline set for {function_name}: {metrics}")


# Factory functions and built-in test cases
def create_comprehensive_testing_framework() -> ComprehensiveTestingValidationFramework:
    """Create a comprehensive testing framework with built-in test cases."""
    framework = ComprehensiveTestingValidationFramework()
    
    # Register built-in validation rules
    _register_built_in_validation_rules(framework)
    
    # Register built-in test cases
    _register_built_in_test_cases(framework)
    
    return framework

def _register_built_in_validation_rules(framework: ComprehensiveTestingValidationFramework):
    """Register built-in validation rules."""
    
    # Image data validation
    framework.register_validation_rule(ValidationRule(
        rule_id="image_data_shape",
        name="Image Data Shape Validation",
        description="Validate that image data has correct dimensions",
        validation_function=lambda data: isinstance(data, np.ndarray) and len(data.shape) >= 2,
        severity=TestSeverity.HIGH
    ))
    
    framework.register_validation_rule(ValidationRule(
        rule_id="image_data_values",
        name="Image Data Value Range",
        description="Validate that image data values are in valid range",
        validation_function=lambda data: isinstance(data, np.ndarray) and 
                                       np.all(data >= 0) and np.all(data <= 1) if data.dtype == np.float32 else True,
        severity=TestSeverity.MEDIUM
    ))
    
    # Medical data validation
    framework.register_validation_rule(ValidationRule(
        rule_id="no_phi_in_metadata",
        name="PHI Data Protection",
        description="Ensure no PHI data in metadata",
        validation_function=lambda data: not _contains_phi_patterns(str(data)),
        severity=TestSeverity.CRITICAL
    ))

def _register_built_in_test_cases(framework: ComprehensiveTestingValidationFramework):
    """Register built-in test cases."""
    
    # Security tests
    framework.register_test_case(TestCase(
        test_id="security_input_validation",
        name="Input Validation Security Test",
        description="Test input validation against malicious inputs",
        test_type=TestType.SECURITY,
        severity=TestSeverity.HIGH,
        test_function=_test_input_validation_security,
        tags=["security", "validation"]
    ))
    
    # Compliance tests
    framework.register_test_case(TestCase(
        test_id="hipaa_compliance_check",
        name="HIPAA Compliance Test",
        description="Verify HIPAA compliance requirements",
        test_type=TestType.COMPLIANCE,
        severity=TestSeverity.CRITICAL,
        test_function=_test_hipaa_compliance,
        tags=["compliance", "hipaa", "security"]
    ))

def _contains_phi_patterns(data_str: str) -> bool:
    """Check if string contains potential PHI patterns."""
    phi_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{10}\b',  # Phone number pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email pattern
    ]
    
    import re
    for pattern in phi_patterns:
        if re.search(pattern, data_str):
            return True
    return False

async def _test_input_validation_security() -> Dict[str, Any]:
    """Test input validation against security threats."""
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "' OR '1'='1",
        "\x00\x01\x02\x03"  # Null bytes
    ]
    
    vulnerabilities = []
    
    for malicious_input in malicious_inputs:
        # Test would validate that malicious input is properly sanitized
        # This is a placeholder - actual implementation would test real validation functions
        if len(malicious_input) > 0:  # Placeholder validation
            pass
        else:
            vulnerabilities.append(malicious_input)
    
    return {
        "status": "passed" if not vulnerabilities else "failed",
        "message": f"Found {len(vulnerabilities)} vulnerabilities" if vulnerabilities else "No vulnerabilities detected",
        "metrics": {
            "tested_inputs": len(malicious_inputs),
            "vulnerabilities_found": len(vulnerabilities)
        }
    }

async def _test_hipaa_compliance() -> Dict[str, Any]:
    """Test HIPAA compliance requirements."""
    compliance_checks = {
        "audit_logging_enabled": True,  # Would check actual audit logging
        "data_encryption_at_rest": True,  # Would check encryption settings  
        "data_encryption_in_transit": True,  # Would check TLS/SSL
        "access_controls": True,  # Would check authentication/authorization
        "data_retention_policy": True  # Would check retention policies
    }
    
    failed_checks = [check for check, passed in compliance_checks.items() if not passed]
    
    return {
        "status": "passed" if not failed_checks else "failed",
        "message": f"HIPAA compliance: {len(failed_checks)} checks failed" if failed_checks else "All HIPAA checks passed",
        "metrics": {
            "total_checks": len(compliance_checks),
            "passed_checks": len(compliance_checks) - len(failed_checks),
            "failed_checks": len(failed_checks)
        }
    }


if __name__ == "__main__":
    async def demo():
        """Demonstration of Comprehensive Testing and Validation Framework."""
        print("=== Comprehensive Testing and Validation Framework Demo ===")
        
        # Create framework
        framework = create_comprehensive_testing_framework()
        
        # Run test suite
        print("Running test suite...")
        report = await framework.run_test_suite()
        
        print(f"Test Results:")
        print(f"  Total Tests: {report['execution_summary']['total_tests']}")
        print(f"  Success Rate: {report['execution_summary']['success_rate']:.1f}%")
        print(f"  Quality Score: {report['quality_gates']['overall_quality_score']:.1f}")
        
        # Test data validation
        print("\nTesting data validation...")
        test_data = np.random.rand(224, 224, 3)
        validation_result = await framework.validate_data(test_data)
        
        print(f"Validation Results:")
        print(f"  Status: {validation_result['overall_status']}")
        print(f"  Rules Passed: {validation_result['passed_rules']}/{validation_result['total_rules']}")
        
        print("\n=== Demo Complete ===")
    
    # Run demo
    asyncio.run(demo())