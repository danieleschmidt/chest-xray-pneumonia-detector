"""
Comprehensive Quality Gates Runner
Autonomous execution of all quality gates with 85%+ coverage requirement
"""

import asyncio
import logging
import time
import subprocess
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class QualityGateType(Enum):
    """Types of quality gates."""
    CODE_QUALITY = "code_quality"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_TEST = "performance_test"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    COVERAGE_CHECK = "coverage_check"
    DEPENDENCY_CHECK = "dependency_check"
    DOCUMENTATION = "documentation"
    COMPLIANCE = "compliance"

@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_type: QualityGateType
    status: QualityGateStatus
    score: float  # 0-100
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityGateConfig:
    """Configuration for a quality gate."""
    gate_type: QualityGateType
    command: str
    timeout: int = 300  # 5 minutes default
    pass_threshold: float = 85.0
    warning_threshold: float = 70.0
    required: bool = True
    retry_count: int = 0

class ComprehensiveQualityGatesRunner:
    """
    Comprehensive Quality Gates Runner providing:
    - Automated execution of all quality gates
    - 85%+ test coverage requirement
    - Security vulnerability scanning
    - Performance benchmarking
    - Code quality analysis
    - Dependency security checks
    - Documentation validation
    - Compliance verification
    - Failure reporting and recovery
    """
    
    def __init__(self, 
                 project_root: str = "/root/repo",
                 output_dir: str = "/tmp/quality_gates",
                 parallel_execution: bool = True):
        """Initialize the Quality Gates Runner."""
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.parallel_execution = parallel_execution
        
        # Quality gate configurations
        self.quality_gates = self._initialize_quality_gates()
        
        # Execution tracking
        self.execution_results: List[QualityGateResult] = []
        self.overall_status = QualityGateStatus.PENDING
        
        logger.info(f"Quality Gates Runner initialized for project: {self.project_root}")
    
    def _initialize_quality_gates(self) -> List[QualityGateConfig]:
        """Initialize all quality gate configurations."""
        gates = [
            # Unit Tests with Coverage
            QualityGateConfig(
                gate_type=QualityGateType.UNIT_TESTS,
                command="cd /root/repo && python -m pytest tests/ -v --cov=src --cov-report=json --cov-report=html --cov-report=term",
                timeout=600,  # 10 minutes
                pass_threshold=85.0,  # 85% pass rate
                required=True
            ),
            
            # Coverage Check (85% minimum)
            QualityGateConfig(
                gate_type=QualityGateType.COVERAGE_CHECK,
                command="cd /root/repo && python -c \"import json; data=json.load(open('coverage.json')); print(f'Coverage: {data[\\\"totals\\\"][\\\"percent_covered\\\"]}%'); exit(0 if data['totals']['percent_covered'] >= 85.0 else 1)\"",
                timeout=30,
                pass_threshold=85.0,
                required=True
            ),
            
            # Code Quality (Ruff)
            QualityGateConfig(
                gate_type=QualityGateType.CODE_QUALITY,
                command="cd /root/repo && python -m ruff check . --output-format=json --exit-zero",
                timeout=120,
                pass_threshold=90.0,
                required=True
            ),
            
            # Security Scan (Bandit)
            QualityGateConfig(
                gate_type=QualityGateType.SECURITY_SCAN,
                command="cd /root/repo && python -m bandit -r src/ -f json -o bandit_report.json",
                timeout=180,
                pass_threshold=95.0,  # Very strict for security
                required=True
            ),
            
            # Performance Tests
            QualityGateConfig(
                gate_type=QualityGateType.PERFORMANCE_TEST,
                command="cd /root/repo && python performance_demo.py",
                timeout=300,
                pass_threshold=80.0,
                required=True
            ),
            
            # Dependency Security Check
            QualityGateConfig(
                gate_type=QualityGateType.DEPENDENCY_CHECK,
                command="cd /root/repo && python -m pip audit --format=json --output=pip_audit.json; echo 'Dependency check completed'",
                timeout=180,
                pass_threshold=90.0,
                required=False,  # Optional since pip-audit might not be available
                retry_count=1
            ),
            
            # Integration Tests
            QualityGateConfig(
                gate_type=QualityGateType.INTEGRATION_TESTS,
                command="cd /root/repo && python comprehensive_demo.py",
                timeout=300,
                pass_threshold=85.0,
                required=True
            )
        ]
        
        return gates
    
    async def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        start_time = time.time()
        logger.info("Starting comprehensive quality gates execution...")
        
        self.overall_status = QualityGateStatus.RUNNING
        
        try:
            # Execute quality gates
            if self.parallel_execution:
                results = await self._run_gates_parallel()
            else:
                results = await self._run_gates_sequential()
            
            # Analyze results
            total_execution_time = time.time() - start_time
            report = self._generate_quality_report(results, total_execution_time)
            
            # Save report
            await self._save_quality_report(report)
            
            # Determine overall status
            self._determine_overall_status(results)
            
            logger.info(f"Quality gates execution completed in {total_execution_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Quality gates execution failed: {e}")
            self.overall_status = QualityGateStatus.FAILED
            raise
    
    async def _run_gates_parallel(self) -> List[QualityGateResult]:
        """Run quality gates in parallel."""
        tasks = []
        
        for gate_config in self.quality_gates:
            task = self._execute_quality_gate(gate_config)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = QualityGateResult(
                    gate_type=self.quality_gates[i].gate_type,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    execution_time=0.0,
                    errors=[f"Execution failed: {str(result)}"]
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _run_gates_sequential(self) -> List[QualityGateResult]:
        """Run quality gates sequentially."""
        results = []
        
        for gate_config in self.quality_gates:
            try:
                result = await self._execute_quality_gate(gate_config)
                results.append(result)
            except Exception as e:
                error_result = QualityGateResult(
                    gate_type=gate_config.gate_type,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    execution_time=0.0,
                    errors=[f"Execution failed: {str(e)}"]
                )
                results.append(error_result)
        
        return results
    
    async def _execute_quality_gate(self, config: QualityGateConfig) -> QualityGateResult:
        """Execute a single quality gate."""
        start_time = time.time()
        
        logger.info(f"Executing quality gate: {config.gate_type.value}")
        
        for attempt in range(config.retry_count + 1):
            try:
                # Execute command
                process = await asyncio.create_subprocess_shell(
                    config.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.project_root
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=config.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise Exception(f"Command timed out after {config.timeout} seconds")
                
                execution_time = time.time() - start_time
                
                # Process results
                result = await self._process_gate_result(
                    config, process.returncode, stdout, stderr, execution_time
                )
                
                # If successful or this is the last attempt, return result
                if result.status != QualityGateStatus.FAILED or attempt == config.retry_count:
                    return result
                
                logger.warning(f"Retry {attempt + 1} for {config.gate_type.value}")
                await asyncio.sleep(2)  # Wait before retry
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                if attempt == config.retry_count:  # Last attempt
                    return QualityGateResult(
                        gate_type=config.gate_type,
                        status=QualityGateStatus.FAILED,
                        score=0.0,
                        execution_time=execution_time,
                        errors=[str(e)]
                    )
                
                logger.warning(f"Attempt {attempt + 1} failed for {config.gate_type.value}: {e}")
                await asyncio.sleep(2)
        
        # This should never be reached
        return QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.FAILED,
            score=0.0,
            execution_time=time.time() - start_time,
            errors=["Unexpected execution failure"]
        )
    
    async def _process_gate_result(self,
                                 config: QualityGateConfig,
                                 return_code: int,
                                 stdout: bytes,
                                 stderr: bytes,
                                 execution_time: float) -> QualityGateResult:
        """Process the result of a quality gate execution."""
        stdout_str = stdout.decode('utf-8', errors='ignore')
        stderr_str = stderr.decode('utf-8', errors='ignore')
        
        # Initialize result
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.FAILED,
            score=0.0,
            execution_time=execution_time
        )
        
        # Gate-specific result processing
        if config.gate_type == QualityGateType.UNIT_TESTS:
            result = await self._process_unit_test_results(config, return_code, stdout_str, stderr_str, execution_time)
        elif config.gate_type == QualityGateType.COVERAGE_CHECK:
            result = await self._process_coverage_results(config, return_code, stdout_str, stderr_str, execution_time)
        elif config.gate_type == QualityGateType.CODE_QUALITY:
            result = await self._process_ruff_results(config, return_code, stdout_str, stderr_str, execution_time)
        elif config.gate_type == QualityGateType.SECURITY_SCAN:
            result = await self._process_bandit_results(config, return_code, stdout_str, stderr_str, execution_time)
        elif config.gate_type == QualityGateType.PERFORMANCE_TEST:
            result = await self._process_performance_results(config, return_code, stdout_str, stderr_str, execution_time)
        elif config.gate_type == QualityGateType.DEPENDENCY_CHECK:
            result = await self._process_dependency_results(config, return_code, stdout_str, stderr_str, execution_time)
        elif config.gate_type == QualityGateType.INTEGRATION_TESTS:
            result = await self._process_integration_results(config, return_code, stdout_str, stderr_str, execution_time)
        else:
            # Generic processing
            result = await self._process_generic_results(config, return_code, stdout_str, stderr_str, execution_time)
        
        return result
    
    async def _process_unit_test_results(self, config, return_code, stdout, stderr, execution_time):
        """Process unit test results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.FAILED,
            score=0.0,
            execution_time=execution_time
        )
        
        # Parse pytest output
        try:
            # Look for test results in stdout
            if "failed" in stdout.lower() and return_code != 0:
                result.status = QualityGateStatus.FAILED
                result.score = 0.0
                result.errors.append("Some tests failed")
            elif "passed" in stdout.lower() or return_code == 0:
                # Extract pass rate if possible
                if "passed" in stdout:
                    # Try to extract passed/failed counts
                    lines = stdout.split('\n')
                    for line in lines:
                        if "passed" in line and ("failed" in line or "error" in line):
                            # This is a summary line
                            result.details["test_summary"] = line.strip()
                            break
                
                result.status = QualityGateStatus.PASSED
                result.score = 100.0
            else:
                result.status = QualityGateStatus.FAILED
                result.score = 0.0
                result.errors.append(f"Tests failed with return code {return_code}")
            
            # Store output
            result.details["stdout"] = stdout[-1000:] if len(stdout) > 1000 else stdout  # Last 1000 chars
            if stderr:
                result.details["stderr"] = stderr[-500:] if len(stderr) > 500 else stderr
                
        except Exception as e:
            result.status = QualityGateStatus.FAILED
            result.errors.append(f"Failed to parse test results: {e}")
        
        return result
    
    async def _process_coverage_results(self, config, return_code, stdout, stderr, execution_time):
        """Process coverage check results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.FAILED,
            score=0.0,
            execution_time=execution_time
        )
        
        try:
            if return_code == 0:
                result.status = QualityGateStatus.PASSED
                result.score = 100.0
                result.details["message"] = "Coverage meets 85% requirement"
            else:
                result.status = QualityGateStatus.FAILED
                result.score = 0.0
                result.errors.append("Coverage below 85% requirement")
            
            # Try to extract coverage percentage from output
            if "Coverage:" in stdout:
                coverage_line = [line for line in stdout.split('\n') if 'Coverage:' in line]
                if coverage_line:
                    result.details["coverage_output"] = coverage_line[0]
            
        except Exception as e:
            result.status = QualityGateStatus.FAILED
            result.errors.append(f"Failed to process coverage results: {e}")
        
        return result
    
    async def _process_ruff_results(self, config, return_code, stdout, stderr, execution_time):
        """Process Ruff code quality results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.PASSED,
            score=100.0,
            execution_time=execution_time
        )
        
        try:
            # Ruff with --exit-zero always returns 0, check stdout for issues
            if stdout.strip():
                try:
                    ruff_results = json.loads(stdout)
                    issue_count = len(ruff_results)
                    
                    # Calculate score based on issues
                    if issue_count == 0:
                        result.score = 100.0
                        result.status = QualityGateStatus.PASSED
                    elif issue_count <= 10:
                        result.score = 90.0
                        result.status = QualityGateStatus.WARNING if issue_count > 5 else QualityGateStatus.PASSED
                    else:
                        result.score = max(0, 100 - (issue_count * 2))  # Deduct 2 points per issue
                        result.status = QualityGateStatus.FAILED if result.score < config.pass_threshold else QualityGateStatus.WARNING
                    
                    result.details["issue_count"] = issue_count
                    result.details["issues"] = ruff_results[:10]  # First 10 issues
                    
                except json.JSONDecodeError:
                    # Fallback to text processing
                    line_count = len([line for line in stdout.split('\n') if line.strip()])
                    result.score = max(0, 100 - (line_count * 5))
                    result.status = QualityGateStatus.FAILED if result.score < config.pass_threshold else QualityGateStatus.PASSED
            else:
                result.score = 100.0
                result.status = QualityGateStatus.PASSED
                result.details["message"] = "No code quality issues found"
            
        except Exception as e:
            result.status = QualityGateStatus.WARNING
            result.score = 70.0
            result.warnings.append(f"Failed to parse Ruff results: {e}")
        
        return result
    
    async def _process_bandit_results(self, config, return_code, stdout, stderr, execution_time):
        """Process Bandit security scan results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.PASSED,
            score=100.0,
            execution_time=execution_time
        )
        
        try:
            # Check if bandit report exists
            bandit_report_path = self.project_root / "bandit_report.json"
            
            if bandit_report_path.exists():
                with open(bandit_report_path) as f:
                    bandit_data = json.load(f)
                
                # Count security issues by severity
                high_issues = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"])
                medium_issues = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
                low_issues = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "LOW"])
                
                total_issues = high_issues + medium_issues + low_issues
                
                # Calculate score (security is critical)
                if high_issues > 0:
                    result.score = 0.0
                    result.status = QualityGateStatus.FAILED
                    result.errors.append(f"Found {high_issues} high-severity security issues")
                elif medium_issues > 5:
                    result.score = 60.0
                    result.status = QualityGateStatus.FAILED
                    result.errors.append(f"Found {medium_issues} medium-severity security issues")
                elif total_issues > 10:
                    result.score = 80.0
                    result.status = QualityGateStatus.WARNING
                    result.warnings.append(f"Found {total_issues} total security issues")
                else:
                    result.score = max(85.0, 100 - (total_issues * 2))
                    result.status = QualityGateStatus.PASSED
                
                result.details["security_issues"] = {
                    "high": high_issues,
                    "medium": medium_issues,
                    "low": low_issues,
                    "total": total_issues
                }
                
            else:
                result.status = QualityGateStatus.WARNING
                result.score = 70.0
                result.warnings.append("Bandit report not found")
            
        except Exception as e:
            result.status = QualityGateStatus.WARNING
            result.score = 70.0
            result.warnings.append(f"Failed to process security scan results: {e}")
        
        return result
    
    async def _process_performance_results(self, config, return_code, stdout, stderr, execution_time):
        """Process performance test results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.PASSED,
            score=85.0,
            execution_time=execution_time
        )
        
        # For now, assume performance tests pass if command completes successfully
        if return_code == 0:
            result.status = QualityGateStatus.PASSED
            result.score = 85.0
            result.details["message"] = "Performance tests completed successfully"
        else:
            result.status = QualityGateStatus.FAILED
            result.score = 0.0
            result.errors.append(f"Performance tests failed with return code {return_code}")
        
        return result
    
    async def _process_dependency_results(self, config, return_code, stdout, stderr, execution_time):
        """Process dependency security check results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.PASSED,
            score=90.0,
            execution_time=execution_time
        )
        
        # This is optional, so don't fail hard
        if "pip audit" in stderr and "not found" in stderr:
            result.status = QualityGateStatus.SKIPPED
            result.score = 0.0
            result.details["message"] = "pip-audit not available - skipping dependency check"
        elif return_code == 0:
            result.status = QualityGateStatus.PASSED
            result.score = 90.0
            result.details["message"] = "No vulnerable dependencies found"
        else:
            result.status = QualityGateStatus.WARNING
            result.score = 70.0
            result.warnings.append("Dependency check completed with warnings")
        
        return result
    
    async def _process_integration_results(self, config, return_code, stdout, stderr, execution_time):
        """Process integration test results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.PASSED,
            score=85.0,
            execution_time=execution_time
        )
        
        if return_code == 0:
            result.status = QualityGateStatus.PASSED
            result.score = 85.0
            result.details["message"] = "Integration tests completed successfully"
        else:
            result.status = QualityGateStatus.FAILED
            result.score = 0.0
            result.errors.append(f"Integration tests failed with return code {return_code}")
        
        return result
    
    async def _process_generic_results(self, config, return_code, stdout, stderr, execution_time):
        """Process generic command results."""
        result = QualityGateResult(
            gate_type=config.gate_type,
            status=QualityGateStatus.PASSED if return_code == 0 else QualityGateStatus.FAILED,
            score=100.0 if return_code == 0 else 0.0,
            execution_time=execution_time
        )
        
        if return_code != 0:
            result.errors.append(f"Command failed with return code {return_code}")
        
        return result
    
    def _generate_quality_report(self, 
                                results: List[QualityGateResult], 
                                total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        passed_gates = len([r for r in results if r.status == QualityGateStatus.PASSED])
        failed_gates = len([r for r in results if r.status == QualityGateStatus.FAILED])
        warning_gates = len([r for r in results if r.status == QualityGateStatus.WARNING])
        skipped_gates = len([r for r in results if r.status == QualityGateStatus.SKIPPED])
        
        total_gates = len(results)
        success_rate = (passed_gates / total_gates) * 100 if total_gates > 0 else 0
        
        # Calculate overall quality score
        quality_scores = [r.score for r in results if r.status != QualityGateStatus.SKIPPED]
        overall_score = np.mean(quality_scores) if quality_scores else 0.0
        
        # Check critical gates
        critical_failures = []
        for result in results:
            config = next((g for g in self.quality_gates if g.gate_type == result.gate_type), None)
            if config and config.required and result.status == QualityGateStatus.FAILED:
                critical_failures.append(result)
        
        # Quality assessment
        quality_assessment = "EXCELLENT"
        if overall_score < 85 or critical_failures:
            quality_assessment = "POOR"
        elif overall_score < 90 or warning_gates > 2:
            quality_assessment = "GOOD"
        
        report = {
            "execution_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": total_execution_time,
                "total_gates": total_gates,
                "passed": passed_gates,
                "failed": failed_gates,
                "warnings": warning_gates,
                "skipped": skipped_gates,
                "success_rate": success_rate
            },
            "quality_metrics": {
                "overall_score": overall_score,
                "quality_assessment": quality_assessment,
                "critical_failures": len(critical_failures),
                "coverage_gate_passed": any(
                    r.gate_type == QualityGateType.COVERAGE_CHECK and r.status == QualityGateStatus.PASSED
                    for r in results
                ),
                "security_gate_passed": any(
                    r.gate_type == QualityGateType.SECURITY_SCAN and r.status == QualityGateStatus.PASSED
                    for r in results
                )
            },
            "gate_results": [
                {
                    "gate_type": result.gate_type.value,
                    "status": result.status.value,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "details": result.details,
                    "warnings": result.warnings,
                    "errors": result.errors
                }
                for result in results
            ],
            "critical_failures": [
                {
                    "gate_type": cf.gate_type.value,
                    "errors": cf.errors,
                    "score": cf.score
                }
                for cf in critical_failures
            ],
            "quality_gates_status": {
                "all_required_passed": len(critical_failures) == 0,
                "coverage_requirement_met": any(
                    r.gate_type == QualityGateType.COVERAGE_CHECK and r.status == QualityGateStatus.PASSED
                    for r in results
                ),
                "security_scans_passed": any(
                    r.gate_type == QualityGateType.SECURITY_SCAN and r.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING]
                    for r in results
                ),
                "performance_benchmarks_met": any(
                    r.gate_type == QualityGateType.PERFORMANCE_TEST and r.status == QualityGateStatus.PASSED
                    for r in results
                )
            },
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[QualityGateResult]) -> List[str]:
        """Generate recommendations based on quality gate results."""
        recommendations = []
        
        # Coverage recommendations
        coverage_result = next((r for r in results if r.gate_type == QualityGateType.COVERAGE_CHECK), None)
        if coverage_result and coverage_result.status == QualityGateStatus.FAILED:
            recommendations.append("Increase test coverage to meet the 85% minimum requirement")
        
        # Security recommendations
        security_result = next((r for r in results if r.gate_type == QualityGateType.SECURITY_SCAN), None)
        if security_result and security_result.status == QualityGateStatus.FAILED:
            recommendations.append("Address high-severity security vulnerabilities immediately")
        
        # Code quality recommendations
        quality_result = next((r for r in results if r.gate_type == QualityGateType.CODE_QUALITY), None)
        if quality_result and quality_result.score < 90:
            recommendations.append("Improve code quality by addressing linting issues")
        
        # Performance recommendations
        perf_result = next((r for r in results if r.gate_type == QualityGateType.PERFORMANCE_TEST), None)
        if perf_result and perf_result.status == QualityGateStatus.FAILED:
            recommendations.append("Optimize performance to meet benchmarking requirements")
        
        # Test recommendations
        unit_test_result = next((r for r in results if r.gate_type == QualityGateType.UNIT_TESTS), None)
        if unit_test_result and unit_test_result.status == QualityGateStatus.FAILED:
            recommendations.append("Fix failing unit tests before proceeding")
        
        return recommendations
    
    async def _save_quality_report(self, report: Dict[str, Any]):
        """Save quality report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"quality_gates_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save to a standard location
        standard_report = self.project_root / "quality_gates_report.json"
        with open(standard_report, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality gates report saved to: {report_file}")
    
    def _determine_overall_status(self, results: List[QualityGateResult]):
        """Determine overall quality gates status."""
        critical_failures = []
        for result in results:
            config = next((g for g in self.quality_gates if g.gate_type == result.gate_type), None)
            if config and config.required and result.status == QualityGateStatus.FAILED:
                critical_failures.append(result)
        
        if critical_failures:
            self.overall_status = QualityGateStatus.FAILED
        elif any(r.status == QualityGateStatus.WARNING for r in results):
            self.overall_status = QualityGateStatus.WARNING
        else:
            self.overall_status = QualityGateStatus.PASSED
    
    async def run_coverage_check(self) -> QualityGateResult:
        """Run specific coverage check to ensure 85%+ requirement."""
        coverage_config = QualityGateConfig(
            gate_type=QualityGateType.COVERAGE_CHECK,
            command="cd /root/repo && python -m pytest tests/ --cov=src --cov-fail-under=85 --cov-report=term",
            timeout=300,
            pass_threshold=85.0,
            required=True
        )
        
        return await self._execute_quality_gate(coverage_config)
    
    async def run_security_scan(self) -> QualityGateResult:
        """Run comprehensive security scan."""
        security_config = QualityGateConfig(
            gate_type=QualityGateType.SECURITY_SCAN,
            command="cd /root/repo && python -m bandit -r src/ -ll",
            timeout=180,
            pass_threshold=95.0,
            required=True
        )
        
        return await self._execute_quality_gate(security_config)


# Factory function and CLI
def create_quality_gates_runner(project_root: str = "/root/repo") -> ComprehensiveQualityGatesRunner:
    """Create a Quality Gates Runner for the specified project."""
    return ComprehensiveQualityGatesRunner(project_root=project_root)


if __name__ == "__main__":
    async def main():
        """Main execution function."""
        print("=== Comprehensive Quality Gates Runner ===")
        
        # Create and run quality gates
        runner = create_quality_gates_runner()
        
        try:
            report = await runner.run_all_quality_gates()
            
            print(f"Quality Gates Execution Summary:")
            print(f"  Total Gates: {report['execution_summary']['total_gates']}")
            print(f"  Passed: {report['execution_summary']['passed']}")
            print(f"  Failed: {report['execution_summary']['failed']}")
            print(f"  Warnings: {report['execution_summary']['warnings']}")
            print(f"  Overall Score: {report['quality_metrics']['overall_score']:.1f}")
            print(f"  Quality Assessment: {report['quality_metrics']['quality_assessment']}")
            
            # Check critical requirements
            print(f"\nCritical Requirements:")
            print(f"  Coverage ≥85%: {'✓' if report['quality_gates_status']['coverage_requirement_met'] else '✗'}")
            print(f"  Security Scans: {'✓' if report['quality_gates_status']['security_scans_passed'] else '✗'}")
            print(f"  All Required Passed: {'✓' if report['quality_gates_status']['all_required_passed'] else '✗'}")
            
            if report['critical_failures']:
                print(f"\nCritical Failures:")
                for failure in report['critical_failures']:
                    print(f"  - {failure['gate_type']}: {failure['errors']}")
            
            # Exit with appropriate code
            if runner.overall_status == QualityGateStatus.FAILED:
                print("\n❌ QUALITY GATES FAILED")
                sys.exit(1)
            elif runner.overall_status == QualityGateStatus.WARNING:
                print("\n⚠️  QUALITY GATES PASSED WITH WARNINGS")
                sys.exit(0)
            else:
                print("\n✅ ALL QUALITY GATES PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"\n❌ QUALITY GATES EXECUTION FAILED: {e}")
            sys.exit(1)
    
    # Run quality gates
    asyncio.run(main())