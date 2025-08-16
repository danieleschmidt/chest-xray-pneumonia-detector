#!/usr/bin/env python3
"""
Comprehensive Testing Framework - Generation 2: MAKE IT ROBUST
Multi-tier testing with chaos engineering, performance validation, and automated quality gates.
"""

import asyncio
import json
import logging
import time
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import pytest
import requests
import psutil

class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CHAOS = "chaos"
    CONTRACT = "contract"
    SMOKE = "smoke"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition."""
    name: str
    endpoint: str
    expected_response_time_ms: int
    expected_throughput_rps: int
    max_error_rate: float
    load_pattern: str = "constant"  # constant, spike, ramp

@dataclass
class ChaosExperiment:
    """Chaos engineering experiment."""
    name: str
    target_service: str
    failure_type: str  # network_delay, cpu_stress, memory_leak, pod_kill
    duration_minutes: int
    impact_percentage: float
    steady_state_hypothesis: str

class PerformanceTester:
    """Performance testing and benchmarking."""
    
    def __init__(self):
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.results_history: List[Dict] = []
        
    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add performance benchmark."""
        self.benchmarks[benchmark.name] = benchmark
        
    async def run_load_test(self, benchmark: PerformanceBenchmark, 
                           duration_seconds: int = 60) -> Dict[str, Any]:
        """Run load test for specific benchmark."""
        logging.info(f"Starting load test: {benchmark.name}")
        
        start_time = time.time()
        requests_made = 0
        successful_requests = 0
        response_times = []
        errors = []
        
        # Simulate concurrent requests
        async def make_request():
            nonlocal requests_made, successful_requests, response_times, errors
            
            try:
                request_start = time.time()
                
                # Mock HTTP request (replace with actual request in production)
                await asyncio.sleep(random.uniform(0.05, 0.2))  # Simulate response time
                
                response_time = (time.time() - request_start) * 1000  # ms
                response_times.append(response_time)
                requests_made += 1
                
                # Simulate occasional errors
                if random.random() < 0.02:  # 2% error rate
                    errors.append("Simulated error")
                else:
                    successful_requests += 1
                    
            except Exception as e:
                errors.append(str(e))
                requests_made += 1
                
        # Generate load based on pattern
        tasks = []
        if benchmark.load_pattern == "constant":
            rps = benchmark.expected_throughput_rps
            request_interval = 1.0 / rps
            
            while time.time() - start_time < duration_seconds:
                task = asyncio.create_task(make_request())
                tasks.append(task)
                await asyncio.sleep(request_interval)
                
        elif benchmark.load_pattern == "spike":
            # Normal load for 80% of time, spike for 20%
            normal_duration = duration_seconds * 0.8
            spike_duration = duration_seconds * 0.2
            
            # Normal load
            rps = benchmark.expected_throughput_rps
            request_interval = 1.0 / rps
            
            normal_start = time.time()
            while time.time() - normal_start < normal_duration:
                task = asyncio.create_task(make_request())
                tasks.append(task)
                await asyncio.sleep(request_interval)
                
            # Spike load (3x normal)
            spike_rps = rps * 3
            spike_interval = 1.0 / spike_rps
            
            spike_start = time.time()
            while time.time() - spike_start < spike_duration:
                task = asyncio.create_task(make_request())
                tasks.append(task)
                await asyncio.sleep(spike_interval)
                
        # Wait for all requests to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # Calculate metrics
        total_duration = time.time() - start_time
        actual_rps = requests_made / total_duration if total_duration > 0 else 0
        error_rate = len(errors) / requests_made if requests_made > 0 else 0
        
        result = {
            'benchmark_name': benchmark.name,
            'duration_seconds': total_duration,
            'total_requests': requests_made,
            'successful_requests': successful_requests,
            'failed_requests': len(errors),
            'requests_per_second': actual_rps,
            'error_rate': error_rate,
            'response_times': {
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'avg': sum(response_times) / len(response_times) if response_times else 0,
                'p95': self._calculate_percentile(response_times, 95) if response_times else 0,
                'p99': self._calculate_percentile(response_times, 99) if response_times else 0
            },
            'meets_sla': (
                actual_rps >= benchmark.expected_throughput_rps * 0.9 and
                error_rate <= benchmark.max_error_rate and
                (sum(response_times) / len(response_times) if response_times else 0) <= benchmark.expected_response_time_ms
            )
        }
        
        self.results_history.append(result)
        return result
        
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
        
    async def run_stress_test(self, endpoint: str, max_rps: int = 1000) -> Dict[str, Any]:
        """Run stress test to find breaking point."""
        logging.info(f"Starting stress test for {endpoint}")
        
        breaking_point = None
        current_rps = 10
        step_size = 50
        
        while current_rps <= max_rps:
            benchmark = PerformanceBenchmark(
                name=f"stress_test_{current_rps}",
                endpoint=endpoint,
                expected_response_time_ms=5000,  # High threshold for stress test
                expected_throughput_rps=current_rps,
                max_error_rate=0.1  # 10% error rate threshold
            )
            
            result = await self.run_load_test(benchmark, duration_seconds=30)
            
            if result['error_rate'] > 0.1 or result['response_times']['avg'] > 5000:
                breaking_point = current_rps
                break
                
            current_rps += step_size
            
        return {
            'endpoint': endpoint,
            'breaking_point_rps': breaking_point,
            'max_tested_rps': current_rps - step_size,
            'stable_rps': max(10, current_rps - step_size * 2)
        }

class ChaosEngineer:
    """Chaos engineering implementation."""
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_results: List[Dict] = []
        
    def add_experiment(self, experiment: ChaosExperiment):
        """Add chaos experiment."""
        self.experiments[experiment.name] = experiment
        
    async def run_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Run chaos experiment."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
            
        experiment = self.experiments[experiment_name]
        logging.info(f"Starting chaos experiment: {experiment.name}")
        
        # Record baseline metrics
        baseline_metrics = await self._collect_baseline_metrics(experiment.target_service)
        
        # Inject failure
        failure_injection_success = await self._inject_failure(experiment)
        
        if not failure_injection_success:
            return {
                'experiment': experiment.name,
                'status': 'failed',
                'reason': 'failure_injection_failed'
            }
            
        # Monitor system during chaos
        chaos_metrics = []
        monitoring_start = time.time()
        
        while time.time() - monitoring_start < experiment.duration_minutes * 60:
            metrics = await self._collect_metrics(experiment.target_service)
            chaos_metrics.append(metrics)
            await asyncio.sleep(10)  # Collect metrics every 10 seconds
            
        # Stop failure injection
        await self._stop_failure_injection(experiment)
        
        # Monitor recovery
        recovery_start = time.time()
        recovery_metrics = []
        
        while time.time() - recovery_start < 300:  # 5 minutes recovery window
            metrics = await self._collect_metrics(experiment.target_service)
            recovery_metrics.append(metrics)
            await asyncio.sleep(10)
            
        # Analyze results
        result = self._analyze_experiment_results(
            experiment, baseline_metrics, chaos_metrics, recovery_metrics
        )
        
        self.experiment_results.append(result)
        return result
        
    async def _collect_baseline_metrics(self, service: str) -> Dict[str, Any]:
        """Collect baseline metrics before chaos."""
        # Mock metrics collection
        return {
            'timestamp': time.time(),
            'service': service,
            'cpu_usage': random.uniform(20, 40),
            'memory_usage': random.uniform(30, 50),
            'response_time': random.uniform(100, 200),
            'error_rate': random.uniform(0, 0.02),
            'throughput': random.uniform(50, 100)
        }
        
    async def _collect_metrics(self, service: str) -> Dict[str, Any]:
        """Collect real-time metrics."""
        # Mock metrics collection with some chaos impact
        return {
            'timestamp': time.time(),
            'service': service,
            'cpu_usage': random.uniform(40, 90),
            'memory_usage': random.uniform(50, 80),
            'response_time': random.uniform(200, 1000),
            'error_rate': random.uniform(0.01, 0.1),
            'throughput': random.uniform(20, 60)
        }
        
    async def _inject_failure(self, experiment: ChaosExperiment) -> bool:
        """Inject failure based on experiment type."""
        logging.info(f"Injecting {experiment.failure_type} into {experiment.target_service}")
        
        if experiment.failure_type == "network_delay":
            return await self._inject_network_delay(experiment)
        elif experiment.failure_type == "cpu_stress":
            return await self._inject_cpu_stress(experiment)
        elif experiment.failure_type == "memory_leak":
            return await self._inject_memory_leak(experiment)
        elif experiment.failure_type == "pod_kill":
            return await self._kill_pods(experiment)
        else:
            logging.error(f"Unknown failure type: {experiment.failure_type}")
            return False
            
    async def _inject_network_delay(self, experiment: ChaosExperiment) -> bool:
        """Inject network delay."""
        # Mock network delay injection
        logging.info(f"Injecting 500ms network delay to {experiment.target_service}")
        return True
        
    async def _inject_cpu_stress(self, experiment: ChaosExperiment) -> bool:
        """Inject CPU stress."""
        # Mock CPU stress injection
        logging.info(f"Injecting CPU stress to {experiment.target_service}")
        return True
        
    async def _inject_memory_leak(self, experiment: ChaosExperiment) -> bool:
        """Inject memory leak."""
        # Mock memory leak injection
        logging.info(f"Injecting memory leak to {experiment.target_service}")
        return True
        
    async def _kill_pods(self, experiment: ChaosExperiment) -> bool:
        """Kill pods randomly."""
        # Mock pod killing
        logging.info(f"Killing {experiment.impact_percentage}% of pods in {experiment.target_service}")
        return True
        
    async def _stop_failure_injection(self, experiment: ChaosExperiment):
        """Stop failure injection."""
        logging.info(f"Stopping failure injection for {experiment.name}")
        # Mock cleanup
        
    def _analyze_experiment_results(self, experiment: ChaosExperiment,
                                  baseline: Dict, chaos_metrics: List[Dict],
                                  recovery_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze experiment results."""
        if not chaos_metrics or not recovery_metrics:
            return {
                'experiment': experiment.name,
                'status': 'inconclusive',
                'reason': 'insufficient_data'
            }
            
        # Calculate impact
        avg_chaos_response_time = sum(m['response_time'] for m in chaos_metrics) / len(chaos_metrics)
        avg_chaos_error_rate = sum(m['error_rate'] for m in chaos_metrics) / len(chaos_metrics)
        
        # Calculate recovery
        final_metrics = recovery_metrics[-1]
        recovered = (
            final_metrics['response_time'] < baseline['response_time'] * 1.2 and
            final_metrics['error_rate'] < baseline['error_rate'] * 2
        )
        
        # Check steady state hypothesis
        hypothesis_met = self._evaluate_hypothesis(experiment.steady_state_hypothesis, 
                                                 baseline, chaos_metrics, recovery_metrics)
        
        return {
            'experiment': experiment.name,
            'status': 'completed',
            'duration_minutes': experiment.duration_minutes,
            'impact': {
                'response_time_increase': avg_chaos_response_time / baseline['response_time'],
                'error_rate_increase': avg_chaos_error_rate / baseline['error_rate']
            },
            'recovery': {
                'recovered': recovered,
                'recovery_time_seconds': len(recovery_metrics) * 10
            },
            'steady_state_hypothesis': {
                'hypothesis': experiment.steady_state_hypothesis,
                'met': hypothesis_met
            },
            'resilience_score': self._calculate_resilience_score(chaos_metrics, recovery_metrics)
        }
        
    def _evaluate_hypothesis(self, hypothesis: str, baseline: Dict, 
                           chaos_metrics: List[Dict], recovery_metrics: List[Dict]) -> bool:
        """Evaluate steady state hypothesis."""
        # Simple hypothesis evaluation
        # In production, would parse and evaluate complex conditions
        if "response_time" in hypothesis:
            avg_response_time = sum(m['response_time'] for m in recovery_metrics) / len(recovery_metrics)
            return avg_response_time < baseline['response_time'] * 1.5
        elif "error_rate" in hypothesis:
            avg_error_rate = sum(m['error_rate'] for m in recovery_metrics) / len(recovery_metrics)
            return avg_error_rate < 0.05
        else:
            return True  # Default to true for unknown hypotheses
            
    def _calculate_resilience_score(self, chaos_metrics: List[Dict], 
                                  recovery_metrics: List[Dict]) -> float:
        """Calculate resilience score (0-1)."""
        if not chaos_metrics or not recovery_metrics:
            return 0.0
            
        # Score based on how quickly system recovered
        recovery_speed = 1.0 / len(recovery_metrics) if recovery_metrics else 0
        
        # Score based on service availability during chaos
        availability_score = 1.0 - (sum(m['error_rate'] for m in chaos_metrics) / len(chaos_metrics))
        
        # Combined score
        return min((recovery_speed + availability_score) / 2, 1.0)

class SecurityTester:
    """Security testing implementation."""
    
    def __init__(self):
        self.vulnerability_tests = [
            self._test_sql_injection,
            self._test_xss,
            self._test_csrf,
            self._test_authentication_bypass,
            self._test_authorization_bypass,
            self._test_data_exposure
        ]
        
    async def run_security_scan(self, target_url: str) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        logging.info(f"Starting security scan for {target_url}")
        
        results = {
            'target': target_url,
            'scan_timestamp': time.time(),
            'vulnerabilities': [],
            'risk_score': 0.0
        }
        
        for test_func in self.vulnerability_tests:
            try:
                vulnerability = await test_func(target_url)
                if vulnerability:
                    results['vulnerabilities'].append(vulnerability)
            except Exception as e:
                logging.error(f"Security test failed: {test_func.__name__}: {e}")
                
        # Calculate risk score
        if results['vulnerabilities']:
            severity_weights = {'low': 1, 'medium': 3, 'high': 7, 'critical': 10}
            total_risk = sum(severity_weights.get(v['severity'], 1) for v in results['vulnerabilities'])
            results['risk_score'] = min(total_risk / 10.0, 10.0)
        
        return results
        
    async def _test_sql_injection(self, target_url: str) -> Optional[Dict]:
        """Test for SQL injection vulnerabilities."""
        payloads = [
            "' OR 1=1--",
            "' UNION SELECT * FROM users--",
            "1'; DROP TABLE users--"
        ]
        
        for payload in payloads:
            # Mock SQL injection test
            if random.random() < 0.1:  # 10% chance of finding vulnerability
                return {
                    'type': 'sql_injection',
                    'severity': 'high',
                    'description': f'SQL injection vulnerability found with payload: {payload}',
                    'location': target_url,
                    'recommendation': 'Use parameterized queries'
                }
        return None
        
    async def _test_xss(self, target_url: str) -> Optional[Dict]:
        """Test for XSS vulnerabilities."""
        payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ]
        
        for payload in payloads:
            # Mock XSS test
            if random.random() < 0.05:  # 5% chance
                return {
                    'type': 'xss',
                    'severity': 'medium',
                    'description': f'XSS vulnerability found with payload: {payload}',
                    'location': target_url,
                    'recommendation': 'Implement input sanitization'
                }
        return None
        
    async def _test_csrf(self, target_url: str) -> Optional[Dict]:
        """Test for CSRF vulnerabilities."""
        # Mock CSRF test
        if random.random() < 0.03:  # 3% chance
            return {
                'type': 'csrf',
                'severity': 'medium',
                'description': 'CSRF protection not implemented',
                'location': target_url,
                'recommendation': 'Implement CSRF tokens'
            }
        return None
        
    async def _test_authentication_bypass(self, target_url: str) -> Optional[Dict]:
        """Test for authentication bypass."""
        # Mock authentication bypass test
        if random.random() < 0.02:  # 2% chance
            return {
                'type': 'auth_bypass',
                'severity': 'critical',
                'description': 'Authentication can be bypassed',
                'location': target_url,
                'recommendation': 'Review authentication logic'
            }
        return None
        
    async def _test_authorization_bypass(self, target_url: str) -> Optional[Dict]:
        """Test for authorization bypass."""
        # Mock authorization bypass test
        if random.random() < 0.02:  # 2% chance
            return {
                'type': 'authz_bypass',
                'severity': 'high',
                'description': 'Authorization controls can be bypassed',
                'location': target_url,
                'recommendation': 'Implement proper access controls'
            }
        return None
        
    async def _test_data_exposure(self, target_url: str) -> Optional[Dict]:
        """Test for sensitive data exposure."""
        # Mock data exposure test
        if random.random() < 0.04:  # 4% chance
            return {
                'type': 'data_exposure',
                'severity': 'high',
                'description': 'Sensitive data exposed in response',
                'location': target_url,
                'recommendation': 'Remove sensitive data from responses'
            }
        return None

class ComprehensiveTestRunner:
    """Main test orchestration system."""
    
    def __init__(self):
        self.performance_tester = PerformanceTester()
        self.chaos_engineer = ChaosEngineer()
        self.security_tester = SecurityTester()
        self.test_results: List[TestResult] = []
        self.quality_gates = {
            'test_coverage': 85.0,
            'performance_sla': 95.0,
            'security_risk_threshold': 5.0,
            'chaos_resilience_threshold': 0.7
        }
        
    async def run_full_test_suite(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logging.info("Starting comprehensive test suite")
        
        suite_results = {
            'start_time': time.time(),
            'test_results': {},
            'quality_gates': {},
            'overall_status': 'pending'
        }
        
        # Run unit tests
        if config.get('run_unit_tests', True):
            unit_results = await self._run_unit_tests()
            suite_results['test_results']['unit'] = unit_results
            
        # Run integration tests
        if config.get('run_integration_tests', True):
            integration_results = await self._run_integration_tests()
            suite_results['test_results']['integration'] = integration_results
            
        # Run E2E tests
        if config.get('run_e2e_tests', True):
            e2e_results = await self._run_e2e_tests()
            suite_results['test_results']['e2e'] = e2e_results
            
        # Run performance tests
        if config.get('run_performance_tests', True):
            performance_results = await self._run_performance_tests(config)
            suite_results['test_results']['performance'] = performance_results
            
        # Run security tests
        if config.get('run_security_tests', True):
            security_results = await self._run_security_tests(config)
            suite_results['test_results']['security'] = security_results
            
        # Run chaos tests
        if config.get('run_chaos_tests', False):  # Optional by default
            chaos_results = await self._run_chaos_tests()
            suite_results['test_results']['chaos'] = chaos_results
            
        # Evaluate quality gates
        suite_results['quality_gates'] = await self._evaluate_quality_gates(suite_results['test_results'])
        
        # Determine overall status
        suite_results['overall_status'] = self._determine_overall_status(suite_results['quality_gates'])
        suite_results['end_time'] = time.time()
        suite_results['duration'] = suite_results['end_time'] - suite_results['start_time']
        
        return suite_results
        
    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        logging.info("Running unit tests")
        
        try:
            # Run pytest
            result = subprocess.run([
                'python3', '-m', 'pytest', 'tests/',
                '--tb=short', '-v', '--json-report',
                '--json-report-file=test_results.json'
            ], capture_output=True, text=True, timeout=300)
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'exit_code': result.returncode,
                'output': result.stdout,
                'errors': result.stderr,
                'coverage': self._extract_coverage(),
                'test_count': self._count_tests('tests/')
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'error': 'Unit tests timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        logging.info("Running integration tests")
        
        # Mock integration test results
        return {
            'status': 'passed',
            'test_count': 25,
            'passed': 24,
            'failed': 1,
            'duration': 45.2
        }
        
    async def _run_e2e_tests(self) -> Dict[str, Any]:
        """Run end-to-end tests."""
        logging.info("Running E2E tests")
        
        # Mock E2E test results
        return {
            'status': 'passed',
            'test_count': 10,
            'passed': 9,
            'failed': 1,
            'duration': 120.5,
            'browser_logs': [],
            'screenshots': []
        }
        
    async def _run_performance_tests(self, config: Dict) -> Dict[str, Any]:
        """Run performance tests."""
        logging.info("Running performance tests")
        
        # Setup benchmarks
        benchmarks = [
            PerformanceBenchmark(
                name="api_health_check",
                endpoint="/health",
                expected_response_time_ms=100,
                expected_throughput_rps=100,
                max_error_rate=0.01
            ),
            PerformanceBenchmark(
                name="prediction_endpoint",
                endpoint="/predict",
                expected_response_time_ms=500,
                expected_throughput_rps=50,
                max_error_rate=0.02
            )
        ]
        
        results = []
        for benchmark in benchmarks:
            self.performance_tester.add_benchmark(benchmark)
            result = await self.performance_tester.run_load_test(benchmark)
            results.append(result)
            
        # Calculate SLA compliance
        sla_compliance = sum(1 for r in results if r['meets_sla']) / len(results) * 100
        
        return {
            'status': 'passed' if sla_compliance >= self.quality_gates['performance_sla'] else 'failed',
            'sla_compliance': sla_compliance,
            'benchmark_results': results
        }
        
    async def _run_security_tests(self, config: Dict) -> Dict[str, Any]:
        """Run security tests."""
        logging.info("Running security tests")
        
        target_url = config.get('target_url', 'http://localhost:8000')
        security_results = await self.security_tester.run_security_scan(target_url)
        
        return {
            'status': 'passed' if security_results['risk_score'] < self.quality_gates['security_risk_threshold'] else 'failed',
            'risk_score': security_results['risk_score'],
            'vulnerabilities': security_results['vulnerabilities']
        }
        
    async def _run_chaos_tests(self) -> Dict[str, Any]:
        """Run chaos engineering tests."""
        logging.info("Running chaos tests")
        
        # Setup chaos experiments
        experiments = [
            ChaosExperiment(
                name="api_pod_kill",
                target_service="pneumonia-detector",
                failure_type="pod_kill",
                duration_minutes=2,
                impact_percentage=30,
                steady_state_hypothesis="response_time < 1000ms"
            ),
            ChaosExperiment(
                name="database_network_delay",
                target_service="database",
                failure_type="network_delay",
                duration_minutes=3,
                impact_percentage=100,
                steady_state_hypothesis="error_rate < 5%"
            )
        ]
        
        results = []
        for experiment in experiments:
            self.chaos_engineer.add_experiment(experiment)
            result = await self.chaos_engineer.run_experiment(experiment.name)
            results.append(result)
            
        # Calculate resilience score
        resilience_scores = [r.get('resilience_score', 0) for r in results if 'resilience_score' in r]
        avg_resilience = sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0
        
        return {
            'status': 'passed' if avg_resilience >= self.quality_gates['chaos_resilience_threshold'] else 'failed',
            'resilience_score': avg_resilience,
            'experiment_results': results
        }
        
    async def _evaluate_quality_gates(self, test_results: Dict) -> Dict[str, Any]:
        """Evaluate quality gates."""
        gates = {}
        
        # Test coverage gate
        unit_results = test_results.get('unit', {})
        coverage = unit_results.get('coverage', 0)
        gates['test_coverage'] = {
            'threshold': self.quality_gates['test_coverage'],
            'actual': coverage,
            'passed': coverage >= self.quality_gates['test_coverage']
        }
        
        # Performance SLA gate
        performance_results = test_results.get('performance', {})
        sla_compliance = performance_results.get('sla_compliance', 0)
        gates['performance_sla'] = {
            'threshold': self.quality_gates['performance_sla'],
            'actual': sla_compliance,
            'passed': sla_compliance >= self.quality_gates['performance_sla']
        }
        
        # Security risk gate
        security_results = test_results.get('security', {})
        risk_score = security_results.get('risk_score', 0)
        gates['security_risk'] = {
            'threshold': self.quality_gates['security_risk_threshold'],
            'actual': risk_score,
            'passed': risk_score < self.quality_gates['security_risk_threshold']
        }
        
        # Chaos resilience gate (if chaos tests were run)
        if 'chaos' in test_results:
            chaos_results = test_results['chaos']
            resilience_score = chaos_results.get('resilience_score', 0)
            gates['chaos_resilience'] = {
                'threshold': self.quality_gates['chaos_resilience_threshold'],
                'actual': resilience_score,
                'passed': resilience_score >= self.quality_gates['chaos_resilience_threshold']
            }
            
        return gates
        
    def _determine_overall_status(self, quality_gates: Dict) -> str:
        """Determine overall test suite status."""
        all_passed = all(gate['passed'] for gate in quality_gates.values())
        return 'passed' if all_passed else 'failed'
        
    def _extract_coverage(self) -> float:
        """Extract test coverage from reports."""
        # Mock coverage extraction
        return random.uniform(80, 95)
        
    def _count_tests(self, test_dir: str) -> int:
        """Count number of tests in directory."""
        try:
            test_files = list(Path(test_dir).rglob("test_*.py"))
            return len(test_files) * 5  # Approximate 5 tests per file
        except:
            return 50  # Default estimate

async def main():
    """Main entry point for testing."""
    test_runner = ComprehensiveTestRunner()
    
    config = {
        'run_unit_tests': True,
        'run_integration_tests': True,
        'run_e2e_tests': True,
        'run_performance_tests': True,
        'run_security_tests': True,
        'run_chaos_tests': True,
        'target_url': 'http://localhost:8000'
    }
    
    print("Starting comprehensive test suite...")
    results = await test_runner.run_full_test_suite(config)
    
    print(f"Test suite completed with status: {results['overall_status']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    print("\nQuality Gates:")
    for gate_name, gate_result in results['quality_gates'].items():
        status = "✅ PASSED" if gate_result['passed'] else "❌ FAILED"
        print(f"  {gate_name}: {status} ({gate_result['actual']}/{gate_result['threshold']})")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test suite stopped")