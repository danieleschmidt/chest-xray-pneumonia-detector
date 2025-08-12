#!/usr/bin/env python3
"""
Test script for new Generation 2 and 3 features.
Verifies advanced functionality works correctly.
"""

import sys
import traceback
from pathlib import Path

def test_advanced_input_sanitizer():
    """Test advanced input sanitizer."""
    try:
        from src.security.advanced_input_sanitizer import create_sanitizer
        
        sanitizer = create_sanitizer()
        test_params = {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'model_name': 'ResNet50',
            'img_size': (224, 224)
        }
        
        result = sanitizer.sanitize_training_parameters(test_params)
        assert result, "Sanitizer should return validated parameters"
        assert 'batch_size' in result, "Batch size should be validated"
        
        print("✅ Advanced Input Sanitizer: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Advanced Input Sanitizer: FAILED - {e}")
        return False

def test_comprehensive_health_checks():
    """Test comprehensive health monitoring."""
    try:
        from src.monitoring.comprehensive_health_checks import create_health_monitor
        
        monitor = create_health_monitor()
        results = monitor.run_all_checks()
        
        assert results, "Health checks should return results"
        assert len(results) > 0, "Should have at least one health check"
        
        overall_health = monitor.get_overall_health()
        assert 'overall_status' in overall_health, "Should have overall status"
        
        print("✅ Comprehensive Health Checks: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive Health Checks: FAILED - {e}")
        return False

def test_advanced_error_recovery():
    """Test advanced error recovery system."""
    try:
        from src.error_handling.advanced_error_recovery import create_error_recovery_system
        
        recovery_system = create_error_recovery_system()
        
        # Test error classification
        try:
            raise FileNotFoundError("Test model file not found")
        except Exception as e:
            error_context = recovery_system.classify_error(e)
            assert error_context.error_type == "FileNotFoundError"
            assert "not found" in error_context.error_message.lower()
        
        stats = recovery_system.get_error_statistics()
        assert 'total_errors' in stats, "Should have error statistics"
        
        print("✅ Advanced Error Recovery: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Advanced Error Recovery: FAILED - {e}")
        return False

def test_comprehensive_input_validation():
    """Test comprehensive input validation."""
    try:
        from src.validation.comprehensive_input_validation import create_validator, ValidationLevel, DataType
        
        validator = create_validator(ValidationLevel.STANDARD)
        
        # Test parameter validation
        test_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'optimizer': 'adam'
        }
        
        result = validator.validate(test_params, DataType.MODEL_PARAMETERS)
        assert result.is_valid or len(result.errors) == 0, f"Validation should pass: {result.errors}"
        
        stats = validator.get_validation_statistics()
        assert 'total_validations' in stats, "Should have validation statistics"
        
        print("✅ Comprehensive Input Validation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive Input Validation: FAILED - {e}")
        return False

def test_intelligent_auto_scaler():
    """Test intelligent auto-scaling system."""
    try:
        from src.scaling.intelligent_auto_scaler import create_intelligent_autoscaler
        
        autoscaler = create_intelligent_autoscaler({
            'scaling_interval': 10,
            'min_scaling_interval': 5
        })
        
        status = autoscaler.get_current_status()
        assert 'is_running' in status, "Should have running status"
        assert 'current_resources' in status, "Should have resource information"
        
        # Test force scaling check
        recommendations = autoscaler.force_scaling_check()
        assert isinstance(recommendations, list), "Should return list of recommendations"
        
        print("✅ Intelligent Auto-Scaler: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Intelligent Auto-Scaler: FAILED - {e}")
        return False

def test_adaptive_performance_optimizer():
    """Test adaptive performance optimization."""
    try:
        from src.optimization.adaptive_performance_optimizer import create_performance_optimizer
        
        optimizer = create_performance_optimizer({
            'cache_size': 100,
            'cache_strategy': 'adaptive'
        })
        
        status = optimizer.get_optimization_status()
        assert 'cache_stats' in status, "Should have cache statistics"
        assert 'recent_performance' in status, "Should have performance metrics"
        
        # Test cache functionality
        cache = optimizer.cache
        cache.put('test_key', 'test_value')
        value = cache.get('test_key')
        assert value == 'test_value', "Cache should store and retrieve values"
        
        cache_stats = cache.get_stats()
        assert cache_stats['hit_rate'] > 0, "Should have cache hits"
        
        print("✅ Adaptive Performance Optimizer: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive Performance Optimizer: FAILED - {e}")
        return False

def run_all_tests():
    """Run all feature tests."""
    print("🧪 Testing New Advanced Features...\n")
    
    tests = [
        test_advanced_input_sanitizer,
        test_comprehensive_health_checks,
        test_advanced_error_recovery,
        test_comprehensive_input_validation,
        test_intelligent_auto_scaler,
        test_adaptive_performance_optimizer
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: FAILED - {e}")
            failed += 1
        print()  # Empty line for readability
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All advanced features are working correctly!")
        return True
    else:
        print("⚠️  Some advanced features need attention.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
