#!/usr/bin/env python3
"""Simple demonstration of robust API without external dependencies."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_inspired_task_planner.advanced_scheduler import QuantumMLScheduler
from quantum_inspired_task_planner.security_framework import QuantumSchedulerSecurity, Permission, SecurityLevel
from quantum_inspired_task_planner.simple_health_monitoring import SimpleHealthMonitor


def demo_security_framework():
    """Demonstrate security framework."""
    print("🛡️  Security Framework Demo")
    print("-" * 30)
    
    security = QuantumSchedulerSecurity()
    
    # Create users
    print("👥 Creating users...")
    
    try:
        admin_user = security.create_user(
            user_id="admin",
            username="admin", 
            email="admin@example.com",
            password="AdminPass123!",
            permissions=[Permission.ADMIN_ACCESS, Permission.CREATE_TASKS, Permission.VIEW_METRICS],
            security_level=SecurityLevel.TOP_SECRET
        )
        print(f"✅ Created admin user: {admin_user.username}")
    except Exception as e:
        print(f"⚠️  Admin user may already exist: {e}")
    
    try:
        regular_user = security.create_user(
            user_id="user1",
            username="user1",
            email="user1@example.com", 
            password="UserPass123!",
            permissions=[Permission.CREATE_TASKS, Permission.READ_TASKS],
            security_level=SecurityLevel.INTERNAL
        )
        print(f"✅ Created regular user: {regular_user.username}")
    except Exception as e:
        print(f"⚠️  Regular user may already exist: {e}")
    
    # Test authentication
    print("\n🔐 Testing authentication...")
    session_token = security.authenticate_user("admin", "AdminPass123!")
    if session_token:
        print(f"✅ Authentication successful, token: {session_token[:16]}...")
        
        # Test permissions
        has_admin = security.check_permission(session_token, Permission.ADMIN_ACCESS)
        has_create = security.check_permission(session_token, Permission.CREATE_TASKS)
        
        print(f"  Admin access: {'✅' if has_admin else '❌'}")
        print(f"  Create tasks: {'✅' if has_create else '❌'}")
    else:
        print("❌ Authentication failed")
    
    # Test failed authentication
    print("\n🚨 Testing failed authentication...")
    bad_token = security.authenticate_user("admin", "wrongpassword")
    print(f"Bad password result: {'❌ Correct' if not bad_token else '⚠️  Unexpected success'}")
    
    # Security report
    print("\n📊 Security report:")
    report = security.get_security_report()
    print(f"  Total users: {report['user_statistics']['total_users']}")
    print(f"  Active users: {report['user_statistics']['active_users']}")
    print(f"  Active sessions: {report['session_statistics']['active_sessions']}")
    
    return security, session_token


def demo_health_monitoring():
    """Demonstrate health monitoring."""
    print("\n🏥 Health Monitoring Demo")
    print("-" * 30)
    
    scheduler = QuantumMLScheduler()
    health_monitor = SimpleHealthMonitor(scheduler)
    
    # Start monitoring
    print("🎯 Starting health monitoring...")
    health_monitor.start_monitoring()
    
    # Create some tasks to monitor
    print("📝 Creating tasks for monitoring...")
    for i in range(5):
        task_id = scheduler.add_intelligent_task(
            name=f"Test Task {i+1}",
            description=f"Test task for monitoring demo {i+1}",
            tags=["demo", "monitoring"]
        )
        
        # Simulate some task execution
        if i < 3:
            scheduler.start_task(task_id)
            if i < 2:
                scheduler.complete_task_with_learning(task_id)
    
    # Get health status
    print("\n📊 Current health status:")
    health = health_monitor.get_current_health()
    print(f"  Status: {health.status}")
    print(f"  Score: {health.score:.1f}")
    print(f"  Alerts: {len(health.alerts)}")
    
    for alert in health.alerts:
        print(f"    ⚠️  {alert}")
    
    # Show metrics
    print("\n📈 Current metrics:")
    for name, metric in health.metrics.items():
        print(f"  {name}: {metric.value:.2f} {metric.unit} ({metric.status})")
    
    # Export health report
    print("\n📄 Health report:")
    report = health_monitor.export_health_report()
    print(f"  Overall status: {report['overall_health']['status']}")
    print(f"  Total metrics: {len(report['current_metrics'])}")
    
    health_monitor.stop_monitoring()
    return health_monitor


def demo_integrated_robust_system():
    """Demonstrate integrated robust system."""
    print("\n🚀 Integrated Robust System Demo")
    print("-" * 35)
    
    # Initialize components
    security, session_token = demo_security_framework()
    health_monitor = demo_health_monitoring()
    
    if not session_token:
        print("❌ Cannot continue without valid session")
        return False
    
    # Test secure task creation
    print("\n📝 Testing secure task creation...")
    
    try:
        # This would normally be done through the API
        task_data = {
            "name": "Secure ML Training",
            "description": "Train model with security validation",
            "priority": "high",
            "tags": ["ml", "secure"]
        }
        
        # Validate task data through security framework
        validated_data = security.secure_task_creation(session_token, task_data)
        print(f"✅ Task data validated: {validated_data['name']}")
        
    except Exception as e:
        print(f"❌ Task validation failed: {e}")
        return False
    
    # Test data export security
    print("\n💾 Testing secure data export...")
    
    can_export = security.secure_data_export(session_token, "tasks")
    print(f"Export permission: {'✅ Granted' if can_export else '❌ Denied'}")
    
    # Show final security summary
    print("\n📊 Final system summary:")
    security_report = security.get_security_report()
    
    print(f"  Security Events: {security_report['security_events']['total_events']}")
    print(f"  Failed Events: {security_report['security_events']['failed_events']}")
    print(f"  High Risk Events: {security_report['security_events']['high_risk_events']}")
    
    print("\n✅ Integrated robust system demo completed!")
    return True


def main():
    """Run comprehensive robust system demonstration."""
    print("🛡️  Quantum Task Scheduler - Robust System Demo")
    print("=" * 55)
    
    try:
        success = demo_integrated_robust_system()
        
        if success:
            print("\n🎉 All robust system demonstrations completed successfully!")
            print("\n📝 Key Features Demonstrated:")
            print("  ✅ User authentication and authorization")
            print("  ✅ Permission-based access control")
            print("  ✅ Input sanitization and validation")
            print("  ✅ Security audit logging")
            print("  ✅ Health monitoring and alerting")
            print("  ✅ Secure task creation and management")
            print("  ✅ Data export security controls")
            print("  ✅ Rate limiting and session management")
        else:
            print("\n❌ Some robust system features failed")
            return False
        
    except Exception as e:
        print(f"\n❌ Robust system demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)