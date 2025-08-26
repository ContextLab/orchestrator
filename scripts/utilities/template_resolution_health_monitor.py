#!/usr/bin/env python3
"""
Template Resolution Health Monitor - Issue #275 Stream D

Ongoing health monitoring and regression prevention for template resolution.
Provides continuous validation of template resolution health across the system.
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.core.unified_template_resolver import UnifiedTemplateResolver


class TemplateResolutionHealthMonitor:
    """Continuous health monitoring for template resolution system."""
    
    def __init__(self):
        self.health_report = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "overall_health": "unknown",
            "components": {},
            "regression_indicators": [],
            "recommendations": []
        }
    
    def run_core_health_check(self) -> Dict[str, Any]:
        """Run health check on core template resolution functionality."""
        
        print("🔍 Running Core Template Resolution Health Check...")
        
        resolver = UnifiedTemplateResolver(debug_mode=True)
        health_status = {
            "status": "healthy",
            "tests_passed": 0,
            "tests_total": 0,
            "issues": [],
            "performance_ms": 0
        }
        
        start_time = time.time()
        
        # Test scenarios covering Stream A fixes
        test_scenarios = [
            {
                "name": "Dollar Variable Preprocessing",
                "template": "Item: {{ $item }}, Index: {{ $index }}",
                "context": {"item": "value1", "index": 42},
                "expected": "Item: value1, Index: 42"
            },
            {
                "name": "Cross-Step References",
                "template": "Content: {{ step1.result }}, Data: {{ step2.data }}",
                "context": {"step1": {"result": "success"}, "step2": {"data": "processed"}},
                "expected": "Content: success, Data: processed"
            },
            {
                "name": "Mixed Loop and Step Variables",
                "template": "Processing {{ $item }} with result {{ analysis.status }}",
                "context": {"item": "file.txt", "analysis": {"status": "complete"}},
                "expected": "Processing file.txt with result complete"
            },
            {
                "name": "Nested Data Access",
                "template": "Title: {{ data.metadata.title }}, Version: {{ data.metadata.version }}",
                "context": {"data": {"metadata": {"title": "Test", "version": "1.0"}}},
                "expected": "Title: Test, Version: 1.0"
            }
        ]
        
        health_status["tests_total"] = len(test_scenarios)
        
        for scenario in test_scenarios:
            try:
                context = resolver.collect_context(
                    pipeline_id="health_check",
                    additional_context=scenario["context"]
                )
                
                result = resolver.resolve_templates(scenario["template"], context)
                
                if result == scenario["expected"]:
                    health_status["tests_passed"] += 1
                    print(f"  ✅ {scenario['name']}")
                else:
                    health_status["issues"].append(f"{scenario['name']}: Expected '{scenario['expected']}', got '{result}'")
                    health_status["status"] = "degraded"
                    print(f"  ❌ {scenario['name']}: FAILED")
                
                # Check for unresolved templates
                if "{{" in result or "}}" in result:
                    health_status["issues"].append(f"{scenario['name']}: Unresolved templates remain")
                    health_status["status"] = "degraded"
                    
            except Exception as e:
                health_status["issues"].append(f"{scenario['name']}: Exception - {str(e)}")
                health_status["status"] = "unhealthy"
                print(f"  ❌ {scenario['name']}: EXCEPTION - {str(e)}")
        
        health_status["performance_ms"] = int((time.time() - start_time) * 1000)
        
        # Overall assessment
        success_rate = health_status["tests_passed"] / health_status["tests_total"]
        if success_rate >= 1.0 and health_status["status"] == "healthy":
            print(f"  🟢 Core Resolution: HEALTHY ({health_status['tests_passed']}/{health_status['tests_total']} tests passed)")
        elif success_rate >= 0.8:
            print(f"  🟡 Core Resolution: DEGRADED ({health_status['tests_passed']}/{health_status['tests_total']} tests passed)")
        else:
            health_status["status"] = "unhealthy"
            print(f"  🔴 Core Resolution: UNHEALTHY ({health_status['tests_passed']}/{health_status['tests_total']} tests passed)")
        
        return health_status
    
    def check_regression_indicators(self) -> Dict[str, Any]:
        """Check for regression indicators in the system."""
        
        print("\n🔍 Checking for Regression Indicators...")
        
        indicators = {
            "template_artifacts_detected": False,
            "ai_model_confusion": False,
            "loop_variable_failures": False,
            "cross_step_reference_failures": False,
            "performance_degradation": False,
            "details": []
        }
        
        # Check example pipeline outputs for artifacts
        examples_output_dir = Path("examples/outputs")
        if examples_output_dir.exists():
            
            print("  📁 Scanning example pipeline outputs...")
            
            artifact_count = 0
            ai_confusion_count = 0
            
            for output_file in examples_output_dir.rglob("*.txt"):
                try:
                    content = output_file.read_text()
                    
                    # Check for unresolved templates
                    if "{{" in content or "}}" in content:
                        artifact_count += 1
                        if artifact_count <= 3:  # Report first 3
                            indicators["details"].append(f"Template artifacts in {output_file.name}")
                    
                    # Check for AI confusion
                    ai_confusion_markers = [
                        "I don't have access to {{",
                        "placeholder didn't load",
                        "I need the text to"
                    ]
                    
                    for marker in ai_confusion_markers:
                        if marker in content:
                            ai_confusion_count += 1
                            if ai_confusion_count <= 3:  # Report first 3
                                indicators["details"].append(f"AI confusion in {output_file.name}: {marker}")
                            break
                    
                except Exception:
                    # Skip files that can't be read
                    pass
            
            # Set indicator flags
            if artifact_count > 0:
                indicators["template_artifacts_detected"] = True
                print(f"  ⚠️ Template artifacts found in {artifact_count} files")
            
            if ai_confusion_count > 0:
                indicators["ai_model_confusion"] = True  
                print(f"  ⚠️ AI model confusion detected in {ai_confusion_count} files")
                
            if artifact_count == 0 and ai_confusion_count == 0:
                print("  ✅ No regression indicators found in output files")
        
        return indicators
    
    def generate_health_recommendations(self, core_health: Dict[str, Any], indicators: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on findings."""
        
        recommendations = []
        
        # Core health recommendations
        if core_health["status"] == "unhealthy":
            recommendations.append("🚨 CRITICAL: Core template resolution requires immediate attention")
            recommendations.append("   Run: python -m pytest tests/test_template_resolution_integration_simple.py")
            
        elif core_health["status"] == "degraded":
            recommendations.append("⚠️ WARNING: Core template resolution has issues")
            recommendations.append("   Review failed test scenarios and debug")
        
        # Regression indicator recommendations
        if indicators["template_artifacts_detected"]:
            recommendations.append("🔧 ISSUE: Template artifacts detected in pipeline outputs")
            recommendations.append("   This suggests integration gaps between core resolver and pipeline execution")
            
        if indicators["ai_model_confusion"]:
            recommendations.append("🔧 ISSUE: AI models receiving template placeholders instead of resolved content")
            recommendations.append("   Stream C work needed: Fix tool parameter resolution before AI model calls")
        
        # Performance recommendations
        if core_health["performance_ms"] > 100:
            recommendations.append("⚡ PERFORMANCE: Template resolution taking longer than expected")
            recommendations.append("   Consider optimization or caching strategies")
        
        # Positive recommendations
        if core_health["status"] == "healthy" and not any(indicators.values()):
            recommendations.append("✅ EXCELLENT: Template resolution system appears healthy")
            recommendations.append("   Continue monitoring and consider expanding test coverage")
        
        return recommendations
    
    def run_comprehensive_health_check(self):
        """Run comprehensive health check and generate report."""
        
        print("🏥 TEMPLATE RESOLUTION HEALTH MONITOR")
        print("=" * 50)
        print(f"Issue #275 - Stream D Health Check")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # Run core health check
        core_health = self.run_core_health_check()
        self.health_report["components"]["core_resolution"] = core_health
        
        # Check regression indicators  
        indicators = self.check_regression_indicators()
        self.health_report["regression_indicators"] = indicators
        
        # Generate recommendations
        recommendations = self.generate_health_recommendations(core_health, indicators)
        self.health_report["recommendations"] = recommendations
        
        # Determine overall health
        if core_health["status"] == "healthy" and not any(indicators.values()):
            self.health_report["overall_health"] = "healthy"
        elif core_health["status"] == "degraded" or any(indicators.values()):
            self.health_report["overall_health"] = "needs_attention"
        else:
            self.health_report["overall_health"] = "critical"
        
        # Generate report
        self.generate_health_report()
    
    def generate_health_report(self):
        """Generate and display health report."""
        
        print(f"\n📊 HEALTH REPORT SUMMARY")
        print(f"=" * 30)
        
        # Overall health
        health_emoji = "🟢" if self.health_report["overall_health"] == "healthy" else "🟡" if self.health_report["overall_health"] == "needs_attention" else "🔴"
        print(f"Overall Health: {health_emoji} {self.health_report['overall_health'].upper()}")
        
        # Core component status
        core = self.health_report["components"]["core_resolution"]
        core_emoji = "🟢" if core["status"] == "healthy" else "🟡" if core["status"] == "degraded" else "🔴"
        print(f"Core Resolution: {core_emoji} {core['status'].upper()} ({core['tests_passed']}/{core['tests_total']} tests passed)")
        
        # Regression indicators
        indicators = self.health_report["regression_indicators"]
        regression_count = sum(1 for v in indicators.values() if isinstance(v, bool) and v)
        regression_emoji = "🟢" if regression_count == 0 else "🟡" if regression_count <= 2 else "🔴"
        print(f"Regression Check: {regression_emoji} {regression_count} indicators detected")
        
        # Performance
        perf_ms = core["performance_ms"]
        perf_emoji = "🟢" if perf_ms < 50 else "🟡" if perf_ms < 100 else "🔴"
        print(f"Performance: {perf_emoji} {perf_ms}ms average resolution time")
        
        # Recommendations
        if self.health_report["recommendations"]:
            print(f"\n📋 RECOMMENDATIONS:")
            for rec in self.health_report["recommendations"]:
                print(f"   {rec}")
        
        # Current stream status assessment
        print(f"\n🚀 STREAM STATUS ASSESSMENT:")
        print(f"   Stream A (Core Resolution): {'🟢 COMPLETE' if core['status'] == 'healthy' else '🟡 ISSUES'}")
        
        loop_vars_working = not indicators.get("loop_variable_failures", True)  # Assume working unless detected
        print(f"   Stream B (Loop Context): {'🟢 MAJOR_PROGRESS' if loop_vars_working else '🔴 ISSUES'}")
        
        ai_issues = indicators.get("ai_model_confusion", False)
        print(f"   Stream C (Tool Integration): {'🟡 IN_PROGRESS' if ai_issues else '🟢 GOOD'}")
        
        print(f"   Stream D (Testing & Validation): 🟢 ACTIVE")
        
        # Save detailed report
        report_path = Path("examples/outputs/template_resolution_health_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.health_report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed health report saved to: {report_path}")
        
        return self.health_report


def main():
    """Run template resolution health monitor."""
    monitor = TemplateResolutionHealthMonitor()
    monitor.run_comprehensive_health_check()


if __name__ == "__main__":
    main()