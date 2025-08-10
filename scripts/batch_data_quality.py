#!/usr/bin/env python3
"""
Batch Data Quality Processing Script
P2 Enhancement - Automated data quality checks and batch processing
"""

import asyncio
import sys
import os
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.services.data_quality import (
    DataQualityFramework, 
    BatchDataProcessor, 
    QualityLevel,
    ValidationSeverity
)
from backend.app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_quality.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('batch_quality')

class BatchQualityRunner:
    """Main batch quality processing runner"""
    
    def __init__(self):
        self.quality_framework = DataQualityFramework()
        self.batch_processor = BatchDataProcessor()
        self.results = {}
        
    async def run_full_quality_audit(self, datasets: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive quality audit"""
        if datasets is None:
            datasets = ["videos", "channels", "analytics"]
        
        logger.info(f"Starting full quality audit for datasets: {datasets}")
        
        audit_results = {
            "timestamp": datetime.now(),
            "datasets": {},
            "summary": {},
            "alerts": [],
            "recommendations": []
        }
        
        # Process each dataset
        for dataset in datasets:
            try:
                logger.info(f"Processing dataset: {dataset}")
                
                # Generate quality report
                report = await self.quality_framework.generate_quality_report(dataset)
                
                # Store results
                audit_results["datasets"][dataset] = {
                    "overall_score": report.metrics.overall_score,
                    "quality_level": report.metrics.quality_level.value,
                    "total_records": report.metrics.total_records,
                    "issues_count": len(report.issues),
                    "critical_issues": sum(1 for i in report.issues if i.severity == ValidationSeverity.CRITICAL),
                    "error_issues": sum(1 for i in report.issues if i.severity == ValidationSeverity.ERROR),
                    "recommendations": report.recommendations[:5],  # Top 5 recommendations
                    "top_issues": [
                        {
                            "rule": issue.rule_name,
                            "severity": issue.severity.value,
                            "affected_records": issue.affected_records,
                            "description": issue.description
                        }
                        for issue in sorted(report.issues, key=lambda x: x.affected_records, reverse=True)[:3]
                    ]
                }
                
                # Collect critical alerts
                critical_issues = [i for i in report.issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
                for issue in critical_issues:
                    audit_results["alerts"].append({
                        "dataset": dataset,
                        "severity": issue.severity.value,
                        "rule": issue.rule_name,
                        "description": issue.description,
                        "affected_records": issue.affected_records
                    })
                
                logger.info(f"Completed {dataset}: {report.metrics.overall_score:.1f}% quality score")
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset}: {e}")
                audit_results["datasets"][dataset] = {
                    "error": str(e),
                    "overall_score": 0.0,
                    "quality_level": "critical"
                }
        
        # Generate overall summary
        audit_results["summary"] = self._generate_audit_summary(audit_results["datasets"])
        
        # Generate system-wide recommendations
        audit_results["recommendations"] = self._generate_system_recommendations(audit_results)
        
        logger.info("Full quality audit completed")
        return audit_results
    
    async def run_batch_cleanup(self, dataset: str, dry_run: bool = True) -> Dict[str, Any]:
        """Run batch data cleanup operations"""
        logger.info(f"Starting batch cleanup for {dataset} (dry_run: {dry_run})")
        
        try:
            cleanup_results = await self.batch_processor.batch_data_cleanup(dataset)
            
            logger.info(f"Cleanup completed for {dataset}: {cleanup_results}")
            return {
                "success": True,
                "dataset": dataset,
                "dry_run": dry_run,
                "results": cleanup_results,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in batch cleanup for {dataset}: {e}")
            return {
                "success": False,
                "dataset": dataset,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def monitor_real_time_quality(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Monitor data quality in real-time"""
        logger.info(f"Starting real-time quality monitoring for {duration_minutes} minutes")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        monitoring_results = {
            "start_time": start_time,
            "end_time": end_time,
            "snapshots": [],
            "alerts": [],
            "trends": {}
        }
        
        while datetime.now() < end_time:
            try:
                # Take quality snapshot
                snapshot = await self._take_quality_snapshot()
                monitoring_results["snapshots"].append(snapshot)
                
                # Check for quality degradation
                if len(monitoring_results["snapshots"]) > 1:
                    alerts = self._check_quality_degradation(
                        monitoring_results["snapshots"][-2],
                        monitoring_results["snapshots"][-1]
                    )
                    monitoring_results["alerts"].extend(alerts)
                
                # Wait before next snapshot
                await asyncio.sleep(300)  # 5-minute intervals
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
                await asyncio.sleep(60)
        
        # Calculate trends
        monitoring_results["trends"] = self._calculate_monitoring_trends(monitoring_results["snapshots"])
        
        logger.info("Real-time quality monitoring completed")
        return monitoring_results
    
    async def generate_quality_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for quality dashboard"""
        logger.info("Generating quality dashboard data")
        
        try:
            # Get current quality status for all datasets
            dashboard_data = {
                "timestamp": datetime.now(),
                "overall_health": 0.0,
                "datasets": {},
                "system_alerts": [],
                "trending_issues": [],
                "improvements": [],
                "statistics": {
                    "total_records": 0,
                    "datasets_monitored": 0,
                    "active_alerts": 0,
                    "quality_distribution": {}
                }
            }
            
            total_score = 0
            dataset_count = 0
            
            for dataset in ["videos", "channels", "analytics"]:
                report = await self.quality_framework.generate_quality_report(dataset)
                
                dashboard_data["datasets"][dataset] = {
                    "score": report.metrics.overall_score,
                    "level": report.metrics.quality_level.value,
                    "records": report.metrics.total_records,
                    "issues": len(report.issues),
                    "last_updated": datetime.now()
                }
                
                total_score += report.metrics.overall_score
                dataset_count += 1
                dashboard_data["statistics"]["total_records"] += report.metrics.total_records
                
                # Collect system alerts
                critical_issues = [i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]
                for issue in critical_issues:
                    dashboard_data["system_alerts"].append({
                        "dataset": dataset,
                        "issue": issue.description,
                        "severity": "critical",
                        "records": issue.affected_records
                    })
            
            # Calculate overall health
            dashboard_data["overall_health"] = total_score / dataset_count if dataset_count > 0 else 0.0
            dashboard_data["statistics"]["datasets_monitored"] = dataset_count
            dashboard_data["statistics"]["active_alerts"] = len(dashboard_data["system_alerts"])
            
            # Quality distribution
            quality_levels = [data["level"] for data in dashboard_data["datasets"].values()]
            dashboard_data["statistics"]["quality_distribution"] = {
                level: quality_levels.count(level.value) 
                for level in QualityLevel
            }
            
            logger.info("Quality dashboard data generated successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {"error": str(e), "timestamp": datetime.now()}
    
    def _generate_audit_summary(self, datasets_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of audit results"""
        summary = {
            "overall_health": "unknown",
            "average_score": 0.0,
            "datasets_healthy": 0,
            "datasets_critical": 0,
            "total_alerts": 0,
            "most_problematic_dataset": None,
            "best_performing_dataset": None
        }
        
        if not datasets_results:
            return summary
        
        scores = []
        for dataset, data in datasets_results.items():
            if isinstance(data.get('overall_score'), (int, float)):
                scores.append(data['overall_score'])
                
                if data['quality_level'] in ['excellent', 'good']:
                    summary['datasets_healthy'] += 1
                elif data['quality_level'] == 'critical':
                    summary['datasets_critical'] += 1
        
        if scores:
            summary['average_score'] = sum(scores) / len(scores)
            
            # Determine overall health
            if summary['average_score'] >= 85:
                summary['overall_health'] = "excellent"
            elif summary['average_score'] >= 70:
                summary['overall_health'] = "good"
            elif summary['average_score'] >= 50:
                summary['overall_health'] = "fair"
            else:
                summary['overall_health'] = "critical"
            
            # Find best and worst performing datasets
            dataset_scores = [(k, v.get('overall_score', 0)) for k, v in datasets_results.items() 
                            if isinstance(v.get('overall_score'), (int, float))]
            
            if dataset_scores:
                summary['best_performing_dataset'] = max(dataset_scores, key=lambda x: x[1])[0]
                summary['most_problematic_dataset'] = min(dataset_scores, key=lambda x: x[1])[0]
        
        return summary
    
    def _generate_system_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        summary = audit_results.get("summary", {})
        
        if summary.get("average_score", 0) < 70:
            recommendations.append("🔴 URGENT: System-wide data quality is below acceptable levels")
        
        if summary.get("datasets_critical", 0) > 0:
            recommendations.append(f"🚨 CRITICAL: {summary['datasets_critical']} datasets require immediate attention")
        
        if len(audit_results.get("alerts", [])) > 10:
            recommendations.append("⚠️ HIGH ALERT: Implement automated data validation at ingestion points")
        
        # Dataset-specific recommendations
        for dataset, data in audit_results.get("datasets", {}).items():
            if data.get("overall_score", 0) < 50:
                recommendations.append(f"🔥 PRIORITY: Review {dataset} data pipeline and validation rules")
        
        recommendations.extend([
            "📊 IMPLEMENT: Real-time quality monitoring dashboard",
            "🔄 AUTOMATE: Scheduled quality checks and alerting",
            "📈 TRACK: Quality trend analysis and reporting",
            "🏗️ ESTABLISH: Data governance framework"
        ])
        
        return recommendations[:10]  # Limit to top 10
    
    async def _take_quality_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current quality metrics"""
        snapshot = {
            "timestamp": datetime.now(),
            "datasets": {}
        }
        
        for dataset in ["videos", "channels", "analytics"]:
            try:
                report = await self.quality_framework.generate_quality_report(dataset)
                snapshot["datasets"][dataset] = {
                    "overall_score": report.metrics.overall_score,
                    "completeness": report.metrics.completeness_score,
                    "accuracy": report.metrics.accuracy_score,
                    "consistency": report.metrics.consistency_score,
                    "freshness": report.metrics.freshness_score,
                    "total_records": report.metrics.total_records,
                    "issues": len(report.issues)
                }
            except Exception as e:
                logger.error(f"Error taking snapshot for {dataset}: {e}")
                snapshot["datasets"][dataset] = {"error": str(e)}
        
        return snapshot
    
    def _check_quality_degradation(self, previous: Dict[str, Any], current: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for quality degradation between snapshots"""
        alerts = []
        
        for dataset in previous.get("datasets", {}):
            if dataset not in current.get("datasets", {}):
                continue
            
            prev_data = previous["datasets"][dataset]
            curr_data = current["datasets"][dataset]
            
            if "error" in prev_data or "error" in curr_data:
                continue
            
            # Check for significant score drops
            score_drop = prev_data.get("overall_score", 0) - curr_data.get("overall_score", 0)
            if score_drop > 5:  # More than 5% drop
                alerts.append({
                    "type": "quality_degradation",
                    "dataset": dataset,
                    "severity": "warning",
                    "message": f"Quality score dropped by {score_drop:.1f}% in {dataset}",
                    "timestamp": current["timestamp"]
                })
        
        return alerts
    
    def _calculate_monitoring_trends(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends from monitoring snapshots"""
        if len(snapshots) < 2:
            return {}
        
        trends = {}
        
        for dataset in ["videos", "channels", "analytics"]:
            dataset_scores = []
            for snapshot in snapshots:
                if dataset in snapshot.get("datasets", {}):
                    score = snapshot["datasets"][dataset].get("overall_score")
                    if score is not None:
                        dataset_scores.append(score)
            
            if len(dataset_scores) > 1:
                # Simple trend calculation
                first_half_avg = sum(dataset_scores[:len(dataset_scores)//2]) / (len(dataset_scores)//2)
                second_half_avg = sum(dataset_scores[len(dataset_scores)//2:]) / (len(dataset_scores) - len(dataset_scores)//2)
                
                trend_change = second_half_avg - first_half_avg
                
                if trend_change > 2:
                    trend_direction = "improving"
                elif trend_change < -2:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
                
                trends[dataset] = {
                    "direction": trend_direction,
                    "change": trend_change,
                    "confidence": "medium" if len(dataset_scores) > 5 else "low"
                }
        
        return trends

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Batch Data Quality Processing")
    parser.add_argument('--mode', choices=['audit', 'cleanup', 'monitor', 'dashboard'], 
                       default='audit', help='Processing mode')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['videos', 'channels', 'analytics'],
                       default=['videos', 'channels', 'analytics'],
                       help='Datasets to process')
    parser.add_argument('--output', default='quality_results.json', 
                       help='Output file for results')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run in dry-run mode (cleanup only)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in minutes')
    
    args = parser.parse_args()
    
    runner = BatchQualityRunner()
    
    try:
        if args.mode == 'audit':
            logger.info("Running full quality audit")
            results = await runner.run_full_quality_audit(args.datasets)
            
        elif args.mode == 'cleanup':
            if len(args.datasets) != 1:
                logger.error("Cleanup mode requires exactly one dataset")
                sys.exit(1)
            
            logger.info(f"Running batch cleanup for {args.datasets[0]}")
            results = await runner.run_batch_cleanup(args.datasets[0], args.dry_run)
            
        elif args.mode == 'monitor':
            logger.info(f"Running real-time monitoring for {args.duration} minutes")
            results = await runner.monitor_real_time_quality(args.duration)
            
        elif args.mode == 'dashboard':
            logger.info("Generating dashboard data")
            results = await runner.generate_quality_dashboard_data()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        if args.mode == 'audit':
            print("\n" + "="*80)
            print("📊 DATA QUALITY AUDIT SUMMARY")
            print("="*80)
            summary = results.get("summary", {})
            print(f"Overall Health: {summary.get('overall_health', 'unknown').upper()}")
            print(f"Average Score:  {summary.get('average_score', 0):.1f}%")
            print(f"Healthy Datasets: {summary.get('datasets_healthy', 0)}")
            print(f"Critical Datasets: {summary.get('datasets_critical', 0)}")
            print(f"Total Alerts: {len(results.get('alerts', []))}")
            
            if results.get("recommendations"):
                print("\n🎯 TOP RECOMMENDATIONS:")
                for i, rec in enumerate(results["recommendations"][:5], 1):
                    print(f"{i:2d}. {rec}")
        
        print(f"\n✅ {args.mode.title()} completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the main function
    asyncio.run(main())