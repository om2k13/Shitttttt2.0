import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from ..core.settings import settings

class AdvancedReporter:
    """Advanced reporting and analytics for code review findings"""
    
    def __init__(self):
        self.report_templates = self._load_report_templates()
        self.chart_styles = self._load_chart_styles()
    
    def _load_report_templates(self) -> Dict:
        """Load report templates for different report types"""
        return {
            "executive_summary": {
                "title": "Code Review Executive Summary",
                "sections": [
                    "overview",
                    "risk_assessment",
                    "trends",
                    "recommendations",
                    "business_impact"
                ]
            },
            "technical_detailed": {
                "title": "Technical Code Review Report",
                "sections": [
                    "executive_summary",
                    "methodology",
                    "findings_summary",
                    "detailed_findings",
                    "risk_analysis",
                    "remediation_plan",
                    "appendix"
                ]
            },
            "security_focus": {
                "title": "Security Code Review Report",
                "sections": [
                    "executive_summary",
                    "security_overview",
                    "vulnerability_analysis",
                    "risk_scoring",
                    "remediation_priorities",
                    "security_recommendations"
                ]
            },
            "performance_focus": {
                "title": "Performance Code Review Report",
                "sections": [
                    "executive_summary",
                    "performance_overview",
                    "bottleneck_analysis",
                    "optimization_opportunities",
                    "performance_metrics",
                    "recommendations"
                ]
            }
        }
    
    def _load_chart_styles(self) -> Dict:
        """Load chart styling configurations"""
        return {
            "colors": {
                "critical": "#dc3545",
                "high": "#fd7e14",
                "medium": "#ffc107",
                "low": "#28a745",
                "info": "#17a2b8"
            },
            "chart_theme": "plotly_white",
            "font_family": "Arial, sans-serif",
            "title_font_size": 18,
            "axis_font_size": 12
        }
    
    async def generate_executive_report(self, findings: List[Dict], repo_info: Dict) -> Dict:
        """Generate an executive-level report with business impact analysis"""
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "repository": repo_info.get("name", "Unknown"),
                "analysis_date": repo_info.get("analysis_date", datetime.now().isoformat()),
                "total_files_analyzed": repo_info.get("total_files", 0)
            },
            "executive_summary": self._generate_executive_summary(findings),
            "risk_assessment": self._generate_risk_assessment(findings),
            "trends": self._generate_trend_analysis(findings),
            "business_impact": self._generate_business_impact_analysis(findings),
            "recommendations": self._generate_executive_recommendations(findings),
            "charts": await self._generate_executive_charts(findings)
        }
        
        return report
    
    def _generate_executive_summary(self, findings: List[Dict]) -> Dict:
        """Generate executive summary section"""
        total_findings = len(findings)
        critical_findings = len([f for f in findings if f.get("severity") == "critical"])
        high_findings = len([f for f in findings if f.get("severity") == "high"])
        
        # Calculate overall risk score
        risk_score = self._calculate_overall_risk_score(findings)
        
        # Determine risk level
        if risk_score >= 8:
            risk_level = "Critical"
            risk_description = "Immediate attention required. High probability of security breaches or system failures."
        elif risk_score >= 6:
            risk_level = "High"
            risk_description = "Significant risks identified. Should be addressed before next release."
        elif risk_score >= 4:
            risk_level = "Medium"
            risk_description = "Moderate risks present. Plan remediation for upcoming sprints."
        else:
            risk_level = "Low"
            risk_description = "Minimal risks. Continue with current development practices."
        
        return {
            "total_findings": total_findings,
            "critical_findings": critical_findings,
            "high_findings": high_findings,
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "risk_description": risk_description,
            "key_insights": self._extract_key_insights(findings)
        }
    
    def _generate_risk_assessment(self, findings: List[Dict]) -> Dict:
        """Generate comprehensive risk assessment"""
        # Categorize findings by type
        security_findings = [f for f in findings if "security" in f.get("tool", "").lower()]
        performance_findings = [f for f in findings if "performance" in f.get("tool", "").lower()]
        quality_findings = [f for f in findings if f.get("tool") not in ["security", "performance"]]
        
        # Calculate risk scores by category
        security_risk = self._calculate_category_risk_score(security_findings)
        performance_risk = self._calculate_category_risk_score(performance_findings)
        quality_risk = self._calculate_category_risk_score(quality_findings)
        
        # Identify top risks
        top_risks = self._identify_top_risks(findings)
        
        return {
            "overall_risk_score": self._calculate_overall_risk_score(findings),
            "risk_by_category": {
                "security": {
                    "score": security_risk,
                    "findings_count": len(security_findings),
                    "level": self._get_risk_level(security_risk)
                },
                "performance": {
                    "score": performance_risk,
                    "findings_count": len(performance_findings),
                    "level": self._get_risk_level(performance_risk)
                },
                "quality": {
                    "score": quality_risk,
                    "findings_count": len(quality_findings),
                    "level": self._get_risk_level(quality_risk)
                }
            },
            "top_risks": top_risks,
            "risk_trends": self._analyze_risk_trends(findings)
        }
    
    def _generate_trend_analysis(self, findings: List[Dict]) -> Dict:
        """Generate trend analysis section"""
        # Group findings by tool and severity
        tool_breakdown = {}
        severity_breakdown = {}
        
        for finding in findings:
            tool = finding.get("tool", "unknown")
            severity = finding.get("severity", "unknown")
            
            if tool not in tool_breakdown:
                tool_breakdown[tool] = 0
            tool_breakdown[tool] += 1
            
            if severity not in severity_breakdown:
                severity_breakdown[severity] = 0
            severity_breakdown[severity] += 1
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(findings)
        
        return {
            "tool_breakdown": tool_breakdown,
            "severity_breakdown": severity_breakdown,
            "improvement_metrics": improvement_metrics,
            "trend_insights": self._generate_trend_insights(findings)
        }
    
    def _generate_business_impact_analysis(self, findings: List[Dict]) -> Dict:
        """Generate business impact analysis"""
        # Calculate potential costs
        security_breach_cost = self._estimate_security_breach_cost(findings)
        performance_impact_cost = self._estimate_performance_impact_cost(findings)
        maintenance_cost = self._estimate_maintenance_cost(findings)
        
        # Calculate ROI of fixing issues
        fix_cost_estimate = self._estimate_fix_costs(findings)
        risk_reduction_benefit = self._estimate_risk_reduction_benefit(findings)
        
        return {
            "potential_costs": {
                "security_breach": security_breach_cost,
                "performance_impact": performance_impact_cost,
                "maintenance": maintenance_cost,
                "total_potential": security_breach_cost + performance_impact_cost + maintenance_cost
            },
            "remediation_roi": {
                "estimated_fix_cost": fix_cost_estimate,
                "risk_reduction_benefit": risk_reduction_benefit,
                "roi_ratio": risk_reduction_benefit / fix_cost_estimate if fix_cost_estimate > 0 else 0,
                "payback_period_days": fix_cost_estimate / (risk_reduction_benefit / 365) if risk_reduction_benefit > 0 else float('inf')
            },
            "business_priorities": self._identify_business_priorities(findings)
        }
    
    def _generate_executive_recommendations(self, findings: List[Dict]) -> List[Dict]:
        """Generate executive-level recommendations"""
        recommendations = []
        
        # Security recommendations
        security_findings = [f for f in findings if "security" in f.get("tool", "").lower()]
        if security_findings:
            critical_security = [f for f in security_findings if f.get("severity") in ["critical", "high"]]
            if critical_security:
                recommendations.append({
                    "priority": "Immediate",
                    "category": "Security",
                    "action": "Address critical and high-severity security findings",
                    "business_impact": "Prevent potential security breaches and compliance violations",
                    "estimated_effort": "2-4 weeks",
                    "resources_needed": "Security team, development team"
                })
        
        # Performance recommendations
        performance_findings = [f for f in findings if "performance" in f.get("tool", "").lower()]
        if performance_findings:
            recommendations.append({
                "priority": "High",
                "category": "Performance",
                "action": "Optimize performance bottlenecks",
                "business_impact": "Improve user experience and reduce infrastructure costs",
                "estimated_effort": "3-6 weeks",
                "resources_needed": "Performance engineer, development team"
            })
        
        # Quality recommendations
        quality_findings = [f for f in findings if f.get("severity") in ["critical", "high"]]
        if quality_findings:
            recommendations.append({
                "priority": "Medium",
                "category": "Code Quality",
                "action": "Implement automated code review in CI/CD pipeline",
                "business_impact": "Prevent future issues and improve development velocity",
                "estimated_effort": "1-2 weeks",
                "resources_needed": "DevOps team, development team"
            })
        
        return recommendations
    
    async def _generate_executive_charts(self, findings: List[Dict]) -> Dict:
        """Generate executive-level charts and visualizations"""
        charts = {}
        
        # Risk distribution chart
        charts["risk_distribution"] = await self._create_risk_distribution_chart(findings)
        
        # Severity breakdown chart
        charts["severity_breakdown"] = await self._create_severity_breakdown_chart(findings)
        
        # Tool effectiveness chart
        charts["tool_effectiveness"] = await self._create_tool_effectiveness_chart(findings)
        
        # Trend analysis chart
        charts["trend_analysis"] = await self._create_trend_analysis_chart(findings)
        
        return charts
    
    async def _create_risk_distribution_chart(self, findings: List[Dict]) -> Dict:
        """Create risk distribution chart"""
        # Group findings by risk level
        risk_levels = ["critical", "high", "medium", "low"]
        risk_counts = []
        
        for level in risk_levels:
            count = len([f for f in findings if f.get("severity") == level])
            risk_counts.append(count)
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=risk_levels,
            values=risk_counts,
            hole=0.3,
            marker_colors=[self.chart_styles["colors"][level] for level in risk_levels]
        )])
        
        fig.update_layout(
            title="Risk Distribution by Severity",
            title_font_size=self.chart_styles["title_font_size"],
            font_family=self.chart_styles["font_family"]
        )
        
        return {
            "type": "pie_chart",
            "data": fig.to_dict(),
            "title": "Risk Distribution by Severity"
        }
    
    async def _create_severity_breakdown_chart(self, findings: List[Dict]) -> Dict:
        """Create severity breakdown chart"""
        # Group findings by tool and severity
        tools = list(set([f.get("tool", "unknown") for f in findings]))
        severity_levels = ["critical", "high", "medium", "low"]
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for severity in severity_levels:
            tool_counts = []
            for tool in tools:
                count = len([f for f in findings if f.get("tool") == tool and f.get("severity") == severity])
                tool_counts.append(count)
            
            fig.add_trace(go.Bar(
                name=severity.title(),
                x=tools,
                y=tool_counts,
                marker_color=self.chart_styles["colors"][severity]
            ))
        
        fig.update_layout(
            title="Findings by Tool and Severity",
            title_font_size=self.chart_styles["title_font_size"],
            font_family=self.chart_styles["font_family"],
            barmode='stack',
            xaxis_title="Analysis Tools",
            yaxis_title="Number of Findings"
        )
        
        return {
            "type": "stacked_bar_chart",
            "data": fig.to_dict(),
            "title": "Findings by Tool and Severity"
        }
    
    async def _create_tool_effectiveness_chart(self, findings: List[Dict]) -> Dict:
        """Create tool effectiveness chart"""
        # Calculate effectiveness metrics for each tool
        tools = list(set([f.get("tool", "unknown") for f in findings]))
        tool_metrics = []
        
        for tool in tools:
            tool_findings = [f for f in findings if f.get("tool") == tool]
            total_findings = len(tool_findings)
            critical_findings = len([f for f in tool_findings if f.get("severity") == "critical"])
            high_findings = len([f for f in tool_findings if f.get("severity") == "high"])
            
            # Effectiveness score (higher is better)
            effectiveness = (critical_findings * 3 + high_findings * 2) / total_findings if total_findings > 0 else 0
            
            tool_metrics.append({
                "tool": tool,
                "total_findings": total_findings,
                "critical_findings": critical_findings,
                "high_findings": high_findings,
                "effectiveness_score": effectiveness
            })
        
        # Sort by effectiveness score
        tool_metrics.sort(key=lambda x: x["effectiveness_score"], reverse=True)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=[m["tool"] for m in tool_metrics],
                y=[m["effectiveness_score"] for m in tool_metrics],
                marker_color=[self.chart_styles["colors"]["info"] for _ in tool_metrics]
            )
        ])
        
        fig.update_layout(
            title="Tool Effectiveness (Higher is Better)",
            title_font_size=self.chart_styles["title_font_size"],
            font_family=self.chart_styles["font_family"],
            xaxis_title="Analysis Tools",
            yaxis_title="Effectiveness Score"
        )
        
        return {
            "type": "bar_chart",
            "data": fig.to_dict(),
            "title": "Tool Effectiveness Analysis"
        }
    
    async def _create_trend_analysis_chart(self, findings: List[Dict]) -> Dict:
        """Create trend analysis chart"""
        # For now, create a simple trend chart
        # In a real implementation, this would use historical data
        
        # Simulate trend data
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        trend_data = []
        
        for date in dates:
            # Simulate finding count trend
            base_count = 100
            trend_factor = 1 + 0.1 * np.sin((date - pd.Timestamp('2024-01-01')).days / 30)
            trend_data.append(base_count * trend_factor)
        
        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=trend_data,
                mode='lines+markers',
                name='Findings Trend',
                line=dict(color=self.chart_styles["colors"]["info"], width=3)
            )
        ])
        
        fig.update_layout(
            title="Code Review Findings Trend (Simulated)",
            title_font_size=self.chart_styles["title_font_size"],
            font_family=self.chart_styles["font_family"],
            xaxis_title="Date",
            yaxis_title="Number of Findings"
        )
        
        return {
            "type": "line_chart",
            "data": fig.to_dict(),
            "title": "Findings Trend Analysis"
        }
    
    def _calculate_overall_risk_score(self, findings: List[Dict]) -> float:
        """Calculate overall risk score (0-10)"""
        if not findings:
            return 0.0
        
        # Weighted scoring based on severity
        severity_weights = {
            "critical": 10.0,
            "high": 7.0,
            "medium": 4.0,
            "low": 1.0
        }
        
        total_score = 0
        total_weight = 0
        
        for finding in findings:
            severity = finding.get("severity", "medium")
            weight = severity_weights.get(severity, 4.0)
            
            # Additional weight for security findings
            if "security" in finding.get("tool", "").lower():
                weight *= 1.2
            
            total_score += weight
            total_weight += 1
        
        # Normalize to 0-10 scale
        if total_weight > 0:
            normalized_score = (total_score / total_weight) * 10 / max(severity_weights.values())
            return min(10.0, normalized_score)
        
        return 0.0
    
    def _calculate_category_risk_score(self, findings: List[Dict]) -> float:
        """Calculate risk score for a specific category of findings"""
        return self._calculate_overall_risk_score(findings)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level based on score"""
        if risk_score >= 8:
            return "Critical"
        elif risk_score >= 6:
            return "High"
        elif risk_score >= 4:
            return "Medium"
        else:
            return "Low"
    
    def _identify_top_risks(self, findings: List[Dict]) -> List[Dict]:
        """Identify top risks based on severity and impact"""
        # Sort findings by risk score
        risk_scores = []
        for finding in findings:
            risk_score = self._calculate_finding_risk_score(finding)
            risk_scores.append((finding, risk_score))
        
        # Sort by risk score (descending)
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 risks
        top_risks = []
        for finding, score in risk_scores[:5]:
            top_risks.append({
                "finding": finding,
                "risk_score": score,
                "description": finding.get("message", "No description"),
                "file": finding.get("file", "Unknown"),
                "line": finding.get("line", 0)
            })
        
        return top_risks
    
    def _calculate_finding_risk_score(self, finding: Dict) -> float:
        """Calculate individual finding risk score"""
        severity_weights = {
            "critical": 10.0,
            "high": 7.0,
            "medium": 4.0,
            "low": 1.0
        }
        
        base_score = severity_weights.get(finding.get("severity", "medium"), 4.0)
        
        # Adjust for security findings
        if "security" in finding.get("tool", "").lower():
            base_score *= 1.5
        
        # Adjust for autofixable findings (lower risk)
        if finding.get("autofixable", False):
            base_score *= 0.8
        
        return min(10.0, base_score)
    
    def _extract_key_insights(self, findings: List[Dict]) -> List[str]:
        """Extract key insights from findings"""
        insights = []
        
        # Security insights
        security_findings = [f for f in findings if "security" in f.get("tool", "").lower()]
        if security_findings:
            critical_security = [f for f in security_findings if f.get("severity") == "critical"]
            if critical_security:
                insights.append(f"🚨 {len(critical_security)} critical security vulnerabilities detected")
        
        # Performance insights
        performance_findings = [f for f in findings if "performance" in f.get("tool", "").lower()]
        if performance_findings:
            insights.append(f"⚡ {len(performance_findings)} performance optimization opportunities identified")
        
        # Quality insights
        total_findings = len(findings)
        if total_findings > 50:
            insights.append(f"📊 High number of findings ({total_findings}) suggests need for improved development practices")
        
        return insights
    
    def _calculate_improvement_metrics(self, findings: List[Dict]) -> Dict:
        """Calculate improvement metrics"""
        # This would typically compare with historical data
        # For now, return basic metrics
        
        total_findings = len(findings)
        critical_findings = len([f for f in findings if f.get("severity") == "critical"])
        high_findings = len([f for f in findings if f.get("severity") == "high"])
        
        return {
            "total_findings": total_findings,
            "critical_findings": critical_findings,
            "high_findings": high_findings,
            "critical_percentage": (critical_findings / total_findings * 100) if total_findings > 0 else 0,
            "high_percentage": (high_findings / total_findings * 100) if total_findings > 0 else 0
        }
    
    def _generate_trend_insights(self, findings: List[Dict]) -> List[str]:
        """Generate trend insights"""
        insights = []
        
        # Basic insights based on current data
        total_findings = len(findings)
        
        if total_findings == 0:
            insights.append("✅ No issues found - excellent code quality!")
        elif total_findings < 10:
            insights.append("✅ Low number of findings - good code quality maintained")
        elif total_findings < 50:
            insights.append("⚠️ Moderate number of findings - room for improvement")
        else:
            insights.append("🚨 High number of findings - significant improvement needed")
        
        return insights
    
    def _estimate_security_breach_cost(self, findings: List[Dict]) -> float:
        """Estimate potential cost of security breach"""
        # This is a simplified estimation
        security_findings = [f for f in findings if "security" in f.get("tool", "").lower()]
        critical_security = len([f for f in security_findings if f.get("severity") == "critical"])
        high_security = len([f for f in security_findings if f.get("severity") == "high"])
        
        # Estimated costs (in USD)
        critical_cost = 100000  # $100k per critical vulnerability
        high_cost = 25000       # $25k per high vulnerability
        
        return critical_security * critical_cost + high_security * high_cost
    
    def _estimate_performance_impact_cost(self, findings: List[Dict]) -> float:
        """Estimate cost of performance issues"""
        performance_findings = [f for f in findings if "performance" in f.get("tool", "").lower()]
        
        # Estimated cost per performance issue
        cost_per_issue = 5000  # $5k per performance issue
        
        return len(performance_findings) * cost_per_issue
    
    def _estimate_maintenance_cost(self, findings: List[Dict]) -> float:
        """Estimate maintenance cost of code quality issues"""
        quality_findings = [f for f in findings if f.get("severity") in ["critical", "high"]]
        
        # Estimated cost per quality issue
        cost_per_issue = 2000  # $2k per quality issue
        
        return len(quality_findings) * cost_per_issue
    
    def _estimate_fix_costs(self, findings: List[Dict]) -> float:
        """Estimate cost to fix all issues"""
        total_findings = len(findings)
        
        # Estimated fix costs per finding type
        critical_cost = 2000    # $2k per critical issue
        high_cost = 1000        # $1k per high issue
        medium_cost = 500       # $500 per medium issue
        low_cost = 200          # $200 per low issue
        
        total_cost = 0
        for finding in findings:
            severity = finding.get("severity", "medium")
            if severity == "critical":
                total_cost += critical_cost
            elif severity == "high":
                total_cost += high_cost
            elif severity == "medium":
                total_cost += medium_cost
            else:
                total_cost += low_cost
        
        return total_cost
    
    def _estimate_risk_reduction_benefit(self, findings: List[Dict]) -> float:
        """Estimate benefit of reducing risks"""
        # This is a simplified calculation
        # In practice, this would be based on historical incident data
        
        security_risk = self._estimate_security_breach_cost(findings)
        performance_risk = self._estimate_performance_impact_cost(findings)
        maintenance_risk = self._estimate_maintenance_cost(findings)
        
        # Assume 80% risk reduction after fixing issues
        risk_reduction_factor = 0.8
        
        return (security_risk + performance_risk + maintenance_risk) * risk_reduction_factor
    
    def _identify_business_priorities(self, findings: List[Dict]) -> List[Dict]:
        """Identify business priorities based on findings"""
        priorities = []
        
        # Security priority
        security_findings = [f for f in findings if "security" in f.get("tool", "").lower()]
        if security_findings:
            priorities.append({
                "category": "Security",
                "priority": "High",
                "rationale": f"{len(security_findings)} security issues identified",
                "business_impact": "Prevent data breaches and compliance violations"
            })
        
        # Performance priority
        performance_findings = [f for f in findings if "performance" in f.get("tool", "").lower()]
        if performance_findings:
            priorities.append({
                "category": "Performance",
                "priority": "Medium",
                "rationale": f"{len(performance_findings)} performance issues identified",
                "business_impact": "Improve user experience and reduce costs"
            })
        
        # Quality priority
        quality_findings = [f for f in findings if f.get("severity") in ["critical", "high"]]
        if quality_findings:
            priorities.append({
                "category": "Code Quality",
                "priority": "Medium",
                "rationale": f"{len(quality_findings)} critical/high quality issues identified",
                "business_impact": "Reduce maintenance costs and improve reliability"
            })
        
        return priorities

# Global advanced reporter instance
advanced_reporter = AdvancedReporter()

# Convenience functions
async def generate_executive_report(findings: List[Dict], repo_info: Dict) -> Dict:
    """Generate executive-level report"""
    return await advanced_reporter.generate_executive_report(findings, repo_info)
