import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import math
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FindingType(Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COMPLIANCE = "compliance"
    ACCESSIBILITY = "accessibility"

@dataclass
class BusinessImpact:
    """Business impact assessment for findings"""
    finding_id: str
    risk_level: RiskLevel
    business_criticality: float  # 0-1 scale
    customer_impact: float  # 0-1 scale
    revenue_impact: float  # 0-1 scale
    compliance_impact: float  # 0-1 scale
    estimated_cost: float  # USD
    estimated_savings: float  # USD if fixed
    time_to_fix: int  # hours
    priority_score: float  # 0-100

class BusinessIntelligenceEngine:
    """Advanced business intelligence and ROI analysis for code review"""
    
    def __init__(self):
        self.industry_benchmarks = self._load_industry_benchmarks()
        self.cost_models = self._load_cost_models()
        self.risk_weights = self._load_risk_weights()
        self.business_metrics = self._load_business_metrics()
    
    def _load_industry_benchmarks(self) -> Dict:
        """Load industry benchmarks for various metrics"""
        return {
            "security_incident_costs": {
                "data_breach": {
                    "average_cost": 4.35,  # million USD
                    "cost_per_record": 150,  # USD
                    "time_to_contain": 277,  # days
                    "probability": 0.27  # 27% chance per year
                },
                "ransomware": {
                    "average_cost": 1.85,  # million USD
                    "downtime_cost_per_hour": 5000,  # USD
                    "probability": 0.15  # 15% chance per year
                },
                "compliance_violation": {
                    "average_fine": 2.5,  # million USD
                    "legal_costs": 500000,  # USD
                    "probability": 0.12  # 12% chance per year
                }
            },
            "development_metrics": {
                "bug_fix_cost": {
                    "requirements": 1,  # 1x cost
                    "design": 5,  # 5x cost
                    "coding": 10,  # 10x cost
                    "testing": 50,  # 50x cost
                    "production": 100  # 100x cost
                },
                "code_review_efficiency": {
                    "automated_review": 0.8,  # 80% of manual review time
                    "false_positive_rate": 0.15,  # 15% false positives
                    "coverage_improvement": 0.4  # 40% better coverage
                }
            },
            "business_metrics": {
                "customer_satisfaction": {
                    "security_incident_impact": -0.3,  # -30% satisfaction
                    "performance_issue_impact": -0.2,  # -20% satisfaction
                    "quality_issue_impact": -0.15  # -15% satisfaction
                },
                "revenue_impact": {
                    "security_incident": -0.25,  # -25% revenue
                    "performance_degradation": -0.15,  # -15% revenue
                    "compliance_violation": -0.35  # -35% revenue
                }
            }
        }
    
    def _load_cost_models(self) -> Dict:
        """Load cost models for various activities"""
        return {
            "developer_costs": {
                "junior": 75,  # USD per hour
                "mid": 125,  # USD per hour
                "senior": 175,  # USD per hour
                "architect": 225  # USD per hour
            },
            "fix_costs": {
                "security": {
                    "critical": 40,  # hours
                    "high": 24,  # hours
                    "medium": 16,  # hours
                    "low": 8  # hours
                },
                "performance": {
                    "critical": 32,  # hours
                    "high": 20,  # hours
                    "medium": 12,  # hours
                    "low": 6  # hours
                },
                "quality": {
                    "critical": 20,  # hours
                    "high": 12,  # hours
                    "medium": 8,  # hours
                    "low": 4  # hours
                }
            },
            "review_costs": {
                "manual_review": 2,  # hours per finding
                "automated_review": 0.4,  # hours per finding
                "false_positive_cost": 0.5  # hours per false positive
            }
        }
    
    def _load_risk_weights(self) -> Dict:
        """Load risk weighting factors"""
        return {
            "risk_levels": {
                RiskLevel.CRITICAL: 1.0,
                RiskLevel.HIGH: 0.7,
                RiskLevel.MEDIUM: 0.4,
                RiskLevel.LOW: 0.1
            },
            "finding_types": {
                FindingType.SECURITY: 1.0,
                FindingType.COMPLIANCE: 0.9,
                FindingType.PERFORMANCE: 0.7,
                FindingType.QUALITY: 0.5,
                FindingType.ACCESSIBILITY: 0.3
            },
            "business_factors": {
                "customer_facing": 1.2,
                "revenue_critical": 1.3,
                "compliance_required": 1.4,
                "public_exposure": 1.5
            }
        }
    
    def _load_business_metrics(self) -> Dict:
        """Load business-specific metrics"""
        return {
            "company_size": {
                "startup": {"revenue_multiplier": 0.1, "risk_tolerance": 0.8},
                "sme": {"revenue_multiplier": 0.3, "risk_tolerance": 0.6},
                "enterprise": {"revenue_multiplier": 1.0, "risk_tolerance": 0.3},
                "fortune500": {"revenue_multiplier": 2.0, "risk_tolerance": 0.1}
            },
            "industry": {
                "fintech": {"compliance_weight": 1.5, "security_weight": 1.4},
                "healthcare": {"compliance_weight": 1.6, "security_weight": 1.5},
                "ecommerce": {"revenue_weight": 1.3, "performance_weight": 1.2},
                "saas": {"availability_weight": 1.3, "security_weight": 1.2}
            }
        }
    
    async def analyze_business_impact(self, findings: List[Dict], repo_info: Dict) -> Dict:
        """Analyze business impact of findings"""
        impact_analysis = {
            "timestamp": datetime.now().isoformat(),
            "repository": repo_info.get("name", "Unknown"),
            "total_findings": len(findings),
            "business_impact": {},
            "roi_analysis": {},
            "risk_assessment": {},
            "recommendations": {},
            "executive_summary": {}
        }
        
        try:
            # Analyze business impact
            impact_analysis["business_impact"] = await self._calculate_business_impact(findings, repo_info)
            
            # Calculate ROI
            impact_analysis["roi_analysis"] = await self._calculate_roi(findings, repo_info)
            
            # Assess overall risk
            impact_analysis["risk_assessment"] = await self._assess_business_risk(findings, repo_info)
            
            # Generate recommendations
            impact_analysis["recommendations"] = await self._generate_business_recommendations(
                findings, impact_analysis
            )
            
            # Create executive summary
            impact_analysis["executive_summary"] = await self._create_executive_summary(impact_analysis)
            
        except Exception as e:
            impact_analysis["error"] = str(e)
        
        return impact_analysis
    
    async def _calculate_business_impact(self, findings: List[Dict], repo_info: Dict) -> Dict:
        """Calculate business impact for each finding"""
        business_impacts = []
        total_impact = {
            "critical_impact": 0,
            "high_impact": 0,
            "medium_impact": 0,
            "low_impact": 0,
            "total_cost": 0,
            "total_savings": 0,
            "total_time": 0
        }
        
        for finding in findings:
            impact = await self._assess_finding_impact(finding, repo_info)
            business_impacts.append(impact)
            
            # Aggregate impacts
            risk_level = impact.risk_level.value
            if risk_level == "critical":
                total_impact["critical_impact"] += 1
            elif risk_level == "high":
                total_impact["high_impact"] += 1
            elif risk_level == "medium":
                total_impact["medium_impact"] += 1
            else:
                total_impact["low_impact"] += 1
            
            total_impact["total_cost"] += impact.estimated_cost
            total_impact["total_savings"] += impact.estimated_savings
            total_impact["total_time"] += impact.time_to_fix
        
        return {
            "individual_impacts": [impact.__dict__ for impact in business_impacts],
            "aggregate_impacts": total_impact,
            "priority_ranking": sorted(business_impacts, key=lambda x: x.priority_score, reverse=True)
        }
    
    async def _assess_finding_impact(self, finding: Dict, repo_info: Dict) -> BusinessImpact:
        """Assess business impact of a single finding"""
        # Determine finding type
        finding_type = self._classify_finding_type(finding)
        risk_level = RiskLevel(finding.get("severity", "medium"))
        
        # Calculate business criticality
        business_criticality = self._calculate_business_criticality(finding, repo_info)
        
        # Calculate customer impact
        customer_impact = self._calculate_customer_impact(finding, finding_type)
        
        # Calculate revenue impact
        revenue_impact = self._calculate_revenue_impact(finding, finding_type)
        
        # Calculate compliance impact
        compliance_impact = self._calculate_compliance_impact(finding, finding_type)
        
        # Estimate costs and savings
        estimated_cost = self._estimate_fix_cost(finding, finding_type)
        estimated_savings = self._estimate_potential_savings(finding, finding_type)
        
        # Estimate time to fix
        time_to_fix = self._estimate_fix_time(finding, finding_type)
        
        # Calculate priority score
        priority_score = self._calculate_priority_score(
            risk_level, finding_type, business_criticality,
            customer_impact, revenue_impact, compliance_impact
        )
        
        return BusinessImpact(
            finding_id=finding.get("id", "unknown"),
            risk_level=risk_level,
            business_criticality=business_criticality,
            customer_impact=customer_impact,
            revenue_impact=revenue_impact,
            compliance_impact=compliance_impact,
            estimated_cost=estimated_cost,
            estimated_savings=estimated_savings,
            time_to_fix=time_to_fix,
            priority_score=priority_score
        )
    
    def _classify_finding_type(self, finding: Dict) -> FindingType:
        """Classify finding by type"""
        tool = finding.get("tool", "").lower()
        message = finding.get("message", "").lower()
        
        if any(sec in tool for sec in ["bandit", "semgrep", "safety", "npm-audit"]):
            return FindingType.SECURITY
        elif any(perf in tool for perf in ["performance", "complexity", "memory"]):
            return FindingType.PERFORMANCE
        elif any(comp in tool for comp in ["compliance", "gdpr", "hipaa", "pci"]):
            return FindingType.COMPLIANCE
        elif any(acc in tool for acc in ["accessibility", "a11y", "wcag"]):
            return FindingType.ACCESSIBILITY
        else:
            return FindingType.QUALITY
    
    def _calculate_business_criticality(self, finding: Dict, repo_info: Dict) -> float:
        """Calculate business criticality of a finding"""
        criticality = 0.5  # Base criticality
        
        # Check if customer-facing
        if self._is_customer_facing(finding, repo_info):
            criticality += 0.3
        
        # Check if revenue-critical
        if self._is_revenue_critical(finding, repo_info):
            criticality += 0.2
        
        # Check if compliance-required
        if self._is_compliance_required(finding, repo_info):
            criticality += 0.2
        
        # Check if publicly exposed
        if self._is_publicly_exposed(finding, repo_info):
            criticality += 0.1
        
        return min(1.0, criticality)
    
    def _calculate_customer_impact(self, finding: Dict, finding_type: FindingType) -> float:
        """Calculate potential customer impact"""
        base_impact = 0.3
        
        if finding_type == FindingType.SECURITY:
            base_impact += 0.4  # Security issues have high customer impact
        elif finding_type == FindingType.PERFORMANCE:
            base_impact += 0.3  # Performance affects user experience
        elif finding_type == FindingType.QUALITY:
            base_impact += 0.2  # Quality issues affect reliability
        
        return min(1.0, base_impact)
    
    def _calculate_revenue_impact(self, finding: Dict, finding_type: FindingType) -> float:
        """Calculate potential revenue impact"""
        base_impact = 0.2
        
        if finding_type == FindingType.SECURITY:
            base_impact += 0.4  # Security incidents can cause revenue loss
        elif finding_type == FindingType.PERFORMANCE:
            base_impact += 0.3  # Performance affects conversion rates
        elif finding_type == FindingType.COMPLIANCE:
            base_impact += 0.3  # Compliance violations can result in fines
        
        return min(1.0, base_impact)
    
    def _calculate_compliance_impact(self, finding: Dict, finding_type: FindingType) -> float:
        """Calculate compliance impact"""
        if finding_type == FindingType.COMPLIANCE:
            return 0.9  # High compliance impact
        elif finding_type == FindingType.SECURITY:
            return 0.7  # Security affects compliance
        else:
            return 0.3  # Lower compliance impact
    
    def _estimate_fix_cost(self, finding: Dict, finding_type: FindingType) -> float:
        """Estimate cost to fix a finding"""
        severity = finding.get("severity", "medium")
        base_hours = self.cost_models["fix_costs"][finding_type.value][severity]
        
        # Use senior developer cost as default
        hourly_rate = self.cost_models["developer_costs"]["senior"]
        
        return base_hours * hourly_rate
    
    def _estimate_potential_savings(self, finding: Dict, finding_type: FindingType) -> float:
        """Estimate potential savings from fixing a finding"""
        severity = finding.get("severity", "medium")
        
        if finding_type == FindingType.SECURITY:
            # Security findings can prevent costly incidents
            if severity == "critical":
                return self.industry_benchmarks["security_incident_costs"]["data_breach"]["average_cost"] * 1000000 * 0.1
            elif severity == "high":
                return self.industry_benchmarks["security_incident_costs"]["ransomware"]["average_cost"] * 1000000 * 0.05
            else:
                return 10000  # Base savings for security fixes
        
        elif finding_type == FindingType.PERFORMANCE:
            # Performance improvements can increase revenue
            return 5000  # Base savings for performance improvements
        
        elif finding_type == FindingType.COMPLIANCE:
            # Compliance fixes prevent fines
            return self.industry_benchmarks["security_incident_costs"]["compliance_violation"]["average_fine"] * 1000000 * 0.1
        
        else:
            return 2000  # Base savings for quality improvements
    
    def _estimate_fix_time(self, finding: Dict, finding_type: FindingType) -> int:
        """Estimate time to fix a finding"""
        severity = finding.get("severity", "medium")
        return self.cost_models["fix_costs"][finding_type.value][severity]
    
    def _calculate_priority_score(self, risk_level: RiskLevel, finding_type: FindingType,
                                business_criticality: float, customer_impact: float,
                                revenue_impact: float, compliance_impact: float) -> float:
        """Calculate priority score for a finding"""
        # Base score from risk level
        base_score = self.risk_weights["risk_levels"][risk_level] * 40
        
        # Add finding type weight
        type_score = self.risk_weights["finding_types"][finding_type] * 20
        
        # Add business impact scores
        business_score = business_criticality * 15
        customer_score = customer_impact * 10
        revenue_score = revenue_impact * 10
        compliance_score = compliance_impact * 5
        
        total_score = base_score + type_score + business_score + customer_score + revenue_score + compliance_score
        
        return min(100, total_score)
    
    def _is_customer_facing(self, finding: Dict, repo_info: Dict) -> bool:
        """Check if finding affects customer-facing functionality"""
        # This is a simplified check - in practice, you'd have more sophisticated logic
        customer_keywords = ["api", "ui", "frontend", "user", "customer", "web", "mobile"]
        file_path = finding.get("file", "").lower()
        
        return any(keyword in file_path for keyword in customer_keywords)
    
    def _is_revenue_critical(self, finding: Dict, repo_info: Dict) -> bool:
        """Check if finding affects revenue-critical functionality"""
        revenue_keywords = ["payment", "billing", "subscription", "order", "checkout", "transaction"]
        file_path = finding.get("file", "").lower()
        
        return any(keyword in file_path for keyword in revenue_keywords)
    
    def _is_compliance_required(self, finding: Dict, repo_info: Dict) -> bool:
        """Check if finding affects compliance-required functionality"""
        compliance_keywords = ["gdpr", "hipaa", "pci", "soc2", "compliance", "audit"]
        file_path = finding.get("file", "").lower()
        message = finding.get("message", "").lower()
        
        return any(keyword in file_path or keyword in message for keyword in compliance_keywords)
    
    def _is_publicly_exposed(self, finding: Dict, repo_info: Dict) -> bool:
        """Check if finding affects publicly exposed functionality"""
        public_keywords = ["public", "api", "endpoint", "web", "internet", "external"]
        file_path = finding.get("file", "").lower()
        
        return any(keyword in file_path for keyword in public_keywords)
    
    async def _calculate_roi(self, findings: List[Dict], repo_info: Dict) -> Dict:
        """Calculate ROI of code review activities"""
        total_fix_cost = 0
        total_potential_savings = 0
        total_review_cost = 0
        
        # Calculate fix costs and potential savings
        for finding in findings:
            finding_type = self._classify_finding_type(finding)
            total_fix_cost += self._estimate_fix_cost(finding, finding_type)
            total_potential_savings += self._estimate_potential_savings(finding, finding_type)
        
        # Calculate review costs
        total_review_cost = len(findings) * self.cost_models["review_costs"]["automated_review"] * \
                           self.cost_models["developer_costs"]["senior"]
        
        # Calculate ROI
        total_investment = total_fix_cost + total_review_cost
        roi_percentage = ((total_potential_savings - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        # Calculate payback period
        monthly_savings = total_potential_savings / 12  # Assume savings over 1 year
        payback_months = total_investment / monthly_savings if monthly_savings > 0 else float('inf')
        
        return {
            "total_investment": total_investment,
            "total_potential_savings": total_potential_savings,
            "total_fix_cost": total_fix_cost,
            "total_review_cost": total_review_cost,
            "roi_percentage": roi_percentage,
            "payback_period_months": payback_months,
            "net_benefit": total_potential_savings - total_investment,
            "cost_benefit_ratio": total_potential_savings / total_investment if total_investment > 0 else 0
        }
    
    async def _assess_business_risk(self, findings: List[Dict], repo_info: Dict) -> Dict:
        """Assess overall business risk"""
        risk_scores = {
            "security_risk": 0,
            "compliance_risk": 0,
            "performance_risk": 0,
            "quality_risk": 0,
            "overall_risk": 0
        }
        
        # Calculate risk scores by category
        for finding in findings:
            finding_type = self._classify_finding_type(finding)
            severity = finding.get("severity", "medium")
            severity_weight = self.risk_weights["risk_levels"][RiskLevel(severity)]
            
            if finding_type == FindingType.SECURITY:
                risk_scores["security_risk"] += severity_weight
            elif finding_type == FindingType.COMPLIANCE:
                risk_scores["compliance_risk"] += severity_weight
            elif finding_type == FindingType.PERFORMANCE:
                risk_scores["performance_risk"] += severity_weight
            else:
                risk_scores["quality_risk"] += severity_weight
        
        # Normalize risk scores
        max_possible_risk = len(findings) * 1.0  # Assuming all findings could be critical
        for risk_type in risk_scores:
            if max_possible_risk > 0:
                risk_scores[risk_type] = min(10.0, (risk_scores[risk_type] / max_possible_risk) * 10)
        
        # Calculate overall risk
        risk_scores["overall_risk"] = sum([
            risk_scores["security_risk"] * 0.4,  # Security is 40% of overall risk
            risk_scores["compliance_risk"] * 0.3,  # Compliance is 30% of overall risk
            risk_scores["performance_risk"] * 0.2,  # Performance is 20% of overall risk
            risk_scores["quality_risk"] * 0.1  # Quality is 10% of overall risk
        ])
        
        # Determine risk level
        if risk_scores["overall_risk"] >= 8:
            risk_level = "Critical"
        elif risk_scores["overall_risk"] >= 6:
            risk_level = "High"
        elif risk_scores["overall_risk"] >= 4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "risk_scores": risk_scores,
            "risk_level": risk_level,
            "risk_description": self._get_risk_description(risk_level),
            "risk_trends": await self._analyze_risk_trends(findings)
        }
    
    def _get_risk_description(self, risk_level: str) -> str:
        """Get human-readable risk description"""
        descriptions = {
            "Critical": "Immediate action required. High probability of significant business impact.",
            "High": "Urgent attention needed. Potential for moderate to high business impact.",
            "Medium": "Attention required. Some business impact possible.",
            "Low": "Monitor and address when possible. Minimal business impact expected."
        }
        return descriptions.get(risk_level, "Risk level unknown.")
    
    async def _analyze_risk_trends(self, findings: List[Dict]) -> Dict:
        """Analyze risk trends over time"""
        # This would typically analyze historical data
        # For now, we'll provide a basic analysis
        return {
            "trend": "stable",  # stable, improving, deteriorating
            "confidence": 0.7,
            "factors": ["Limited historical data available"],
            "recommendations": ["Collect more historical data for trend analysis"]
        }
    
    async def _generate_business_recommendations(self, findings: List[Dict], impact_analysis: Dict) -> Dict:
        """Generate business-focused recommendations"""
        recommendations = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_strategy": [],
            "resource_allocation": {},
            "success_metrics": []
        }
        
        # Immediate actions for critical findings
        critical_findings = [f for f in findings if f.get("severity") == "critical"]
        if critical_findings:
            recommendations["immediate_actions"].append({
                "action": "Address all critical findings within 24 hours",
                "priority": "Critical",
                "business_impact": "Prevent potential security incidents and compliance violations",
                "estimated_effort": "High"
            })
        
        # Short-term goals
        high_findings = [f for f in findings if f.get("severity") == "high"]
        if high_findings:
            recommendations["short_term_goals"].append({
                "goal": "Resolve high-severity findings within 1 week",
                "priority": "High",
                "business_impact": "Reduce business risk and improve system reliability",
                "estimated_effort": "Medium"
            })
        
        # Long-term strategy
        recommendations["long_term_strategy"].append({
            "strategy": "Implement automated code review in CI/CD pipeline",
            "priority": "Medium",
            "business_impact": "Prevent future issues and reduce review costs",
            "estimated_effort": "Medium",
            "timeline": "3-6 months"
        })
        
        # Resource allocation
        total_fix_time = impact_analysis["business_impact"]["aggregate_impacts"]["total_time"]
        recommendations["resource_allocation"] = {
            "estimated_developer_hours": total_fix_time,
            "recommended_team_size": max(1, math.ceil(total_fix_time / 40)),  # 40 hours per week
            "timeline_weeks": math.ceil(total_fix_time / 40),
            "cost_estimate": impact_analysis["business_impact"]["aggregate_impacts"]["total_cost"]
        }
        
        # Success metrics
        recommendations["success_metrics"] = [
            "Reduce critical and high-severity findings by 80%",
            "Achieve compliance score of 90% or higher",
            "Reduce security incident probability by 50%",
            "Improve code review efficiency by 40%"
        ]
        
        return recommendations
    
    async def _create_executive_summary(self, impact_analysis: Dict) -> Dict:
        """Create executive-level summary"""
        business_impact = impact_analysis["business_impact"]["aggregate_impacts"]
        roi_analysis = impact_analysis["roi_analysis"]
        risk_assessment = impact_analysis["risk_assessment"]
        
        return {
            "overview": f"Code review analysis of {impact_analysis['repository']} identified {business_impact['total_findings']} findings requiring attention.",
            "key_findings": {
                "critical_issues": business_impact["critical_impact"],
                "high_priority_issues": business_impact["high_impact"],
                "estimated_fix_cost": f"${business_impact['total_cost']:,.2f}",
                "potential_savings": f"${business_impact['total_savings']:,.2f}"
            },
            "business_impact": {
                "risk_level": risk_assessment["risk_level"],
                "roi": f"{roi_analysis['roi_percentage']:.1f}%",
                "payback_period": f"{roi_analysis['payback_period_months']:.1f} months",
                "net_benefit": f"${roi_analysis['net_benefit']:,.2f}"
            },
            "recommendations": [
                "Immediate attention to critical findings",
                "Address high-priority issues within 1 week",
                "Implement automated review processes",
                "Establish ongoing monitoring and review"
            ],
            "next_steps": [
                "Review detailed findings and prioritize fixes",
                "Allocate resources for immediate remediation",
                "Develop long-term improvement strategy",
                "Schedule follow-up review in 30 days"
            ]
        }
    
    async def generate_business_report(self, findings: List[Dict], repo_info: Dict, format: str = "json") -> str:
        """Generate comprehensive business intelligence report"""
        impact_analysis = await self.analyze_business_impact(findings, repo_info)
        
        if format == "json":
            return json.dumps(impact_analysis, indent=2, default=str)
        elif format == "yaml":
            return yaml.dump(impact_analysis, default_flow_style=False, default_representer=str)
        elif format == "html":
            return self._generate_html_business_report(impact_analysis)
        else:
            return str(impact_analysis)
    
    def _generate_html_business_report(self, impact_analysis: Dict) -> str:
        """Generate HTML business intelligence report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Business Intelligence Report - {impact_analysis.get('repository', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
                .metric-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .critical {{ border-left: 5px solid #f44336; }}
                .high {{ border-left: 5px solid #ff9800; }}
                .medium {{ border-left: 5px solid #ffc107; }}
                .low {{ border-left: 5px solid #4caf50; }}
                .roi-positive {{ color: #4caf50; font-weight: bold; }}
                .roi-negative {{ color: #f44336; font-weight: bold; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .recommendation {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #667eea; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Business Intelligence Report</h1>
                    <p><strong>Repository:</strong> {impact_analysis.get('repository', 'Unknown')}</p>
                    <p><strong>Generated:</strong> {impact_analysis.get('timestamp', 'Unknown')}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>Overall Risk Level</h3>
                            <p style="font-size: 24px; font-weight: bold; color: #667eea;">
                                {impact_analysis.get('risk_assessment', {}).get('risk_level', 'Unknown')}
                            </p>
                        </div>
                        <div class="metric-card">
                            <h3>ROI</h3>
                            <p style="font-size: 24px; font-weight: bold;" class="{'roi-positive' if impact_analysis.get('roi_analysis', {}).get('roi_percentage', 0) > 0 else 'roi-negative'}">
                                {impact_analysis.get('roi_analysis', {}).get('roi_percentage', 0):.1f}%
                            </p>
                        </div>
                        <div class="metric-card">
                            <h3>Payback Period</h3>
                            <p style="font-size: 24px; font-weight: bold; color: #667eea;">
                                {impact_analysis.get('roi_analysis', {}).get('payback_period_months', 0):.1f} months
                            </p>
                        </div>
                        <div class="metric-card">
                            <h3>Total Findings</h3>
                            <p style="font-size: 24px; font-weight: bold; color: #667eea;">
                                {impact_analysis.get('total_findings', 0)}
                            </p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>💰 Business Impact Analysis</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>Estimated Fix Cost</h3>
                            <p style="font-size: 20px; color: #f44336;">
                                ${impact_analysis.get('business_impact', {}).get('aggregate_impacts', {}).get('total_cost', 0):,.2f}
                            </p>
                        </div>
                        <div class="metric-card">
                            <h3>Potential Savings</h3>
                            <p style="font-size: 20px; color: #4caf50;">
                                ${impact_analysis.get('business_impact', {}).get('aggregate_impacts', {}).get('total_savings', 0):,.2f}
                            </p>
                        </div>
                        <div class="metric-card">
                            <h3>Net Benefit</h3>
                            <p style="font-size: 20px; color: #4caf50;">
                                ${impact_analysis.get('roi_analysis', {}).get('net_benefit', 0):,.2f}
                            </p>
                        </div>
                        <div class="metric-card">
                            <h3>Total Fix Time</h3>
                            <p style="font-size: 20px; color: #667eea;">
                                {impact_analysis.get('business_impact', {}).get('aggregate_impacts', {}).get('total_time', 0)} hours
                            </p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Recommendations</h2>
                    <div class="recommendation">
                        <h3>Immediate Actions</h3>
                        <ul>
        """
        
        # Add immediate actions
        for action in impact_analysis.get("recommendations", {}).get("immediate_actions", []):
            html_content += f'<li><strong>{action.get("action", "")}</strong> - {action.get("business_impact", "")}</li>'
        
        html_content += """
                        </ul>
                    </div>
                    
                    <div class="recommendation">
                        <h3>Short-term Goals</h3>
                        <ul>
        """
        
        # Add short-term goals
        for goal in impact_analysis.get("recommendations", {}).get("short_term_goals", []):
            html_content += f'<li><strong>{goal.get("goal", "")}</strong> - {goal.get("business_impact", "")}</li>'
        
        html_content += """
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Success Metrics</h2>
                    <ul>
        """
        
        # Add success metrics
        for metric in impact_analysis.get("recommendations", {}).get("success_metrics", []):
            html_content += f'<li>{metric}</li>'
        
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content

# Global business intelligence engine instance
business_intelligence_engine = BusinessIntelligenceEngine()

# Convenience functions
async def analyze_business_impact(findings: List[Dict], repo_info: Dict) -> Dict:
    """Analyze business impact of findings"""
    return await business_intelligence_engine.analyze_business_impact(findings, repo_info)

async def generate_business_report(findings: List[Dict], repo_info: Dict, format: str = "json") -> str:
    """Generate business intelligence report"""
    return await business_intelligence_engine.generate_business_report(findings, repo_info, format)
