from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class AnalysisType(str, Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    API_CHANGES = "api_changes"
    TEST_GENERATION = "test_generation"
    PR_ANALYSIS = "pr_analysis"
    COMPREHENSIVE = "comprehensive"

class SecurityLevel(str, Enum):
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    PARANOID = "paranoid"

class PerformanceThreshold(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ToolConfig(BaseModel):
    """Configuration for individual analysis tools"""
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    severity_threshold: str = Field(default="low", description="Minimum severity to report")
    custom_rules: List[str] = Field(default=[], description="Custom rules for the tool")
    timeout: int = Field(default=300, description="Timeout in seconds")
    max_file_size: int = Field(default=10485760, description="Max file size to analyze (10MB)")

class SecurityConfig(BaseModel):
    """Configuration for security analysis"""
    level: SecurityLevel = Field(default=SecurityLevel.STANDARD)
    enable_owasp_top10: bool = Field(default=True)
    enable_dependency_scanning: bool = Field(default=True)
    enable_secret_detection: bool = Field(default=True)
    enable_sast: bool = Field(default=True)
    custom_security_patterns: List[str] = Field(default=[])
    ignore_patterns: List[str] = Field(default=[])
    tools: Dict[str, ToolConfig] = Field(default_factory=dict)

class PerformanceConfig(BaseModel):
    """Configuration for performance analysis"""
    enable_n_plus_one_detection: bool = Field(default=True)
    enable_memory_leak_detection: bool = Field(default=True)
    enable_algorithm_analysis: bool = Field(default=True)
    enable_resource_management: bool = Field(default=True)
    performance_threshold: PerformanceThreshold = Field(default=PerformanceThreshold.MEDIUM)
    ignore_files: List[str] = Field(default=["tests/", "docs/", "examples/"])
    custom_performance_patterns: List[str] = Field(default=[])

class APIAnalysisConfig(BaseModel):
    """Configuration for API change analysis"""
    enable_breaking_change_detection: bool = Field(default=True)
    enable_compatibility_scoring: bool = Field(default=True)
    enable_framework_detection: bool = Field(default=True)
    supported_frameworks: List[str] = Field(default=["fastapi", "flask", "django", "express", "koa"])
    breaking_change_threshold: float = Field(default=0.8, description="Threshold for breaking change detection")
    ignore_endpoints: List[str] = Field(default=["/health", "/metrics", "/docs"])

class TestGenerationConfig(BaseModel):
    """Configuration for test generation"""
    enable_unit_test_generation: bool = Field(default=True)
    enable_integration_test_generation: bool = Field(default=True)
    enable_test_coverage_analysis: bool = Field(default=True)
    preferred_frameworks: Dict[str, str] = Field(default={
        "python": "pytest",
        "javascript": "jest",
        "typescript": "jest"
    })
    test_templates: Dict[str, str] = Field(default={})
    coverage_threshold: float = Field(default=0.8)

class GitHubIntegrationConfig(BaseModel):
    """Configuration for GitHub integration"""
    enable_pr_comments: bool = Field(default=True)
    enable_pr_reviews: bool = Field(default=True)
    enable_status_checks: bool = Field(default=True)
    comment_template: str = Field(default="default", description="Template for PR comments")
    review_template: str = Field(default="default", description="Template for PR reviews")
    auto_comment_threshold: str = Field(default="medium", description="Minimum severity for auto-comments")
    enable_auto_fix_suggestions: bool = Field(default=True)

class RiskScoringConfig(BaseModel):
    """Configuration for risk scoring"""
    enable_risk_scoring: bool = Field(default=True)
    risk_score_weights: Dict[str, float] = Field(default={
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2
    })
    merge_blocking_rules: Dict[str, bool] = Field(default={
        "critical_security": True,
        "hardcoded_secrets": True,
        "sql_injection": True,
        "breaking_api_changes": True
    })
    risk_score_threshold: int = Field(default=7, description="Threshold for merge blocking")

class MultiUserConfig(BaseModel):
    """Configuration for multi-user support"""
    enable_user_management: bool = Field(default=True)
    enable_organization_support: bool = Field(default=True)
    enable_role_based_access: bool = Field(default=True)
    default_user_role: str = Field(default="user")
    token_encryption: bool = Field(default=True)
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    max_users_per_org: int = Field(default=100)

class EnhancedConfig(BaseModel):
    """Comprehensive configuration for enhanced code review agent"""
    
    # Core analysis types
    enabled_analyses: List[AnalysisType] = Field(default=[
        AnalysisType.SECURITY,
        AnalysisType.PERFORMANCE,
        AnalysisType.API_CHANGES,
        AnalysisType.TEST_GENERATION
    ])
    
    # Individual configurations
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    api_analysis: APIAnalysisConfig = Field(default_factory=APIAnalysisConfig)
    test_generation: TestGenerationConfig = Field(default_factory=TestGenerationConfig)
    github_integration: GitHubIntegrationConfig = Field(default_factory=GitHubIntegrationConfig)
    risk_scoring: RiskScoringConfig = Field(default_factory=RiskScoringConfig)
    multi_user: MultiUserConfig = Field(default_factory=MultiUserConfig)
    
    # Global settings
    max_concurrent_jobs: int = Field(default=5, description="Maximum concurrent analysis jobs")
    job_timeout: int = Field(default=1800, description="Default job timeout in seconds")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Reporting
    enable_detailed_reports: bool = Field(default=True)
    enable_json_exports: bool = Field(default=True)
    enable_html_reports: bool = Field(default=False)
    report_template: str = Field(default="default")
    
    # Notifications
    enable_email_notifications: bool = Field(default=False)
    enable_slack_notifications: bool = Field(default=False)
    enable_webhook_notifications: bool = Field(default=False)
    
    # Advanced features
    enable_machine_learning: bool = Field(default=False, description="Enable ML-based analysis")
    enable_custom_rules: bool = Field(default=True, description="Enable custom rule definitions")
    enable_rule_imports: bool = Field(default=True, description="Enable rule import from external sources")
    
    class Config:
        use_enum_values = True

# Default configuration
DEFAULT_CONFIG = EnhancedConfig()

# Configuration presets
PRESETS = {
    "basic": {
        "enabled_analyses": [AnalysisType.SECURITY],
        "security": {"level": SecurityLevel.BASIC},
        "performance": {"enable_n_plus_one_detection": False},
        "api_analysis": {"enable_breaking_change_detection": False},
        "test_generation": {"enable_unit_test_generation": False}
    },
    "standard": {
        "enabled_analyses": [AnalysisType.SECURITY, AnalysisType.PERFORMANCE],
        "security": {"level": SecurityLevel.STANDARD},
        "performance": {"performance_threshold": PerformanceThreshold.MEDIUM}
    },
    "advanced": {
        "enabled_analyses": [AnalysisType.SECURITY, AnalysisType.PERFORMANCE, AnalysisType.API_CHANGES],
        "security": {"level": SecurityLevel.ADVANCED},
        "performance": {"performance_threshold": PerformanceThreshold.HIGH},
        "api_analysis": {"enable_breaking_change_detection": True}
    },
    "enterprise": {
        "enabled_analyses": [AnalysisType.SECURITY, AnalysisType.PERFORMANCE, AnalysisType.API_CHANGES, AnalysisType.TEST_GENERATION],
        "security": {"level": SecurityLevel.PARANOID},
        "performance": {"performance_threshold": PerformanceThreshold.CRITICAL},
        "api_analysis": {"enable_breaking_change_detection": True},
        "test_generation": {"enable_unit_test_generation": True, "enable_integration_test_generation": True},
        "multi_user": {"enable_organization_support": True, "enable_role_based_access": True}
    }
}

def get_preset_config(preset_name: str) -> EnhancedConfig:
    """Get a configuration preset"""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset_data = PRESETS[preset_name]
    return EnhancedConfig(**preset_data)

def merge_configs(base_config: EnhancedConfig, override_config: Dict[str, Any]) -> EnhancedConfig:
    """Merge a base configuration with override values"""
    config_dict = base_config.dict()
    
    # Deep merge
    for key, value in override_config.items():
        if key in config_dict and isinstance(config_dict[key], dict) and isinstance(value, dict):
            config_dict[key].update(value)
        else:
            config_dict[key] = value
    
    return EnhancedConfig(**config_dict)

def validate_config(config: EnhancedConfig) -> List[str]:
    """Validate configuration and return any issues"""
    issues = []
    
    # Check for conflicting settings
    if config.performance.enable_n_plus_one_detection and not config.performance.enable_algorithm_analysis:
        issues.append("N+1 detection requires algorithm analysis to be enabled")
    
    if config.github_integration.enable_pr_reviews and not config.github_integration.enable_pr_comments:
        issues.append("PR reviews require PR comments to be enabled")
    
    if config.risk_scoring.enable_risk_scoring and not config.security.enable_owasp_top10:
        issues.append("Risk scoring works best with OWASP Top 10 enabled")
    
    # Check for resource constraints
    if config.max_concurrent_jobs < 1:
        issues.append("max_concurrent_jobs must be at least 1")
    
    if config.job_timeout < 60:
        issues.append("job_timeout must be at least 60 seconds")
    
    # Check for valid thresholds
    if not 0 <= config.risk_scoring.risk_score_threshold <= 10:
        issues.append("risk_score_threshold must be between 0 and 10")
    
    if not 0 <= config.api_analysis.breaking_change_threshold <= 1:
        issues.append("breaking_change_threshold must be between 0 and 1")
    
    return issues

def get_recommended_config(repo_size: str, team_size: int, security_requirements: str) -> EnhancedConfig:
    """Get recommended configuration based on repository and team characteristics"""
    
    if repo_size == "small" and team_size <= 3:
        return get_preset_config("basic")
    elif repo_size == "medium" and team_size <= 10:
        return get_preset_config("standard")
    elif repo_size == "large" and team_size <= 25:
        return get_preset_config("advanced")
    elif repo_size == "enterprise" or team_size > 25:
        return get_preset_config("enterprise")
    else:
        return get_preset_config("standard")

# Configuration utilities
def export_config(config: EnhancedConfig, format: str = "json") -> str:
    """Export configuration to various formats"""
    if format == "json":
        return config.json(indent=2)
    elif format == "yaml":
        import yaml
        return yaml.dump(config.dict(), default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def import_config(config_data: str, format: str = "json") -> EnhancedConfig:
    """Import configuration from various formats"""
    if format == "json":
        import json
        data = json.loads(config_data)
    elif format == "yaml":
        import yaml
        data = yaml.safe_load(config_data)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return EnhancedConfig(**data)
