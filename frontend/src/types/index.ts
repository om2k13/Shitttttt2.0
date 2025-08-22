export interface Job {
  id: string;
  user_id?: number;
  repo_url: string;
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  error_message?: string;
  findings_count: number;
  severity_breakdown: string;
  tools_used: string;
  pr_number?: number;
  base_branch?: string;
  head_branch?: string;
  is_pr_analysis: boolean;
  organization_id?: number;
  current_stage?: string;
  branch?: string;
  progress: number;
}

export interface Finding {
  id: number;
  job_id: string;
  tool: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  file: string;
  line?: number;
  rule_id: string;
  message: string;
  remediation?: string;
  autofixable: boolean;
  vulnerability_type?: string;
  code_snippet?: string;
  pr_context: string;
  risk_score?: number;
  merge_blocking: boolean;
  test_coverage?: string;
  breaking_change: boolean;
  created_at?: string;
}

export interface Report {
  summary: {
    total: number;
    by_tool: Record<string, number>;
    by_severity: Record<string, number>;
  };
  findings: Finding[];
}

export interface AnalyticsTrends {
  total_findings: number;
  by_severity: Record<string, number>;
  by_tool: Record<string, number>;
  by_date: Record<string, number>;
}

export interface MLInsights {
  risk_prediction: string;
  false_positive_rate: number;
  recommendations: string[];
  trends: {
    security_score: number;
    quality_score: number;
    performance_score: number;
  };
}

export interface BusinessAnalytics {
  roi_metrics: {
    cost_savings: string;
    time_savings: string;
    risk_reduction: string;
  };
  business_impact: {
    customer_satisfaction: string;
    compliance_score: string;
    revenue_impact: string;
  };
}

export interface ComplianceAnalytics {
  overall_score: number;
  standards: Record<string, string>;
  recommendations: string[];
}

export interface PerformanceAnalytics {
  code_quality: {
    cyclomatic_complexity: string;
    maintainability_index: number;
    technical_debt: string;
  };
  performance_metrics: {
    response_time: string;
    throughput: string;
    resource_usage: string;
  };
}

// Enhanced Analysis Types
export interface CodeSnippet {
  file: string;
  line: number;
  code: string;
  context_before: string;
  context_after: string;
  function_name?: string;
  import_context?: string;
}

export interface RiskScore {
  base_score: number;
  context_multiplier: number;
  final_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  risk_factors: string[];
}

export interface FixSuggestion {
  finding_id: number;
  suggestion_type: 'security' | 'code_quality' | 'type_checking' | 'generic';
  description: string;
  code_example?: string;
  risk_reduction: number;
  effort_required: 'low' | 'medium' | 'high';
  automated: boolean;
}

export interface EnhancedFinding extends Finding {
  category?: string;
  effort?: string;
  // Note: Backend returns basic Finding structure with additional fields
  // code_snippets, risk_score (number), fix_suggestions are not yet implemented
}

export interface EnhancedAnalysis {
  job_id: string;
  findings: EnhancedFinding[];
  summary: {
    total_findings: number;
    auto_fixable: number;
    manual_fixes: number;
    risk_distribution: Record<string, number>;
    priority_distribution: Record<string, number>;
    estimated_fix_time?: string;
  };
  recommendations: string[];
  estimated_fix_time?: string;
  risk_reduction_potential?: number;
}

export interface FixSummary {
  job_id: string;
  total_fixes_applied: number;
  fixes_by_type: Record<string, number>;
  git_status: {
    has_changes: boolean;
    current_branch: string;
    remote_origin: string;
    staged_files: string[];
  };
  validation_results: {
    syntax_check_passed: boolean;
    tests_passed: boolean;
    build_successful: boolean;
  };
  next_steps: string[];
}

export interface AutoFixRequest {
  job_id: string;
  fix_types: ('security' | 'code_quality' | 'type_checking')[];
  create_commit: boolean;
  push_changes: boolean;
  create_pr: boolean;
  commit_message?: string;
  branch_name?: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  role: 'user' | 'admin' | 'org_admin';
  organization?: string;
  github_username?: string;
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

export interface UserProfile {
  profile: User;
  statistics: {
    total_jobs: number;
    completed_jobs: number;
    total_findings: number;
    active_github_tokens: number;
  };
  github_tokens: any[];
}

export interface WorkspaceStats {
  total_repos: number;
  total_size: string;
  repos: Array<{
    job_id: string;
    size_human: string;
    age_hours: number;
    repo_url?: string;
    branch?: string;
    is_current?: boolean;
  }>;
  current_repo?: {
    job_id: string;
    repo_url: string;
    branch: string;
    status: string;
    cloned_at: number;
    path: string;
  };
}

export interface JobCreateRequest {
  repo_url: string;
  branch?: string;
  pr_number?: number;
}

export interface PRAnalysisRequest {
  repo_url: string;
  pr_number: number;
  base_branch?: string;
  head_branch?: string;
}

export interface SecurityAnalysisRequest {
  repo_url: string;
}

export interface TestPlanRequest {
  repo_url: string;
}

export interface PerformanceAnalysisRequest {
  repo_url: string;
}

export interface APIChangeAnalysisRequest {
  repo_url: string;
  base_branch?: string;
  head_branch?: string;
}

export interface ComprehensiveAnalysisRequest {
  repo_url: string;
  include_performance?: boolean;
  include_api_analysis?: boolean;
  include_test_generation?: boolean;
}

export interface GitHubPRRequest {
  repo_url: string;
  pr_number: number;
}
