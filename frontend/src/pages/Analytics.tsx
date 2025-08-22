import { useQuery } from '@tanstack/react-query'
import { 
  BarChart3, 
  Shield, 
  Zap, 
  TestTube, 
  CheckCircle,
  AlertTriangle,
  Info,
  Download
} from 'lucide-react'
import { analyticsApi } from '@/lib/api'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'

export function Analytics() {
  const { data: trendsData } = useQuery({
    queryKey: ['analytics-trends'],
    queryFn: async () => {
      const response = await analyticsApi.trends()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: mlInsights } = useQuery({
    queryKey: ['ml-insights'],
    queryFn: async () => {
      const response = await analyticsApi.mlInsights()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: businessData } = useQuery({
    queryKey: ['business-analytics'],
    queryFn: async () => {
      const response = await analyticsApi.business()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: complianceData } = useQuery({
    queryKey: ['compliance-analytics'],
    queryFn: async () => {
      const response = await analyticsApi.compliance()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: performanceData } = useQuery({
    queryKey: ['performance-analytics'],
    queryFn: async () => {
      const response = await analyticsApi.performance()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const handleDownloadAnalytics = async (format: 'json' | 'csv') => {
    try {
      const response = await fetch(`/api/reports/analytics/download?format=${format}`)
      
      if (!response.ok) {
        throw new Error('Failed to download analytics report')
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `analytics-report-${new Date().toISOString().split('T')[0]}.${format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      toast.success(`Analytics report downloaded as ${format.toUpperCase()}`)
    } catch (error) {
      console.error('Download error:', error)
      toast.error('Failed to download analytics report')
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'high':
        return 'text-orange-600 bg-orange-50 border-orange-200'
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'low':
        return 'text-blue-600 bg-blue-50 border-blue-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getToolIcon = (tool: string) => {
    switch (tool.toLowerCase()) {
      case 'bandit':
        return '🛡️'
      case 'semgrep':
        return '🔍'
      case 'detect-secrets':
        return '🔐'
      case 'pip-audit':
        return '📦'
      case 'ruff':
        return '🐍'
      case 'mypy':
        return '🔧'
      case 'radon':
        return '📊'
      case 'npm-audit':
        return '📋'
      default:
        return '🛠️'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Comprehensive insights into code quality, security, and performance metrics
          </p>
        </div>
        
        {/* Download Analytics */}
        <div className="flex space-x-2">
          <button
            onClick={() => handleDownloadAnalytics('json')}
            className="btn-outline btn-sm"
          >
            <Download className="h-4 w-4 mr-2" />
            Export JSON
          </button>
          <button
            onClick={() => handleDownloadAnalytics('csv')}
            className="btn-outline btn-sm"
          >
            <Download className="h-4 w-4 mr-2" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Overview Statistics */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <BarChart3 className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Findings</p>
                <p className="text-2xl font-bold text-gray-900">
                  {trendsData?.total_findings || 0}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <AlertTriangle className="h-6 w-6 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Risk Level</p>
                <p className="text-2xl font-bold text-gray-900 capitalize">
                  {mlInsights?.risk_prediction || 'Medium'}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Compliance Score</p>
                <p className="text-2xl font-bold text-gray-900">
                  {complianceData?.overall_score || 0}%
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Zap className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Performance</p>
                <p className="text-2xl font-bold text-gray-900">
                  {performanceData?.code_quality?.maintainability_index || 0}/100
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trends Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Severity Distribution */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Findings by Severity</h2>
            <p className="card-description">
              Distribution of security and quality issues by severity level
            </p>
          </div>
          <div className="card-content">
            {trendsData?.by_severity ? (
              <div className="space-y-4">
                {Object.entries(trendsData.by_severity).map(([severity, count]) => (
                  <div key={severity} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className={cn("px-2 py-1 text-xs font-medium rounded-full", getSeverityColor(severity))}>
                        {severity.toUpperCase()}
                      </span>
                      <span className="text-sm text-gray-600">{severity}</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div 
                          className={cn(
                            "h-2 rounded-full",
                            severity === 'critical' ? 'bg-red-500' :
                            severity === 'high' ? 'bg-orange-500' :
                            severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                          )}
                          style={{ 
                            width: `${(count / trendsData.total_findings) * 100}%` 
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-12 text-right">
                        {count}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No severity data available</p>
              </div>
            )}
          </div>
        </div>

        {/* Tool Distribution */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Findings by Tool</h2>
            <p className="card-description">
              Analysis tools and their discovery rates
            </p>
          </div>
          <div className="card-content">
            {trendsData?.by_tool ? (
              <div className="space-y-4">
                {Object.entries(trendsData.by_tool)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 8)
                  .map(([tool, count]) => (
                  <div key={tool} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-lg">{getToolIcon(tool)}</span>
                      <span className="text-sm font-medium text-gray-900">{tool}</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ 
                            width: `${(count / trendsData.total_findings) * 100}%` 
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-12 text-right">
                        {count}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No tool data available</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ML Insights */}
      {mlInsights && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">AI-Powered Insights</h2>
            <p className="card-description">
              Machine learning recommendations and risk assessment based on real data
            </p>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Recommendations</h3>
                <ul className="space-y-2">
                  {mlInsights.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                      <span className="text-sm text-gray-600">{rec}</span>
                    </li>
                  ))}
                </ul>
                
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Info className="h-4 w-4 text-blue-600" />
                    <span className="text-sm text-blue-800">
                      False Positive Rate: {(mlInsights.false_positive_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Quality Scores</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="flex items-center space-x-2">
                        <Shield className="h-4 w-4 text-red-500" />
                        <span>Security</span>
                      </span>
                      <span className="font-medium">{mlInsights.trends.security_score}/10</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className="bg-red-500 h-3 rounded-full transition-all duration-300" 
                        style={{ width: `${(mlInsights.trends.security_score / 10) * 100}%` }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="flex items-center space-x-2">
                        <TestTube className="h-4 w-4 text-yellow-500" />
                        <span>Quality</span>
                      </span>
                      <span className="font-medium">{mlInsights.trends.quality_score}/10</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className="bg-yellow-500 h-3 rounded-full transition-all duration-300" 
                        style={{ width: `${(mlInsights.trends.quality_score / 10) * 100}%` }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="flex items-center space-x-2">
                        <Zap className="h-4 w-4 text-green-500" />
                        <span>Performance</span>
                      </span>
                      <span className="font-medium">{mlInsights.trends.performance_score}/10</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className="bg-green-500 h-3 rounded-full transition-all duration-300" 
                        style={{ width: `${(mlInsights.trends.performance_score / 10) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Business Analytics */}
      {businessData && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Business Impact Analytics</h2>
            <p className="card-description">
              ROI metrics and business value of code review automation based on real data
            </p>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-gray-900 mb-3">ROI Metrics</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                    <span className="text-sm font-medium text-green-800">Cost Savings</span>
                    <span className="text-lg font-bold text-green-900">{businessData.roi_metrics.cost_savings}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <span className="text-sm font-medium text-blue-800">Time Savings</span>
                    <span className="text-lg font-bold text-blue-900">{businessData.roi_metrics.time_savings}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                    <span className="text-sm font-medium text-purple-800">Risk Reduction</span>
                    <span className="text-lg font-bold text-purple-900">{businessData.roi_metrics.risk_reduction}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Business Impact</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                    <span className="text-sm font-medium text-yellow-800">Customer Satisfaction</span>
                    <span className="text-lg font-bold text-yellow-900">{businessData.business_impact.customer_satisfaction}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-indigo-50 rounded-lg">
                    <span className="text-sm font-medium text-indigo-800">Compliance Score</span>
                    <span className="text-lg font-bold text-indigo-900">{businessData.business_impact.compliance_score}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg">
                    <span className="text-sm font-medium text-emerald-800">Revenue Impact</span>
                    <span className="text-lg font-bold text-emerald-900">{businessData.business_impact.revenue_impact}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Compliance Analytics */}
      {complianceData && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Compliance & Standards</h2>
            <p className="card-description">
              Compliance scores and recommendations based on actual findings
            </p>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Overall Compliance</h3>
                <div className="text-center">
                  <div className="relative inline-flex items-center justify-center w-32 h-32">
                    <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="#e5e7eb"
                        strokeWidth="8"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke={complianceData.overall_score >= 80 ? "#10b981" : complianceData.overall_score >= 60 ? "#f59e0b" : "#ef4444"}
                        strokeWidth="8"
                        strokeDasharray={`${(complianceData.overall_score / 100) * 283} 283`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-2xl font-bold text-gray-900">{complianceData.overall_score}%</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Standards Compliance</h3>
                <div className="space-y-3">
                  {Object.entries(complianceData.standards).map(([standard, status]) => (
                    <div key={standard} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                      <span className="text-sm font-medium text-gray-700">{standard.replace('_', ' ')}</span>
                      <span className={cn(
                        "px-2 py-1 text-xs font-medium rounded-full",
                        status === 'Compliant' ? 'bg-green-100 text-green-800' :
                        status === 'Partially Compliant' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      )}>
                        {status}
                      </span>
                    </div>
                  ))}
                </div>
                
                <div className="mt-4">
                  <h4 className="font-medium text-gray-900 mb-2">Recommendations</h4>
                  <ul className="space-y-2">
                    {complianceData.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                        <span className="text-sm text-gray-600">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Analytics */}
      {performanceData && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Performance & Code Quality</h2>
            <p className="card-description">
              Code performance metrics and quality indicators based on real analysis
            </p>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Code Quality Metrics</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Cyclomatic Complexity</span>
                      <span className="font-medium">{performanceData.code_quality.cyclomatic_complexity}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: '60%' }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Maintainability Index</span>
                      <span className="font-medium">{performanceData.code_quality.maintainability_index}/100</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: `${performanceData.code_quality.maintainability_index}%` }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Technical Debt</span>
                      <span className="font-medium">{performanceData.code_quality.technical_debt}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-yellow-500 h-2 rounded-full"
                        style={{ width: '40%' }}
                      />
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Performance Metrics</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <span className="text-sm font-medium text-blue-800">Response Time</span>
                    <span className="text-lg font-bold text-blue-900">{performanceData.performance_metrics.response_time}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                    <span className="text-sm font-medium text-green-800">Throughput</span>
                    <span className="text-lg font-bold text-green-900">{performanceData.performance_metrics.throughput}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                    <span className="text-sm font-medium text-purple-800">Resource Usage</span>
                    <span className="text-lg font-bold text-purple-900">{performanceData.performance_metrics.resource_usage}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
