import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { 
  ArrowLeft, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Info,
  Download,
  FileText,
  Code,
  Zap,
  GitBranch,
  Calendar,
  ExternalLink,
  Search,
  Eye,
  EyeOff
} from 'lucide-react'
import { jobsApi, reportsApi } from '@/lib/api'
import { cn, formatDate, formatDuration, getSeverityColor, getToolIcon, getSeverityIcon } from '@/lib/utils'
import { toast } from 'sonner'


export function JobDetail() {
  const { id: jobId } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<'overview' | 'findings' | 'metrics' | 'files'>('overview')
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [toolFilter, setToolFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [showRawFindings, setShowRawFindings] = useState(false)

  // Fetch job details
  const { data: job, isLoading: jobLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: async () => {
      const response = await jobsApi.get(jobId!)
      return response.data
    },
    enabled: !!jobId,
  })

  // Fetch job findings
  const { data: findings } = useQuery({
    queryKey: ['job-findings', jobId],
    queryFn: async () => {
      const response = await reportsApi.getJobFindings(jobId!)
      return response.data
    },
    enabled: !!jobId,
  })

  // Fetch detailed report
  const { data: detailedReport } = useQuery({
    queryKey: ['job-report', jobId],
    queryFn: async () => {
      const response = await reportsApi.getDetailedReport(jobId!)
      return response.data
    },
    enabled: !!jobId,
  })

  if (jobLoading) {
    return (
      <div className="text-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        <p className="mt-2 text-gray-500">Loading job details...</p>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="text-center py-8">
        <XCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Job Not Found</h2>
        <p className="text-gray-600 mb-4">The requested job could not be found.</p>
        <button
          onClick={() => navigate('/jobs')}
          className="btn-primary"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Jobs
        </button>
      </div>
    )
  }

  const filteredFindings = findings?.findings?.filter((finding: any) => {
    const matchesSeverity = severityFilter === 'all' || finding.severity === severityFilter
    const matchesTool = toolFilter === 'all' || finding.tool === toolFilter
    const matchesSearch = !searchQuery || 
      finding.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      finding.file_path.toLowerCase().includes(searchQuery.toLowerCase())
    
    return matchesSeverity && matchesTool && matchesSearch
  }) || []

  const severityCounts = findings?.findings?.reduce((acc: any, finding: any) => {
    acc[finding.severity] = (acc[finding.severity] || 0) + 1
    return acc
  }, {}) || {}

  const toolCounts = findings?.findings?.reduce((acc: any, finding: any) => {
    acc[finding.tool] = (acc[finding.tool] || 0) + 1
    return acc
  }, {}) || {}

  const handleDownloadReport = async (format: 'json' | 'csv' | 'pdf') => {
    try {
      const response = await reportsApi.download(jobId!, format)
      // Handle download based on response type
      if (format === 'json' || format === 'csv') {
        const blob = new Blob([response.data], { 
          type: format === 'json' ? 'application/json' : 'text/csv' 
        })
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `job-${jobId}-report.${format}`
        a.click()
        window.URL.revokeObjectURL(url)
      }
      toast.success(`${format.toUpperCase()} report downloaded successfully`)
    } catch (error) {
      toast.error(`Failed to download ${format.toUpperCase()} report`)
      console.error('Download error:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/jobs')}
            className="btn-outline btn-sm"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Jobs
          </button>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Job Details</h1>
            <p className="text-gray-600">Analysis results for {job.repo_url}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => handleDownloadReport('json')}
            className="btn-outline btn-sm"
          >
            <Download className="h-4 w-4 mr-2" />
            JSON
          </button>
          <button
            onClick={() => handleDownloadReport('csv')}
            className="btn-outline btn-sm"
          >
            <Download className="h-4 w-4 mr-2" />
            CSV
          </button>
        </div>
      </div>

      {/* Job Status Card */}
      <div className="card">
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{job.id}</div>
              <div className="text-sm text-gray-600">Job ID</div>
            </div>
            
            <div className="text-center">
              <div className={cn(
                "inline-flex items-center px-3 py-1 rounded-full text-sm font-medium",
                job.status === 'completed' ? 'bg-green-100 text-green-800' :
                job.status === 'running' ? 'bg-blue-100 text-blue-800' :
                job.status === 'failed' ? 'bg-red-100 text-red-800' :
                'bg-gray-100 text-gray-800'
              )}>
                {job.status === 'completed' && <CheckCircle className="h-4 w-4 mr-1" />}
                {job.status === 'running' && <Clock className="h-4 w-4 mr-1" />}
                {job.status === 'failed' && <XCircle className="h-4 w-4 mr-1" />}
                {job.status === 'pending' && <Clock className="h-4 w-4 mr-1" />}
                {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Status</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {findings?.findings?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Total Findings</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {'Full Repository'}
              </div>
              <div className="text-sm text-gray-600">Analysis Type</div>
            </div>
          </div>
        </div>
      </div>

      {/* Repository Info */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Repository Information</h2>
        </div>
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Repository URL</label>
              <div className="flex items-center space-x-2">
                <GitBranch className="h-4 w-4 text-gray-400" />
                <a
                  href={job.repo_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 break-all"
                >
                  {job.repo_url}
                </a>
                <ExternalLink className="h-3 w-3 text-gray-400" />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Created</label>
              <div className="flex items-center space-x-2">
                <Calendar className="h-4 w-4 text-gray-400" />
                <span>{formatDate(job.created_at)}</span>
              </div>
            </div>
            
            {job.completed_at && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Completed</label>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-gray-400" />
                  <span>{formatDate(job.completed_at)}</span>
                </div>
              </div>
            )}
            
            {job.completed_at && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Duration</label>
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-gray-400" />
                  <span>{formatDuration(new Date(job.created_at), new Date(job.completed_at))}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', name: 'Overview', icon: Info },
            { id: 'findings', name: 'Findings', icon: AlertTriangle },
            { id: 'metrics', name: 'Metrics', icon: Code },
            { id: 'files', name: 'Files', icon: FileText },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                'flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm',
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              )}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.name}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Summary Statistics */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Findings Summary</h2>
              <p className="card-description">
                Overview of security and quality issues found
              </p>
            </div>
            <div className="card-content">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                {Object.entries(severityCounts).map(([severity, count]) => (
                  <div key={severity} className="text-center">
                    <div className={cn(
                      "text-3xl font-bold mb-2",
                      getSeverityColor(severity)
                    )}>
                      {count as number}
                    </div>
                    <div className="flex items-center justify-center space-x-2">
                      {getSeverityIcon(severity)}
                      <span className="text-sm font-medium text-gray-700 capitalize">
                        {severity}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Tool Breakdown */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Analysis Tools</h2>
              <p className="card-description">
                Issues found by each security and quality analysis tool
              </p>
            </div>
            <div className="card-content">
              <div className="space-y-4">
                {Object.entries(toolCounts).map(([tool, count]) => (
                  <div key={tool} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getToolIcon(tool)}
                      <span className="font-medium text-gray-900">{tool}</span>
                    </div>
                    <div className="text-2xl font-bold text-blue-600">{count as number}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk Assessment */}
          {detailedReport?.risk_assessment && (
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Risk Assessment</h2>
                <p className="card-description">
                  AI-powered risk analysis and recommendations
                </p>
              </div>
              <div className="card-content">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className={cn(
                      "text-3xl font-bold mb-2",
                      detailedReport.risk_assessment.overall_risk_score > 7 ? 'text-red-600' :
                      detailedReport.risk_assessment.overall_risk_score > 4 ? 'text-orange-600' :
                      'text-green-600'
                    )}>
                      {detailedReport.risk_assessment.overall_risk_score}/10
                    </div>
                    <div className="text-sm text-gray-600">Overall Risk Score</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600 mb-2">
                      {detailedReport.risk_assessment.critical_issues_count || 0}
                    </div>
                    <div className="text-sm text-gray-600">Critical Issues</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-600 mb-2">
                      {detailedReport.risk_assessment.recommendations?.length || 0}
                    </div>
                    <div className="text-sm text-gray-600">Recommendations</div>
                  </div>
                </div>
                
                {detailedReport.risk_assessment.recommendations && (
                  <div className="mt-6">
                    <h4 className="font-medium text-gray-900 mb-3">Key Recommendations</h4>
                    <div className="space-y-2">
                      {detailedReport.risk_assessment.recommendations.slice(0, 3).map((rec: any, index: number) => (
                        <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                          <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                          <p className="text-sm text-blue-800">{rec}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'findings' && (
        <div className="space-y-6">
          {/* Filters */}
          <div className="card">
            <div className="card-content">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search findings..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="input w-full pl-10"
                    />
                  </div>
                </div>
                
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="input"
                >
                  <option value="all">All Severities</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                
                <select
                  value={toolFilter}
                  onChange={(e) => setToolFilter(e.target.value)}
                  className="input"
                >
                  <option value="all">All Tools</option>
                  {Object.keys(toolCounts).map((tool) => (
                    <option key={tool} value={tool}>{tool}</option>
                  ))}
                </select>
                
                <button
                  onClick={() => setShowRawFindings(!showRawFindings)}
                  className="btn-outline"
                >
                  {showRawFindings ? <EyeOff className="h-4 w-4 mr-2" /> : <Eye className="h-4 w-4 mr-2" />}
                  {showRawFindings ? 'Hide Raw' : 'Show Raw'}
                </button>
              </div>
            </div>
          </div>

          {/* Findings List */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">
                Findings ({filteredFindings.length})
              </h2>
              <p className="card-description">
                Security and quality issues found during analysis
              </p>
            </div>
            <div className="card-content">
              {filteredFindings.length > 0 ? (
                <div className="space-y-4">
                  {filteredFindings.map((finding: any) => (
                    <div key={finding.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            {getSeverityIcon(finding.severity)}
                            <span className={cn(
                              "px-2 py-1 rounded-full text-xs font-medium",
                              getSeverityColor(finding.severity)
                            )}>
                              {finding.severity.toUpperCase()}
                            </span>
                            <span className="text-sm text-gray-500">{finding.tool}</span>
                          </div>
                          
                          <h3 className="font-medium text-gray-900 mb-2">
                            {finding.message}
                          </h3>
                          
                          <div className="text-sm text-gray-600 mb-3">
                            <span className="font-medium">File:</span> {finding.file_path}
                            {finding.line_number && (
                              <span className="ml-2">
                                <span className="font-medium">Line:</span> {finding.line_number}
                              </span>
                            )}
                          </div>
                          
                          {finding.description && (
                            <p className="text-sm text-gray-700 mb-3">
                              {finding.description}
                            </p>
                          )}
                          
                          {showRawFindings && finding.raw_data && (
                            <details className="mt-3">
                              <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                                Raw Data
                              </summary>
                              <pre className="mt-2 p-3 bg-gray-50 rounded text-xs overflow-x-auto">
                                {JSON.stringify(finding.raw_data, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No findings match the current filters</p>
                  <p className="text-gray-400 mt-1">Try adjusting your search or filter criteria</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'metrics' && (
        <div className="space-y-6">
          {/* Code Quality Metrics */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Code Quality Metrics</h2>
              <p className="card-description">
                Automated analysis of code structure and quality
              </p>
            </div>
            <div className="card-content">
              {detailedReport?.code_quality_metrics ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600">
                      {detailedReport.code_quality_metrics.complexity_score || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Complexity Score</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600">
                      {detailedReport.code_quality_metrics.maintainability_index || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Maintainability Index</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-600">
                      {detailedReport.code_quality_metrics.test_coverage || 'N/A'}%
                    </div>
                    <div className="text-sm text-gray-600">Test Coverage</div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Code className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Code quality metrics not available</p>
                  <p className="text-gray-400 mt-1">Run a comprehensive analysis to get detailed metrics</p>
                </div>
              )}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Performance Metrics</h2>
              <p className="card-description">
                Analysis of performance-related issues and bottlenecks
              </p>
            </div>
            <div className="card-content">
              {detailedReport?.performance_metrics ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-orange-600">
                        {detailedReport.performance_metrics.performance_score || 'N/A'}
                      </div>
                      <div className="text-sm text-gray-600">Performance Score</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-3xl font-bold text-red-600">
                        {detailedReport.performance_metrics.bottlenecks_count || 0}
                      </div>
                      <div className="text-sm text-gray-600">Bottlenecks Found</div>
                    </div>
                  </div>
                  
                  {detailedReport.performance_metrics.recommendations && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Performance Recommendations</h4>
                      <div className="space-y-2">
                        {detailedReport.performance_metrics.recommendations.slice(0, 3).map((rec: any, index: number) => (
                          <div key={index} className="flex items-start space-x-3 p-3 bg-orange-50 rounded-lg">
                            <Zap className="h-4 w-4 text-orange-600 mt-0.5" />
                            <p className="text-sm text-orange-800">{rec}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Zap className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Performance metrics not available</p>
                  <p className="text-gray-400 mt-1">Run a performance analysis to get detailed metrics</p>
                </div>
              )}
            </div>
          </div>

          {/* Security Metrics */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Security Metrics</h2>
              <p className="card-description">
                Security analysis results and vulnerability assessment
              </p>
            </div>
            <div className="card-content">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-red-600">
                    {severityCounts.critical || 0}
                  </div>
                  <div className="text-sm text-gray-600">Critical</div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold text-orange-600">
                    {severityCounts.high || 0}
                  </div>
                  <div className="text-sm text-gray-600">High</div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold text-yellow-600">
                    {severityCounts.medium || 0}
                  </div>
                  <div className="text-sm text-gray-600">Medium</div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">
                    {severityCounts.low || 0}
                  </div>
                  <div className="text-sm text-gray-600">Low</div>
                </div>
              </div>
              
              {detailedReport?.security_metrics && (
                <div className="mt-6">
                  <h4 className="font-medium text-gray-900 mb-3">Security Assessment</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="text-center">
                      <div className={cn(
                        "text-2xl font-bold mb-2",
                        detailedReport.security_metrics.owasp_score > 7 ? 'text-red-600' :
                        detailedReport.security_metrics.owasp_score > 4 ? 'text-orange-600' :
                        'text-green-600'
                      )}>
                        {detailedReport.security_metrics.owasp_score}/10
                      </div>
                      <div className="text-sm text-gray-600">OWASP Score</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600 mb-2">
                        {detailedReport.security_metrics.vulnerability_types?.length || 0}
                      </div>
                      <div className="text-sm text-gray-600">Vulnerability Types</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'files' && (
        <div className="space-y-6">
          {/* Files Analyzed */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Files Analyzed</h2>
              <p className="card-description">
                Overview of files processed during the analysis
              </p>
            </div>
            <div className="card-content">
              {detailedReport?.files_analyzed ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600">
                        {detailedReport.files_analyzed.total_files || 0}
                      </div>
                      <div className="text-sm text-gray-600">Total Files</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-3xl font-bold text-green-600">
                        {detailedReport.files_analyzed.languages?.length || 0}
                      </div>
                      <div className="text-sm text-gray-600">Languages</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-3xl font-bold text-purple-600">
                        {detailedReport.files_analyzed.total_lines || 0}
                      </div>
                      <div className="text-sm text-gray-600">Total Lines</div>
                    </div>
                  </div>
                  
                  {detailedReport.files_analyzed.languages && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Languages Detected</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {detailedReport.files_analyzed.languages.map((lang: any) => (
                          <div key={lang.name} className="text-center p-3 border border-gray-200 rounded-lg">
                            <div className="text-lg font-semibold text-gray-900">{lang.name}</div>
                            <div className="text-sm text-gray-600">{lang.files_count} files</div>
                            <div className="text-xs text-gray-500">{lang.lines_count} lines</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">File analysis details not available</p>
                  <p className="text-gray-400 mt-1">Run a comprehensive analysis to get file details</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
