import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Plus, 
  GitBranch, 
  Shield, 
  TestTube, 
  Zap, 
  Github, 
  BarChart3,
  FileText,
  Workflow,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  Code,
  Settings
} from 'lucide-react'
import { toast } from 'sonner'
import { jobsApi, analyticsApi, actionsApi } from '@/lib/api'
import { cn, formatDate } from '@/lib/utils'



export function Dashboard() {
  const [repoUrl, setRepoUrl] = useState('')
  const [prNumber, setPrNumber] = useState('')
  const [analysisType, setAnalysisType] = useState<'full' | 'pr' | 'security' | 'test' | 'performance' | 'comprehensive'>('full')

  // Fetch only the most recent job
  const { data: jobsData, error: jobsError, refetch: refetchJobs } = useQuery({
    queryKey: ['jobs'],
    queryFn: async () => {
      const response = await jobsApi.list(1, 1) // Only get 1 job
      return response.data
    },
    retry: 3,
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  const { data: analyticsData, error: analyticsError, refetch: refetchAnalytics } = useQuery({
    queryKey: ['analytics-trends'],
    queryFn: async () => {
      const response = await analyticsApi.trends()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: 3,
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  const { data: mlInsights, error: mlError, refetch: refetchML } = useQuery({
    queryKey: ['ml-insights'],
    queryFn: async () => {
      const response = await analyticsApi.mlInsights()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: 3,
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  const handleAnalysis = async () => {
    if (!repoUrl.trim()) {
      toast.error('Please enter a repository URL')
      return
    }

    try {
      let response
      switch (analysisType) {
        case 'full':
          response = await actionsApi.createJob({ repo_url: repoUrl })
          break
        case 'pr':
          if (!prNumber.trim()) {
            toast.error('Please enter a PR number for PR analysis')
            return
          }
          response = await actionsApi.analyzePR({ 
            repo_url: repoUrl, 
            pr_number: parseInt(prNumber) 
          })
          break
        case 'security':
          response = await actionsApi.securityAnalysis({ repo_url: repoUrl })
          break
        case 'test':
          response = await actionsApi.generateTestPlan({ repo_url: repoUrl })
          break
        case 'performance':
          response = await actionsApi.performanceAnalysis({ repo_url: repoUrl })
          break
        case 'comprehensive':
          response = await actionsApi.comprehensiveAnalysis({ 
            repo_url: repoUrl,
            include_performance: true,
            include_api_analysis: true,
            include_test_generation: true
          })
          break
      }

      if (response?.data?.job_id) {
        toast.success(`${analysisType === 'full' ? 'Repository' : analysisType} analysis started! Job ID: ${response.data.job_id}`)
        setRepoUrl('')
        setPrNumber('')
        // Invalidate queries to refresh data
        window.location.reload() // Force refresh to show new job
      }
    } catch (error) {
      toast.error('Failed to start analysis. Please try again.')
      console.error('Analysis error:', error)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'running':
        return <Clock className="h-5 w-5 text-blue-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <AlertCircle className="h-5 w-5 text-yellow-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'running':
        return 'bg-blue-100 text-blue-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-yellow-100 text-yellow-800'
    }
  }

  // Format timestamp to local time
  const formatLocalTime = (timestamp: string | Date) => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
      })
    } catch (error) {
      return 'Invalid date'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Code Review Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Enhanced Multi-User Code Review Agent - Comprehensive code analysis and security scanning
        </p>
      </div>



      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Repository Analysis */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Start New Analysis</h2>
            <p className="card-description">
              Analyze a GitHub repository for security, quality, and performance issues
            </p>
          </div>
          <div className="card-content space-y-4">
            <div>
              <label htmlFor="repo-url" className="block text-sm font-medium text-gray-700 mb-2">
                Repository URL
              </label>
              <input
                id="repo-url"
                type="url"
                placeholder="https://github.com/owner/repo"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                className="input w-full"
              />
            </div>

            {analysisType === 'pr' && (
              <div>
                <label htmlFor="pr-number" className="block text-sm font-medium text-gray-700 mb-2">
                  PR Number
                </label>
                <input
                  id="pr-number"
                  type="number"
                  placeholder="123"
                  value={prNumber}
                  onChange={(e) => setPrNumber(e.target.value)}
                  className="input w-full"
                />
              </div>
            )}

            <div>
              <label htmlFor="analysis-type" className="block text-sm font-medium text-gray-700 mb-2">
                Analysis Type
              </label>
              <select
                id="analysis-type"
                value={analysisType}
                onChange={(e) => setAnalysisType(e.target.value as 'full' | 'pr' | 'security' | 'test' | 'performance' | 'comprehensive')}
                className="input w-full"
              >
                <option value="full">Full Repository Analysis</option>
                <option value="pr">PR/Diff Analysis</option>
                <option value="security">Security Analysis (OWASP Top 10)</option>
                <option value="test">Test Plan Generation</option>
                <option value="performance">Performance Analysis</option>
                <option value="comprehensive">Comprehensive Analysis</option>
              </select>
            </div>

            <button
              onClick={handleAnalysis}
              className="btn-primary w-full"
            >
              <Plus className="h-4 w-4 mr-2" />
              Start Analysis
            </button>
          </div>
        </div>

        {/* Error Handling */}
        {((jobsError && jobsError instanceof Error) || (analyticsError && analyticsError instanceof Error) || (mlError && mlError instanceof Error)) && (
          <div className="card border-red-200 bg-red-50">
            <div className="card-content">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-5 w-5 text-red-600" />
                  <span className="text-red-800 font-medium">Connection Issues Detected</span>
                </div>
                <div className="flex space-x-2">
                  {jobsError && jobsError instanceof Error && (
                    <button
                      onClick={() => refetchJobs()}
                      className="btn-outline btn-sm text-red-700 border-red-300 hover:bg-red-100"
                    >
                      Retry Jobs
                    </button>
                  )}
                  {analyticsError && analyticsError instanceof Error && (
                    <button
                      onClick={() => refetchAnalytics()}
                      className="btn-outline btn-sm text-red-700 border-red-300 hover:bg-red-100"
                    >
                      Retry Analytics
                    </button>
                  )}
                  {mlError && mlError instanceof Error && (
                    <button
                      onClick={() => refetchML()}
                      className="btn-outline btn-sm text-red-700 border-red-300 hover:bg-red-100"
                    >
                      Retry ML Insights
                    </button>
                  )}
                </div>
              </div>
              <p className="text-red-700 text-sm mt-2">
                Some data couldn't be loaded. Click the retry buttons above to reload.
              </p>
            </div>
          </div>
        )}

        {/* Code Review Agent Quick Actions */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Code Review Agent</h2>
            <p className="card-description">
              AI-powered code quality analysis and refactoring suggestions
            </p>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 gap-3">
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-green-50">
                <Code className="h-5 w-5 text-green-600" />
                <div>
                  <h3 className="font-medium text-green-900">Code Quality Analysis</h3>
                  <p className="text-sm text-green-700">Identify code smells, complexity issues, and refactoring opportunities</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-purple-50">
                <GitBranch className="h-5 w-5 text-purple-600" />
                <div>
                  <h3 className="font-medium text-purple-900">Reusable Methods</h3>
                  <p className="text-sm text-purple-700">Suggest extraction of reusable functions and methods</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-orange-50">
                <Zap className="h-5 w-5 text-orange-600" />
                <div>
                  <h3 className="font-medium text-orange-900">Performance Optimization</h3>
                  <p className="text-sm text-orange-700">Identify inefficient code patterns and suggest improvements</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-indigo-50">
                <Settings className="h-5 w-5 text-indigo-600" />
                <div>
                  <h3 className="font-medium text-indigo-900">Hard-coded Values</h3>
                  <p className="text-sm text-indigo-700">Detect and suggest configuration for hard-coded values</p>
                </div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-200">
              <a
                href="/code-review"
                className="btn-outline w-full text-center"
              >
                <Code className="h-4 w-4 mr-2" />
                Go to Code Review Agent
              </a>
            </div>
          </div>
        </div>

        {/* Feature Overview */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Agent Capabilities</h2>
            <p className="card-description">
              What our enhanced code review agent can do for you
            </p>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 gap-3">
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-blue-50">
                <Shield className="h-5 w-5 text-blue-600" />
                <div>
                  <h3 className="font-medium text-blue-900">Security Analysis</h3>
                  <p className="text-sm text-blue-700">OWASP Top 10, SQL injection, XSS, secrets detection</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-green-50">
                <GitBranch className="h-5 w-5 text-green-600" />
                <div>
                  <h3 className="font-medium text-green-900">PR Analysis</h3>
                  <p className="text-sm text-green-700">Focused analysis on pull request changes</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-purple-50">
                <TestTube className="h-5 w-5 text-purple-600" />
                <div>
                  <h3 className="font-medium text-purple-900">Test Generation</h3>
                  <p className="text-sm text-purple-700">Automatic test plans and test cases</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-orange-50">
                <Zap className="h-5 w-5 text-orange-600" />
                <div>
                  <h3 className="font-medium text-orange-900">Performance Analysis</h3>
                  <p className="text-sm text-orange-700">Code optimization and performance insights</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50">
                <Github className="h-5 w-5 text-gray-600" />
                <div>
                  <h3 className="font-medium text-gray-900">GitHub Integration</h3>
                  <p className="text-sm text-gray-700">Direct PR comments and comprehensive reviews</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Statistics and Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Total Findings */}
        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <BarChart3 className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Findings</p>
                <p className="text-2xl font-bold text-gray-900">
                  {analyticsData?.total_findings || 0}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Prediction */}
        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <AlertCircle className="h-6 w-6 text-yellow-600" />
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

        {/* Recent Jobs */}
        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <Workflow className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Recent Jobs</p>
                <p className="text-2xl font-bold text-gray-900">
                  {jobsData?.total || 0}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Current/Recent Job */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Current Analysis Job</h2>
          <p className="card-description">
            {jobsData?.items && jobsData.items.length > 0 
              ? 'Latest code review and analysis activity' 
              : 'No analysis jobs yet. Start your first analysis above!'}
          </p>
        </div>
        <div className="card-content">
          {jobsData?.items && jobsData.items.length > 0 ? (
            <div className="space-y-4">
              {jobsData.items.map((job) => (
                <div key={job.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(job.status)}
                    <div>
                      <h3 className="font-medium text-gray-900">
                        {job.repo_url.split('/').slice(-2).join('/')}
                      </h3>
                      <p className="text-sm text-gray-500">
                        {job.is_pr_analysis ? `PR #${job.pr_number}` : 'Full Repository'} • {formatLocalTime(job.created_at)}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={cn("px-2 py-1 text-xs font-medium rounded-full", getStatusColor(job.status))}>
                      {job.status}
                    </span>
                    {job.findings_count > 0 && (
                      <span className="text-sm text-gray-500">
                        {job.findings_count} findings
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No analysis jobs yet. Start your first analysis above!</p>
            </div>
          )}
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
              </div>
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Quality Scores</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Security</span>
                      <span className="font-medium">{mlInsights.trends.security_score}/10</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-red-500 h-2 rounded-full" 
                        style={{ width: `${(mlInsights.trends.security_score / 10) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Quality</span>
                      <span className="font-medium">{mlInsights.trends.quality_score}/10</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-yellow-500 h-2 rounded-full" 
                        style={{ width: `${(mlInsights.trends.quality_score / 10) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Performance</span>
                      <span className="font-medium">{mlInsights.trends.performance_score}/10</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full" 
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
    </div>
  )
}
