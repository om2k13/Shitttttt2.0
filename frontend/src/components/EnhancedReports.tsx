import { useState } from 'react'
import { 
  Wrench, 
  GitBranch, 
  GitCommit, 
  GitPullRequest, 
  CheckCircle, 
  Info,
  Code,
  Zap,
  Clock,
  TrendingUp
} from 'lucide-react'
import { enhancedActionsApi } from '@/lib/api'
import { cn, getSeverityColor } from '@/lib/utils'
import type { EnhancedAnalysis, FixSummary, AutoFixRequest } from '@/types'
import { toast } from 'sonner'

interface EnhancedReportsProps {
  jobId: string
  onAnalysisComplete?: (analysis: EnhancedAnalysis) => void
}

export function EnhancedReports({ jobId, onAnalysisComplete }: EnhancedReportsProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isApplyingFixes, setIsApplyingFixes] = useState(false)
  const [isCommitting, setIsCommitting] = useState(false)
  const [analysis, setAnalysis] = useState<EnhancedAnalysis | null>(null)
  const [fixSummary, setFixSummary] = useState<FixSummary | null>(null)
  const [selectedFixTypes, setSelectedFixTypes] = useState<('security' | 'code_quality' | 'type_checking')[]>([
    'security', 'code_quality'
  ])
  const [autoCommit, setAutoCommit] = useState(false)
  const [autoPush, setAutoPush] = useState(false)
  const [autoCreatePR, setAutoCreatePR] = useState(false)
  const [commitMessage, setCommitMessage] = useState('')
  const [branchName, setBranchName] = useState('')

  const handleEnhancedAnalysis = async () => {
    if (!jobId) {
      toast.error('No job ID available for analysis')
      return
    }

    setIsAnalyzing(true)
    try {
      const response = await enhancedActionsApi.getEnhancedAnalysis(jobId)
      setAnalysis(response.data.analysis)
      onAnalysisComplete?.(response.data.analysis)
      toast.success('Enhanced analysis completed successfully')
    } catch (error) {
      console.error('Enhanced analysis error:', error)
      toast.error('Failed to perform enhanced analysis')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleApplyFixes = async () => {
    if (!jobId) {
      toast.error('No job ID available for applying fixes')
      return
    }

    setIsApplyingFixes(true)
    try {
      const request: AutoFixRequest = {
        job_id: jobId,
        fix_types: selectedFixTypes,
        create_commit: false,
        push_changes: false,
        create_pr: false
      }

      const response = await enhancedActionsApi.analyzeAndFix(jobId, request)
      setFixSummary(response.data)
      toast.success(`Applied ${response.data.total_fixes_applied} fixes successfully`)
    } catch (error) {
      console.error('Apply fixes error:', error)
      toast.error('Failed to apply fixes')
    } finally {
      setIsApplyingFixes(false)
    }
  }

  const handleCommitFixes = async () => {
    if (!jobId) {
      toast.error('No job ID available for committing fixes')
      return
    }

    if (!commitMessage.trim()) {
      toast.error('Please provide a commit message')
      return
    }

    setIsCommitting(true)
    try {
      const response = await enhancedActionsApi.fixAndCommit(jobId, {
        create_pr: autoCreatePR,
        commit_message: commitMessage,
        branch_name: branchName
      })
      
      if (response.data.success) {
        toast.success('Fixes committed successfully')
        setFixSummary(response.data)
      } else {
        toast.error('Failed to commit fixes')
      }
    } catch (error) {
      console.error('Commit fixes error:', error)
      toast.error('Failed to commit fixes')
    } finally {
      setIsCommitting(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Enhanced Analysis Controls */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title flex items-center">
            <Code className="h-5 w-5 mr-2 text-blue-600" />
            Enhanced Analysis & Auto-Fix
          </h2>
          <p className="card-description">
            Advanced code analysis with automated fixes and Git integration
          </p>
        </div>
        <div className="card-content space-y-4">
          {/* Analysis Controls */}
          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleEnhancedAnalysis}
              disabled={isAnalyzing}
              className="btn btn-primary"
            >
              {isAnalyzing ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              ) : (
                <Info className="h-4 w-4 mr-2" />
              )}
              {isAnalyzing ? 'Analyzing...' : 'Run Enhanced Analysis'}
            </button>

            <button
              onClick={handleApplyFixes}
              disabled={isApplyingFixes || !analysis}
              className="btn-outline"
            >
              {isApplyingFixes ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2" />
              ) : (
                <Wrench className="h-4 w-4 mr-2" />
              )}
              {isApplyingFixes ? 'Applying Fixes...' : 'Apply Fixes Only'}
            </button>

            <button
              onClick={handleCommitFixes}
              disabled={isCommitting || !fixSummary}
              className="btn-outline"
            >
              {isCommitting ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2" />
              ) : (
                <GitCommit className="h-4 w-4 mr-2" />
              )}
              {isCommitting ? 'Committing...' : 'Apply Fixes & Commit'}
            </button>
          </div>

          {/* Fix Type Selection */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700">Fix Types to Apply:</h3>
            <div className="flex flex-wrap gap-2">
              {[
                { key: 'security', label: 'Security Fixes', icon: Code },
                { key: 'code_quality', label: 'Code Quality', icon: Code },
                { key: 'type_checking', label: 'Type Checking', icon: CheckCircle }
              ].map(({ key, label, icon: Icon }) => (
                <label key={key} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedFixTypes.includes(key as any)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedFixTypes([...selectedFixTypes, key as any])
                      } else {
                        setSelectedFixTypes(selectedFixTypes.filter(t => t !== key))
                      }
                    }}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <Icon className="h-4 w-4 text-gray-600" />
                  <span className="text-sm text-gray-700">{label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Git Integration Options */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700">Git Integration Options:</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={autoCommit}
                    onChange={(e) => setAutoCommit(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">Auto-commit fixes</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={autoPush}
                    onChange={(e) => setAutoPush(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">Auto-push to remote</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={autoCreatePR}
                    onChange={(e) => setAutoCreatePR(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">Create Pull Request</span>
                </label>
              </div>
              <div className="space-y-2">
                <input
                  type="text"
                  placeholder="Commit message"
                  value={commitMessage}
                  onChange={(e) => setCommitMessage(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="text"
                  placeholder="Branch name (optional)"
                  value={branchName}
                  onChange={(e) => setBranchName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Analysis Results */}
      {analysis && analysis.summary ? (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title flex items-center">
              <TrendingUp className="h-5 w-5 mr-2 text-purple-600" />
              Enhanced Analysis Results
            </h2>
            <p className="card-description">
              Detailed findings with risk scores and fix suggestions
            </p>
          </div>
          <div className="card-content space-y-6">
            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{analysis.summary.total_findings}</div>
                <div className="text-sm text-blue-700">Total Findings</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{analysis.summary.auto_fixable}</div>
                <div className="text-sm text-green-700">Auto-Fixable</div>
              </div>
              <div className="text-center p-4 bg-yellow-50 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">{analysis.summary.manual_fixes}</div>
                <div className="text-sm text-yellow-700">Manual Fixes</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{analysis.summary.estimated_fix_time || '2-4 hours'}</div>
                <div className="text-sm text-purple-700">Est. Fix Time</div>
              </div>
            </div>

            {/* Risk Distribution */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-3">Risk Distribution</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                {Object.entries(analysis.summary.risk_distribution).map(([level, count]) => (
                  <div key={level} className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className={cn("text-lg font-bold", getSeverityColor(level))}>
                      {count}
                    </div>
                    <div className="text-sm text-gray-600 capitalize">{level}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Enhanced Findings */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-3">Enhanced Findings</h3>
              <div className="space-y-4">
                {analysis.findings.slice(0, 5).map((finding, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <span className={cn("px-2 py-1 text-xs font-medium rounded-full", getSeverityColor(finding.severity))}>
                          {finding.severity.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-blue-600">
                          {finding.risk_score}
                        </div>
                        <div className="text-xs text-gray-500">Risk Score</div>
                      </div>
                    </div>
                    
                    <h4 className="font-medium text-gray-900 mb-2">{finding.message}</h4>
                    
                    {/* File and Line Info */}
                    <p className="text-sm text-gray-600 mb-3">
                      <span className="font-medium">File:</span> {finding.file || 'N/A'}
                      {finding.line && <span className="ml-4"><span className="font-medium">Line:</span> {finding.line}</span>}
                    </p>

                    {/* Tool and Effort */}
                    <div className="flex items-center space-x-3 mt-2">
                      <span className="px-2 py-1 text-xs font-medium rounded-full bg-gray-100 text-gray-700">
                        {finding.tool}
                      </span>
                      <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-700">
                        {finding.effort || 'medium'} effort
                      </span>
                      {finding.autofixable && (
                        <span className="px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-700">
                          Auto-fixable
                        </span>
                      )}
                    </div>

                    {/* Remediation */}
                    {finding.remediation && (
                      <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                        <h5 className="text-sm font-medium text-blue-700 mb-1">Remediation</h5>
                        <p className="text-sm text-blue-600">{finding.remediation}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            {analysis.recommendations && analysis.recommendations.length > 0 && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Recommendations</h3>
                <div className="space-y-2">
                  {analysis.recommendations.map((rec, index) => (
                    <div key={index} className="flex items-start space-x-2 p-3 bg-blue-50 rounded-lg">
                      <Info className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                      <p className="text-sm text-blue-800">{rec}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="card">
          <div className="card-content text-center p-8">
            <div className="text-gray-500 mb-4">
              <TrendingUp className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium text-gray-700 mb-2">No Analysis Data</h3>
              <p className="text-sm text-gray-500">
                Run enhanced analysis to see detailed findings and recommendations.
              </p>
            </div>
            <button
              onClick={handleEnhancedAnalysis}
              disabled={isAnalyzing}
              className="btn btn-primary"
            >
              {isAnalyzing ? 'Analyzing...' : 'Run Enhanced Analysis'}
            </button>
          </div>
        </div>
      )}

      {/* Fix Summary */}
      {fixSummary && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title flex items-center">
              <CheckCircle className="h-5 w-5 mr-2 text-green-600" />
              Fix Summary
            </h2>
            <p className="card-description">
              Results of applied fixes and Git operations
            </p>
          </div>
          <div className="card-content space-y-4">
            {/* Fix Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{fixSummary.total_fixes_applied}</div>
                <div className="text-sm text-green-700">Fixes Applied</div>
              </div>
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {Object.values(fixSummary.fixes_by_type).reduce((a, b) => a + b, 0)}
                </div>
                <div className="text-sm text-blue-700">Total Changes</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {fixSummary.git_status.staged_files.length}
                </div>
                <div className="text-sm text-purple-700">Files Modified</div>
              </div>
            </div>

            {/* Git Status */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-3">Git Status</h3>
              <div className="bg-gray-50 p-4 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Current Branch:</span>
                  <span className="text-sm text-gray-900">{fixSummary.git_status.current_branch}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Remote Origin:</span>
                  <span className="text-sm text-gray-900">{fixSummary.git_status.remote_origin}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Has Changes:</span>
                  <span className={cn("text-sm font-medium", fixSummary.git_status.has_changes ? "text-green-600" : "text-gray-500")}>
                    {fixSummary.git_status.has_changes ? "Yes" : "No"}
                  </span>
                </div>
              </div>
            </div>

            {/* Validation Results */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-3">Validation Results</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {[
                  { key: 'syntax_check_passed', label: 'Syntax Check', icon: CheckCircle },
                  { key: 'tests_passed', label: 'Tests Passed', icon: CheckCircle },
                  { key: 'build_successful', label: 'Build Successful', icon: CheckCircle }
                ].map(({ key, label, icon: Icon }) => {
                  const passed = fixSummary.validation_results[key as keyof typeof fixSummary.validation_results] as boolean
                  return (
                    <div key={key} className={cn("text-center p-3 rounded-lg", passed ? "bg-green-50" : "bg-red-50")}>
                      <Icon className={cn("h-6 w-6 mx-auto mb-2", passed ? "text-green-600" : "text-red-600")} />
                      <div className={cn("text-sm font-medium", passed ? "text-green-700" : "text-red-700")}>
                        {label}
                      </div>
                      <div className={cn("text-xs", passed ? "text-green-600" : "text-red-600")}>
                        {passed ? "Passed" : "Failed"}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Next Steps */}
            {fixSummary.next_steps.length > 0 && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Next Steps</h3>
                <div className="space-y-2">
                  {fixSummary.next_steps.map((step, index) => (
                    <div key={index} className="flex items-start space-x-2 p-2 bg-blue-50 rounded">
                      <Clock className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                      <p className="text-sm text-blue-700">{step}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
