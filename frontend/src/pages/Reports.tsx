import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { 
  Download, 
  Search, 
  FileText, 
  AlertTriangle,
  Info,
  RefreshCw
} from 'lucide-react'
import { reportsApi } from '@/lib/api'
import { cn, getSeverityColor, getSeverityIcon, getToolIcon } from '@/lib/utils'
import type { Finding } from '@/types'
import { toast } from 'sonner'
import { EnhancedReports } from '@/components/EnhancedReports'

export function Reports() {
  const { id: jobId } = useParams<{ id: string }>()
  const [searchTerm, setSearchTerm] = useState('')
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [toolFilter, setToolFilter] = useState<string>('all')

  const { data: reportsData, isLoading, refetch } = useQuery({
    queryKey: ['reports'],
    queryFn: async () => {
      const response = await reportsApi.list()
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Use job ID from URL if available, otherwise get the latest job ID
  const targetJobId = jobId || (reportsData?.reports ? 
    Object.keys(reportsData.reports)[0] : null)

  // Fetch findings for the target job
  const { data: findingsData } = useQuery({
    queryKey: ['findings', targetJobId],
    queryFn: async () => {
      if (!targetJobId) return null
      const response = await reportsApi.getJobFindings(targetJobId)
      return response.data
    },
    enabled: !!targetJobId,
  })

  // Use findings data if available, otherwise use summary data
  const allFindings: Finding[] = findingsData?.findings || []
  
  // Debug logging
  console.log('Reports Debug:', {
    targetJobId,
    findingsData,
    allFindings: allFindings.length,
    reportsData
  })

  const filteredFindings = allFindings.filter((finding) => {
    try {
      const matchesSearch = 
        (finding.file?.toLowerCase() || '').includes(searchTerm.toLowerCase()) ||
        (finding.message?.toLowerCase() || '').includes(searchTerm.toLowerCase()) ||
        (finding.rule_id?.toLowerCase() || '').includes(searchTerm.toLowerCase())
      
      const matchesSeverity = severityFilter === 'all' || finding.severity === severityFilter
      const matchesTool = toolFilter === 'all' || finding.tool === toolFilter
      
      return matchesSearch && matchesSeverity && matchesTool
    } catch (error) {
      console.error('Error filtering finding:', error, finding)
      return false
    }
  })

  const getSeverityCount = (severity: string) => {
    try {
      return allFindings.filter(f => f.severity === severity).length
    } catch (error) {
      console.error('Error counting severity:', error)
      return 0
    }
  }

  const getToolCount = (tool: string) => {
    try {
      return allFindings.filter(f => f.tool === tool).length
    } catch (error) {
      console.error('Error counting tool:', error)
      return 0
    }
  }

  const handleDownload = async (format: 'json' | 'csv' | 'txt') => {
    if (!targetJobId) {
      toast.error('No job data available for download')
      return
    }

    try {
      const response = await reportsApi.download(targetJobId, format)
      
      // The response.data is now a blob due to responseType: 'blob'
      const blob = response.data
      const mimeType = format === 'json' ? 'application/json' : 
                      format === 'csv' ? 'text/csv' : 'text/plain'
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `report-${targetJobId}.${format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      toast.success(`Report downloaded as ${format.toUpperCase()}`)
    } catch (error) {
      console.error('Download error:', error)
      toast.error('Failed to download report')
    }
  }

  const handleRefresh = async () => {
    try {
      await refetch()
      toast.success('Reports data refreshed')
    } catch (error) {
      toast.error('Failed to refresh reports data')
    }
  }

  if (isLoading) {
    return (
      <div className="text-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-500">Loading reports...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Code Review Reports</h1>
          <p className="mt-2 text-gray-600">
            Comprehensive analysis reports and findings from code reviews
          </p>
        </div>
        <button
          onClick={handleRefresh}
          className="btn-outline"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>

      {/* Error Display */}
      {!findingsData && targetJobId && (
        <div className="card border-red-200 bg-red-50">
          <div className="card-content">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-red-600" />
              <span className="text-red-800 font-medium">No findings data available</span>
            </div>
            <p className="text-red-700 text-sm mt-2">
              Unable to load findings for job {targetJobId}. Please try refreshing the page.
            </p>
          </div>
        </div>
      )}

      {/* Summary of Findings */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <AlertTriangle className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Issues</p>
                <p className="text-2xl font-bold text-gray-900">
                  {allFindings.length || 0}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-red-100 rounded-lg">
                <AlertTriangle className="h-6 w-6 text-red-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Critical</p>
                <p className="text-2xl font-bold text-gray-900">
                  {getSeverityCount('critical')}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-orange-100 rounded-lg">
                <AlertTriangle className="h-6 w-6 text-orange-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">High</p>
                <p className="text-2xl font-bold text-gray-900">
                  {getSeverityCount('high')}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <Info className="h-6 w-6 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Medium</p>
                <p className="text-2xl font-bold text-gray-900">
                  {getSeverityCount('medium')}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Info className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Low</p>
                <p className="text-2xl font-bold text-gray-900">
                  {getSeverityCount('low')}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Search Findings</h2>
          <p className="card-description">
            Filter and search through code review findings
          </p>
        </div>
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-2">
              <label htmlFor="search" className="block text-sm font-medium text-gray-700 mb-2">
                Search by file, message, or rule
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  id="search"
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="input w-full pl-10"
                  placeholder="Search findings..."
                />
              </div>
            </div>

            <div>
              <label htmlFor="severity" className="block text-sm font-medium text-gray-700 mb-2">
                Severity
              </label>
              <select
                id="severity"
                value={severityFilter}
                onChange={(e) => setSeverityFilter(e.target.value)}
                className="input w-full"
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>

            <div>
              <label htmlFor="tool" className="block text-sm font-medium text-gray-700 mb-2">
                Tool
              </label>
              <select
                id="tool"
                value={toolFilter}
                onChange={(e) => setToolFilter(e.target.value)}
                className="input w-full"
              >
                <option value="all">All Tools</option>
                <option value="ruff">Ruff</option>
                <option value="mypy">MyPy</option>
                <option value="bandit">Bandit</option>
                <option value="semgrep">Semgrep</option>
                <option value="radon">Radon</option>
                <option value="pip-audit">Pip Audit</option>
              </select>
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between">
            <p className="text-sm text-gray-600">
              Total Findings: <span className="font-medium">{filteredFindings.length}</span>
            </p>
          </div>
        </div>
      </div>

      {/* Export Report */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Export Report</h2>
          <p className="card-description">
            Download findings in various formats for further analysis
          </p>
        </div>
        <div className="card-content">
          <div className="flex space-x-3">
            <button
              onClick={() => handleDownload('json')}
              disabled={!targetJobId}
              className="btn-outline"
            >
              <Download className="h-4 w-4 mr-2" />
              Export JSON
            </button>
            <button
              onClick={() => handleDownload('csv')}
              disabled={!targetJobId}
              className="btn-outline"
            >
              <Download className="h-4 w-4 mr-2" />
              Export CSV
            </button>
            <button
              onClick={() => handleDownload('txt')}
              disabled={!targetJobId}
              className="btn-outline"
            >
              <Download className="h-4 w-4 mr-2" />
              Export Text
            </button>
          </div>
        </div>
      </div>

      {/* Code Review Findings */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Code Review Findings</h2>
          <p className="card-description">
            {filteredFindings.length} findings found
          </p>
        </div>
        <div className="card-content">
          {filteredFindings.length > 0 ? (
            <div className="space-y-4">
              {filteredFindings.map((finding, index) => (
                <div key={index} className="flex items-start justify-between p-4 border border-gray-200 rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <span className={cn("px-2 py-1 text-xs font-medium rounded-full", getSeverityColor(finding.severity))}>
                        {finding.severity.toUpperCase()}
                      </span>
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{getToolIcon(finding.tool)}</span>
                        <span className="text-sm font-medium text-gray-900">{finding.tool}</span>
                      </div>
                    </div>
                    
                    <h3 className="font-medium text-gray-900 mb-1">
                      {finding.message}
                    </h3>
                    
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><span className="font-medium">File:</span> {finding.file}</p>
                      {finding.line && <p><span className="font-medium">Line:</span> {finding.line}</p>}
                      <p><span className="font-medium">Rule ID:</span> {finding.rule_id}</p>
                      {finding.remediation && (
                        <p><span className="font-medium">Remediation:</span> {finding.remediation}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="ml-4">
                    {finding.autofixable ? (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        Auto-fixable
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                        Manual Fix
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No findings available</h3>
              <p className="text-gray-500 mb-4">
                {targetJobId ? 
                  `No findings found for job ${targetJobId}. The analysis may still be running or no issues were detected.` :
                  'No job selected. Please select a job to view findings.'
                }
              </p>
              {targetJobId && (
                <button
                  onClick={handleRefresh}
                  className="btn-outline"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh Data
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Enhanced Reports Section */}
      {targetJobId && (
        <EnhancedReports 
          jobId={targetJobId}
          onAnalysisComplete={(analysis) => {
            console.log('Enhanced analysis completed:', analysis)
            // Optionally refresh the main findings data
            refetch()
          }}
        />
      )}
    </div>
  )
}
