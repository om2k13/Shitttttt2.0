import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { 
  Plus, 
  Search, 
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  GitBranch,
  Shield,
  TestTube,
  Eye,
  Download
} from 'lucide-react'
import { jobsApi } from '@/lib/api'
import { cn, formatDate } from '@/lib/utils'
import type { Job } from '@/types'

export function Jobs() {
  const [page, setPage] = useState(1)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [analysisTypeFilter, setAnalysisTypeFilter] = useState<string>('all')

  const { data: jobsData, isLoading } = useQuery({
    queryKey: ['jobs', page, searchTerm, statusFilter, analysisTypeFilter],
    queryFn: async () => {
      const response = await jobsApi.list(page, 10) // Limit to 10 recent jobs
      return response.data
    },
  })

  // Remove duplicate jobs based on repository URL, keeping only the most recent
  const uniqueJobs = jobsData?.items?.reduce((acc: Job[], job: Job) => {
    const existingJob = acc.find(j => j.repo_url === job.repo_url)
    if (!existingJob) {
      acc.push(job)
    } else if (new Date(job.created_at) > new Date(existingJob.created_at)) {
      // Replace with more recent job
      const index = acc.findIndex(j => j.repo_url === job.repo_url)
      acc[index] = job
    }
    return acc
  }, []) || []

  const filteredJobs = uniqueJobs.filter((job: Job) => {
    const matchesSearch = job.repo_url.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === 'all' || job.status === statusFilter
    const matchesType = analysisTypeFilter === 'all' || 
      (analysisTypeFilter === 'pr' && job.is_pr_analysis) ||
      (analysisTypeFilter === 'full' && !job.is_pr_analysis)
    
    return matchesSearch && matchesStatus && matchesType
  })

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

  const getAnalysisTypeIcon = (isPrAnalysis: boolean) => {
    return isPrAnalysis ? (
      <GitBranch className="h-4 w-4 text-blue-500" />
    ) : (
      <Shield className="h-4 w-4 text-green-500" />
    )
  }

  const getAnalysisTypeLabel = (isPrAnalysis: boolean) => {
    return isPrAnalysis ? 'PR Analysis' : 'Full Repository'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analysis Jobs</h1>
          <p className="mt-2 text-gray-600">
            Monitor and manage your code review and analysis jobs
          </p>
        </div>
        <Link
          to="/"
          className="btn-primary"
        >
          <Plus className="h-4 w-4 mr-2" />
          New Analysis
        </Link>
      </div>

      {/* Filters and Search */}
      <div className="card">
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label htmlFor="search" className="block text-sm font-medium text-gray-700 mb-2">
                Search Repositories
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  id="search"
                  type="text"
                  placeholder="Search by repository URL..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="input pl-10 w-full"
                />
              </div>
            </div>

            <div>
              <label htmlFor="status-filter" className="block text-sm font-medium text-gray-700 mb-2">
                Status
              </label>
              <select
                id="status-filter"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="input w-full"
              >
                <option value="all">All Statuses</option>
                <option value="pending">Pending</option>
                <option value="queued">Queued</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
            </div>

            <div>
              <label htmlFor="type-filter" className="block text-sm font-medium text-gray-700 mb-2">
                Analysis Type
              </label>
              <select
                id="type-filter"
                value={analysisTypeFilter}
                onChange={(e) => setAnalysisTypeFilter(e.target.value)}
                className="input w-full"
              >
                <option value="all">All Types</option>
                <option value="full">Full Repository</option>
                <option value="pr">PR Analysis</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Total Jobs
              </label>
              <div className="text-2xl font-bold text-gray-900">
                {jobsData?.total || 0}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Jobs List */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Analysis Jobs</h2>
          <p className="card-description">
            {filteredJobs.length} jobs found
          </p>
        </div>
        <div className="card-content">
          {isLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-2 text-gray-500">Loading jobs...</p>
            </div>
          ) : filteredJobs.length > 0 ? (
            <div className="space-y-4">
              {filteredJobs.map((job: Job) => (
                <div key={job.id} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-3">
                        {getStatusIcon(job.status)}
                        <div className="flex items-center space-x-2">
                          {getAnalysisTypeIcon(job.is_pr_analysis)}
                          <span className="text-sm text-gray-500">
                            {getAnalysisTypeLabel(job.is_pr_analysis)}
                          </span>
                        </div>
                        <span className={cn("px-2 py-1 text-xs font-medium rounded-full", getStatusColor(job.status))}>
                          {job.status}
                        </span>
                      </div>

                      <h3 className="text-lg font-medium text-gray-900 mb-2">
                        {job.repo_url.split('/').slice(-2).join('/')}
                      </h3>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600 mb-4">
                        <div>
                          <span className="font-medium">Repository:</span>
                          <a 
                            href={job.repo_url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="ml-2 text-blue-600 hover:underline break-all"
                          >
                            {job.repo_url}
                          </a>
                        </div>
                        <div>
                          <span className="font-medium">Branch:</span>
                          <span className="ml-2">{job.base_branch || 'main'}</span>
                        </div>
                        <div>
                          <span className="font-medium">Created:</span>
                          <span className="ml-2">{formatDate(job.created_at)}</span>
                        </div>
                      </div>

                      {job.is_pr_analysis && job.pr_number && (
                        <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                          <div className="flex items-center space-x-2">
                            <GitBranch className="h-4 w-4 text-blue-600" />
                            <span className="text-sm font-medium text-blue-900">
                              Pull Request #{job.pr_number}
                            </span>
                          </div>
                          {job.head_branch && (
                            <p className="text-sm text-blue-700 mt-1">
                              {job.base_branch} → {job.head_branch}
                            </p>
                          )}
                        </div>
                      )}

                      {job.status === 'running' && job.current_stage && (
                        <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                          <div className="flex items-center space-x-2">
                            <Clock className="h-4 w-4 text-blue-600" />
                            <span className="text-sm font-medium text-blue-900">
                              Current Stage: {job.current_stage}
                            </span>
                          </div>
                          {job.progress > 0 && (
                            <div className="mt-2">
                              <div className="flex justify-between text-xs text-blue-700 mb-1">
                                <span>Progress</span>
                                <span>{job.progress}%</span>
                              </div>
                              <div className="w-full bg-blue-200 rounded-full h-2">
                                <div 
                                  className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                                  style={{ width: `${job.progress}%` }}
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {job.findings_count > 0 && (
                        <div className="mb-4 p-3 bg-green-50 rounded-lg">
                          <div className="flex items-center space-x-2">
                            <TestTube className="h-4 w-4 text-green-600" />
                            <span className="text-sm font-medium text-green-900">
                              {job.findings_count} findings discovered
                            </span>
                          </div>
                        </div>
                      )}

                      {job.error_message && (
                        <div className="mb-4 p-3 bg-red-50 rounded-lg">
                          <div className="flex items-center space-x-2">
                            <XCircle className="h-4 w-4 text-red-600" />
                            <span className="text-sm font-medium text-red-900">
                              Error: {job.error_message}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="flex flex-col space-y-2 ml-4">
                      <Link
                        to={`/jobs/${job.id}`}
                        className="btn-outline btn-sm"
                      >
                        <Eye className="h-4 w-4 mr-2" />
                        View Details
                      </Link>
                      
                      {job.status === 'completed' && (
                        <Link
                          to={`/reports/${job.id}`}
                          className="btn-outline btn-sm"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          View Report
                        </Link>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <TestTube className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No jobs found matching your criteria.</p>
              <p className="text-gray-400 mt-1">Try adjusting your filters or start a new analysis.</p>
            </div>
          )}
        </div>
      </div>

      {/* Pagination */}
      {jobsData && jobsData.total > 20 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {((page - 1) * 20) + 1} to {Math.min(page * 20, jobsData.total)} of {jobsData.total} results
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="btn-outline btn-sm disabled:opacity-50"
            >
              Previous
            </button>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page * 20 >= jobsData.total}
              className="btn-outline btn-sm disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
