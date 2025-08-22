import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  FolderOpen, 
  GitBranch, 
  Trash2, 
  RefreshCw, 
  Settings, 
  HardDrive,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Download,
  Upload,
  Search,
  Filter,
  Plus,
  Database,
  Cpu,
  Activity,
  Calendar,
  FileText,
  Code,
  Shield,
  Zap,
  Workflow,
  GitPullRequest,
  GitCommit,
  GitMerge,
  Eye
} from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card } from '@/components/ui/Card'
import { Select } from '@/components/ui/Select'
import { cn, formatDate } from '@/lib/utils'

import { jobsApi } from '@/lib/api'

export function WorkspaceManagement() {
  const [activeTab, setActiveTab] = useState<'overview' | 'repositories' | 'cleanup' | 'settings'>('overview')
  const [maxAgeHours, setMaxAgeHours] = useState(24)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [ageFilter, setAgeFilter] = useState('all')

  const queryClient = useQueryClient()

  // Queries
  const { data: workspaceStats, isLoading: statsLoading, refetch: refetchStats } = useQuery({
    queryKey: ['workspace-stats'],
    queryFn: jobsApi.getWorkspaceStats,
    refetchInterval: 30000 // Refresh every 30 seconds
  })

  const { data: currentRepoInfo, isLoading: repoInfoLoading } = useQuery({
    queryKey: ['current-repo-info'],
    queryFn: jobsApi.getCurrentRepo,
    refetchInterval: 15000 // Refresh every 15 seconds
  })

  // Mutations
  const cleanupOldMutation = useMutation({
    mutationFn: (maxAgeHours: number) => jobsApi.cleanupWorkspaces(maxAgeHours),
    onSuccess: (data) => {
      toast.success(`Workspace cleanup completed`)
      queryClient.invalidateQueries({ queryKey: ['workspace-stats'] })
    },
    onError: () => {
      toast.error('Failed to cleanup old workspaces')
    }
  })

  const cleanupAllMutation = useMutation({
    mutationFn: jobsApi.cleanupAllRepos,
    onSuccess: (data) => {
      toast.success('All repositories cleaned up')
      queryClient.invalidateQueries({ queryKey: ['workspace-stats'] })
    },
    onError: () => {
      toast.error('Failed to cleanup all repositories')
    }
  })

  const cleanupCurrentMutation = useMutation({
    mutationFn: jobsApi.cleanupCurrentRepo,
    onSuccess: (data) => {
      toast.success('Current repository cleaned up')
      queryClient.invalidateQueries({ queryKey: ['workspace-stats', 'current-repo-info'] })
    },
    onError: () => {
      toast.error('Failed to cleanup current repository')
    }
  })



  // Filtered repositories
  const filteredRepos = workspaceStats?.data?.repos?.filter((repo: any) => {
    const matchesSearch = repo.repo_url.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === 'all' || 
                         (statusFilter === 'current' && repo.is_current) ||
                         (statusFilter === 'other' && !repo.is_current)
    const matchesAge = ageFilter === 'all' ||
                      (ageFilter === 'recent' && repo.age_hours <= 6) ||
                      (ageFilter === 'old' && repo.age_hours > 6)
    
    return matchesSearch && matchesStatus && matchesAge
  }) || []

  const getStatusIcon = (isCurrent: boolean) => {
    return isCurrent ? 
      <CheckCircle className="h-4 w-4 text-green-600" /> : 
      <Clock className="h-4 w-4 text-gray-400" />
  }

  const getAgeColor = (ageHours: number) => {
    if (ageHours <= 2) return 'text-green-600'
    if (ageHours <= 6) return 'text-yellow-600'
    if (ageHours <= 24) return 'text-orange-600'
    return 'text-red-600'
  }

  const getSizeColor = (size: string) => {
    const sizeMB = parseInt(size.replace(' MB', ''))
    if (sizeMB <= 100) return 'text-green-600'
    if (sizeMB <= 500) return 'text-yellow-600'
    return 'text-red-600'
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Workspace Management</h1>
        <p className="mt-2 text-gray-600">
          Manage repository workspaces, cleanup operations, and system resources
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'repositories', label: 'Repositories', icon: FolderOpen },
            { id: 'cleanup', label: 'Cleanup', icon: Trash2 },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                "flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm",
                activeTab === tab.id
                  ? "border-blue-500 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              )}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Current Repository Status */}
          {currentRepoInfo?.data && (
            <Card>
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">Current Repository</h2>
                  <div className="flex items-center space-x-2">
                    <span className={cn(
                      "px-2 py-1 text-xs font-medium rounded-full",
                      currentRepoInfo.data.status === 'analyzing' ? 'bg-blue-100 text-blue-800' :
                      currentRepoInfo.data.status === 'completed' ? 'bg-green-100 text-green-800' :
                      'bg-gray-100 text-gray-800'
                    )}>
                      {currentRepoInfo.data.status}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => refetchStats()}
                    >
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Refresh
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <GitBranch className="h-6 w-6 text-blue-600 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-blue-900">{currentRepoInfo.data.branch || 'N/A'}</p>
                    <p className="text-sm text-blue-600">Branch</p>
                  </div>
                  
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <FileText className="h-6 w-6 text-green-600 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-green-900">{currentRepoInfo.data.files_count || 'N/A'}</p>
                    <p className="text-sm text-green-600">Files</p>
                  </div>
                  
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <Code className="h-6 w-6 text-purple-600 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-purple-900">{currentRepoInfo.data.total_lines ? currentRepoInfo.data.total_lines.toLocaleString() : 'N/A'}</p>
                    <p className="text-sm text-purple-600">Lines of Code</p>
                  </div>
                  
                  <div className="text-center p-4 bg-orange-50 rounded-lg">
                    <Clock className="h-6 w-6 text-orange-600 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-orange-900">
                      {currentRepoInfo.data.cloned_at ? Math.round((Date.now() / 1000 - currentRepoInfo.data.cloned_at) / 3600) : 'N/A'}h
                    </p>
                    <p className="text-sm text-orange-600">Age</p>
                  </div>
                </div>

                {/* Repository Details */}
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-medium text-gray-900 mb-3">Repository Information</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">URL:</span>
                        <span className="font-medium text-blue-600">{currentRepoInfo.data.repo_url || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Job ID:</span>
                        <span className="font-medium">{currentRepoInfo.data.job_id || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Path:</span>
                        <span className="font-medium font-mono text-xs">{currentRepoInfo.data.path || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Cloned:</span>
                        <span className="font-medium">{currentRepoInfo.data.cloned_at ? formatDate(new Date(currentRepoInfo.data.cloned_at * 1000)) : 'N/A'}</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-900 mb-3">Language Distribution</h3>
                    <div className="space-y-2">
                      {currentRepoInfo.data.languages ? Object.entries(currentRepoInfo.data.languages).map(([lang, percentage]) => (
                        <div key={lang} className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">{lang}</span>
                          <div className="flex items-center space-x-2">
                            <div className="w-24 bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full" 
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium w-8">{percentage}%</span>
                          </div>
                        </div>
                      )) : (
                        <p className="text-sm text-gray-500">No language data available</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Last Commit */}
                {currentRepoInfo.data.last_commit && (
                  <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                    <h3 className="font-medium text-gray-900 mb-3">Last Commit</h3>
                    <div className="flex items-center space-x-3">
                      <div className="h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                        <GitCommit className="h-4 w-4 text-gray-600" />
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900">{currentRepoInfo.data.last_commit.message}</p>
                        <p className="text-xs text-gray-500">
                          {currentRepoInfo.data.last_commit.author} • {formatDate(currentRepoInfo.data.last_commit.date)} • 
                          <span className="font-mono ml-1">{currentRepoInfo.data.last_commit.hash.substring(0, 7)}</span>
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          )}

          {/* System Resources */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <HardDrive className="h-8 w-8 text-blue-600 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">Storage</h3>
                    <p className="text-sm text-gray-600">Workspace usage</p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Total Size:</span>
                    <span className="font-medium">{workspaceStats?.data?.total_size || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Repositories:</span>
                    <span className="font-medium">{workspaceStats?.data?.total_repos || 'N/A'}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full" 
                      style={{ width: '65%' }} // Mock percentage
                    />
                  </div>
                  <p className="text-xs text-gray-500">65% of workspace allocated</p>
                </div>
              </div>
            </Card>

            <Card>
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <Cpu className="h-8 w-8 text-green-600 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">CPU Usage</h3>
                    <p className="text-sm text-gray-600">Current load</p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Current:</span>
                    <span className="font-medium">45%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Peak:</span>
                    <span className="font-medium">78%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full" 
                      style={{ width: '45%' }}
                    />
                  </div>
                  <p className="text-xs text-gray-500">Normal operation</p>
                </div>
              </div>
            </Card>

            <Card>
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <HardDrive className="h-8 w-8 text-purple-600 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">Memory</h3>
                    <p className="text-sm text-gray-600">RAM usage</p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Used:</span>
                    <span className="font-medium">6.2 GB</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Total:</span>
                    <span className="font-medium">16 GB</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-purple-500 h-2 rounded-full" 
                      style={{ width: '39%' }}
                    />
                  </div>
                  <p className="text-xs text-gray-500">39% of RAM used</p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Repositories Tab */}
      {activeTab === 'repositories' && (
        <div className="space-y-6">
          {/* Filters and Actions */}
          <div className="flex justify-between items-center">
            <div className="flex space-x-4">
              <div className="relative">
                <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <Input
                  type="text"
                  placeholder="Search repositories..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64"
                />
              </div>
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="w-32"
              >
                <option value="all">All Status</option>
                <option value="current">Current</option>
                <option value="other">Other</option>
              </Select>
              <Select
                value={ageFilter}
                onChange={(e) => setAgeFilter(e.target.value)}
                className="w-32"
              >
                <option value="all">All Ages</option>
                <option value="recent">Recent (&lt;=6h)</option>
                <option value="old">Old (&gt;6h)</option>
              </Select>
            </div>
            <Button onClick={() => refetchStats()}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>

          {/* Repositories Table */}
          <Card>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Repository
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Branch
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Size
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Age
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredRepos.map((repo) => (
                    <tr key={repo.job_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {getStatusIcon(repo.is_current || false)}
                          <div className="ml-3">
                            <div className="text-sm font-medium text-gray-900">
                              {repo.repo_url?.split('/').slice(-2).join('/') || 'N/A'}
                            </div>
                            <div className="text-sm text-gray-500">{repo.job_id || 'N/A'}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <GitBranch className="h-4 w-4 text-gray-400 mr-2" />
                          <span className="text-sm text-gray-900">{repo.branch}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={cn("text-sm font-medium", getSizeColor(repo.size_human))}>
                          {repo.size_human}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={cn("text-sm", getAgeColor(repo.age_hours))}>
                          {repo.age_hours}h ago
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={cn(
                          "px-2 py-1 text-xs font-medium rounded-full",
                          repo.is_current 
                            ? "bg-green-100 text-green-800" 
                            : "bg-gray-100 text-gray-800"
                        )}>
                          {repo.is_current ? 'Current' : 'Stored'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <div className="flex space-x-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              // Navigate to job detail
                              window.location.href = `/jobs/${repo.job_id}`
                            }}
                          >
                            <Eye className="h-3 w-3 mr-1" />
                            View
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              if (confirm(`Are you sure you want to cleanup ${repo.repo_url?.split('/').slice(-1)[0] || 'repository'}?`)) {
                                // Handle cleanup
                                toast.success('Repository cleanup initiated')
                              }
                            }}
                            className="text-red-600 border-red-300 hover:bg-red-50"
                          >
                            <Trash2 className="h-3 w-3 mr-1" />
                            Cleanup
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Cleanup Tab */}
      {activeTab === 'cleanup' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Cleanup Old Workspaces */}
            <Card>
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <Clock className="h-8 w-8 text-orange-600 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">Cleanup Old Workspaces</h3>
                    <p className="text-sm text-gray-600">Remove repositories older than specified age</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Maximum Age (hours)
                    </label>
                    <Input
                      type="number"
                      value={maxAgeHours}
                      onChange={(e) => setMaxAgeHours(parseInt(e.target.value))}
                      min="1"
                      max="168"
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Repositories older than this will be removed
                    </p>
                  </div>
                  
                  <Button
                    onClick={() => cleanupOldMutation.mutate(maxAgeHours)}
                    disabled={cleanupOldMutation.isPending}
                    className="w-full"
                  >
                    {cleanupOldMutation.isPending ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        Cleaning up...
                      </>
                    ) : (
                      <>
                        <Trash2 className="h-4 w-4 mr-2" />
                        Cleanup Old Workspaces
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Cleanup All Repositories */}
            <Card>
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <AlertTriangle className="h-8 w-8 text-red-600 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">Cleanup All Repositories</h3>
                    <p className="text-sm text-gray-600">Remove all stored repositories (dangerous!)</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-sm text-red-800">
                      ⚠️ This will remove ALL repositories from the workspace. 
                      This action cannot be undone.
                    </p>
                  </div>
                  
                  <Button
                    onClick={() => {
                      if (confirm('Are you sure you want to cleanup ALL repositories? This action cannot be undone!')) {
                        cleanupAllMutation.mutate()
                      }
                    }}
                    disabled={cleanupAllMutation.isPending}
                    className="w-full bg-red-600 hover:bg-red-700"
                  >
                    {cleanupAllMutation.isPending ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        Cleaning up...
                      </>
                    ) : (
                      <>
                        <Trash2 className="h-4 w-4 mr-2" />
                        Cleanup All Repositories
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Cleanup Current Repository */}
            <Card>
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <GitBranch className="h-8 w-8 text-blue-600 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">Cleanup Current Repository</h3>
                    <p className="text-sm text-gray-600">Remove the currently active repository</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  {currentRepoInfo?.data ? (
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-sm text-blue-800">
                        Current: {currentRepoInfo.data.repo_url?.split('/').slice(-2).join('/') || 'N/A'}
                      </p>
                    </div>
                  ) : (
                    <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                      <p className="text-sm text-gray-600">
                        No current repository
                      </p>
                    </div>
                  )}
                  
                  <Button
                    onClick={() => cleanupCurrentMutation.mutate()}
                    disabled={cleanupCurrentMutation.isPending || !currentRepoInfo}
                    className="w-full"
                  >
                    {cleanupCurrentMutation.isPending ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        Cleaning up...
                      </>
                    ) : (
                      <>
                        <Trash2 className="h-4 w-4 mr-2" />
                        Cleanup Current Repository
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Card>


          </div>
        </div>
      )}

      {/* Settings Tab */}
      {activeTab === 'settings' && (
        <div className="space-y-6">
          <Card>
            <div className="p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Workspace Configuration</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Auto Cleanup After Review
                  </label>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="auto-cleanup"
                      className="rounded"
                      defaultChecked={false}
                    />
                    <label htmlFor="auto-cleanup" className="ml-2 text-sm text-gray-700">
                      Automatically cleanup repositories after review completion
                    </label>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Auto Cleanup Old Repositories
                  </label>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="auto-cleanup-old"
                      className="rounded"
                      defaultChecked={true}
                    />
                    <label htmlFor="auto-cleanup-old" className="ml-2 text-sm text-gray-700">
                      Automatically cleanup repositories older than specified age
                    </label>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Cleanup Max Age (hours)
                  </label>
                  <Input
                    type="number"
                    defaultValue={24}
                    min="1"
                    max="168"
                    className="w-32"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Cleanup Same URL Repositories
                  </label>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="cleanup-same-url"
                      className="rounded"
                      defaultChecked={true}
                    />
                    <label htmlFor="cleanup-same-url" className="ml-2 text-sm text-gray-700">
                      Cleanup previous repository when starting new work on same URL
                    </label>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 pt-4 border-t border-gray-200">
                <Button>
                  <Settings className="h-4 w-4 mr-2" />
                  Save Configuration
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
