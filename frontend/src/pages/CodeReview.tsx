import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Search, 
  Upload, 
  FileText, 
  Code, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Download,
  Eye,
  GitBranch,
  Shield,
  Zap,
  RefreshCw,
  Settings,
  BarChart3,
  TrendingUp,
  FileCode,
  GitPullRequest,
  Workflow,
  Database,
  Cpu,
  HardDrive
} from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card } from '@/components/ui/Card'
import { Select } from '@/components/ui/Select'
import { cn, formatDate } from '@/lib/utils'

import { codeReviewApi } from '@/lib/api'

export function CodeReview() {
  const [analysisMode, setAnalysisMode] = useState<'standalone' | 'pipeline'>('standalone')
  const [repoPath, setRepoPath] = useState('')
  const [jobId, setJobId] = useState('')
  const [analysisOptions, setAnalysisOptions] = useState({
    complexity_threshold: 10,
    function_length_threshold: 20,
    class_length_threshold: 50,
    enable_duplication_detection: true,
    enable_efficiency_analysis: true,
    enable_hardcoded_detection: true
  })
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<any>(null)
  
  const queryClient = useQueryClient()

  const handleStandaloneAnalysis = async () => {
    if (!repoPath.trim()) {
      toast.error('Please enter a repository path')
      return
    }

    setIsAnalyzing(true)
    try {
      const result = await codeReviewApi.standalone({
        repo_path: repoPath,
        analysis_options: analysisOptions
      })
      setResults(result)
      toast.success('Standalone analysis completed successfully!')
    } catch (error) {
      toast.error('Analysis failed. Please try again.')
      console.error('Analysis error:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handlePipelineAnalysis = async () => {
    if (!jobId.trim()) {
      toast.error('Please enter a job ID')
      return
    }

    setIsAnalyzing(true)
    try {
      const result = await codeReviewApi.pipeline(jobId)
      setResults(result)
      toast.success('Pipeline analysis completed successfully!')
    } catch (error) {
      toast.error('Pipeline analysis failed. Please try again.')
      console.error('Analysis error:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      toast.success(`File uploaded: ${file.name}`)
    }
  }

  const exportResults = (format: 'json' | 'markdown') => {
    if (!results) return
    
    const dataStr = format === 'json' 
      ? JSON.stringify(results, null, 2)
      : generateMarkdownReport(results)
    
    const dataBlob = new Blob([dataStr], { type: 'text/plain' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `code-review-report.${format}`
    link.click()
    URL.revokeObjectURL(url)
    
    toast.success(`Report exported as ${format.toUpperCase()}`)
  }

  const generateMarkdownReport = (data: any) => {
    let markdown = '# Code Review Report\n\n'
    markdown += `**Total Findings:** ${data.total_findings}\n\n`
    
    if (data.findings_by_category) {
      markdown += '## Findings by Category\n\n'
      Object.entries(data.findings_by_category).forEach(([category, count]) => {
        markdown += `- **${category}**: ${count}\n`
      })
      markdown += '\n'
    }
    
    if (data.findings) {
      markdown += '## Detailed Findings\n\n'
      data.findings.forEach((finding: any, index: number) => {
        markdown += `### ${index + 1}. ${finding.file}:${finding.line}\n`
        markdown += `**Category:** ${finding.category}\n`
        markdown += `**Severity:** ${finding.severity}\n`
        markdown += `**Message:** ${finding.message}\n`
        if (finding.suggestion) {
          markdown += `**Suggestion:** ${finding.suggestion}\n`
        }
        markdown += '\n'
      })
    }
    
    return markdown
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200'
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200'
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'low': return 'bg-green-100 text-green-800 border-green-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'security': return <Shield className="h-4 w-4" />
      case 'quality': return <Code className="h-4 w-4" />
      case 'refactoring': return <RefreshCw className="h-4 w-4" />
      case 'reusability': return <GitBranch className="h-4 w-4" />
      case 'efficiency': return <Zap className="h-4 w-4" />
      case 'configuration': return <Settings className="h-4 w-4" />
      case 'dependency': return <Database className="h-4 w-4" />
      default: return <FileText className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Code Review Agent</h1>
        <p className="mt-2 text-gray-600">
          Advanced code analysis for quality improvements, refactoring opportunities, and reusable method suggestions
        </p>
      </div>

      {/* Analysis Mode Selection */}
      <Card>
        <div className="p-6">
          <div className="flex space-x-4 mb-6">
            <button
              onClick={() => setAnalysisMode('standalone')}
              className={cn(
                "px-4 py-2 rounded-lg font-medium transition-colors",
                analysisMode === 'standalone'
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              )}
            >
              <Search className="h-4 w-4 inline mr-2" />
              Standalone Analysis
            </button>
            <button
              onClick={() => setAnalysisMode('pipeline')}
              className={cn(
                "px-4 py-2 rounded-lg font-medium transition-colors",
                analysisMode === 'pipeline'
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              )}
            >
              <Workflow className="h-4 w-4 inline mr-2" />
              Pipeline Integration
            </button>
          </div>

          {analysisMode === 'standalone' ? (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Repository Path
                </label>
                <Input
                  type="text"
                  placeholder="/path/to/repository"
                  value={repoPath}
                  onChange={(e) => setRepoPath(e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Code Files (Optional)
                </label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                  <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                  <input
                    type="file"
                    accept=".zip,.tar.gz,.py,.js,.ts,.java"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <span className="text-blue-600 hover:text-blue-500">Click to upload</span>
                    <span className="text-gray-500"> or drag and drop</span>
                  </label>
                  <p className="text-sm text-gray-500 mt-1">
                    ZIP, TAR.GZ, or individual code files
                  </p>
                </div>
                {uploadedFile && (
                  <div className="mt-2 text-sm text-gray-600">
                    📁 {uploadedFile.name} ({Math.round(uploadedFile.size / 1024)} KB)
                  </div>
                )}
              </div>

              <Button
                onClick={handleStandaloneAnalysis}
                disabled={isAnalyzing || !repoPath.trim()}
                className="w-full"
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    Start Standalone Analysis
                  </>
                )}
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Job ID
                </label>
                <Input
                  type="text"
                  placeholder="Enter job ID from pipeline"
                  value={jobId}
                  onChange={(e) => setJobId(e.target.value)}
                  className="w-full"
                />
              </div>

              <Button
                onClick={handlePipelineAnalysis}
                disabled={isAnalyzing || !jobId.trim()}
                className="w-full"
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Workflow className="h-4 w-4 mr-2" />
                    Run Pipeline Analysis
                  </>
                )}
              </Button>
            </div>
          )}
        </div>
      </Card>

      {/* Analysis Options */}
      <Card>
        <div className="p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Analysis Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Complexity Threshold
              </label>
              <Input
                type="number"
                value={analysisOptions.complexity_threshold}
                onChange={(e) => setAnalysisOptions(prev => ({
                  ...prev,
                  complexity_threshold: parseInt(e.target.value)
                }))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Function Length Threshold
              </label>
              <Input
                type="number"
                value={analysisOptions.function_length_threshold}
                onChange={(e) => setAnalysisOptions(prev => ({
                  ...prev,
                  function_length_threshold: parseInt(e.target.value)
                }))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Class Length Threshold
              </label>
              <Input
                type="number"
                value={analysisOptions.class_length_threshold}
                onChange={(e) => setAnalysisOptions(prev => ({
                  ...prev,
                  class_length_threshold: parseInt(e.target.value)
                }))}
                className="w-full"
              />
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="duplication"
                checked={analysisOptions.enable_duplication_detection}
                onChange={(e) => setAnalysisOptions(prev => ({
                  ...prev,
                  enable_duplication_detection: e.target.checked
                }))}
                className="rounded"
              />
              <label htmlFor="duplication" className="text-sm text-gray-700">
                Enable Duplication Detection
              </label>
            </div>
          </div>
        </div>
      </Card>

      {/* Results Display */}
      {results && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <div className="p-4 text-center">
                <BarChart3 className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                <p className="text-2xl font-bold text-gray-900">{results.total_findings}</p>
                <p className="text-sm text-gray-600">Total Findings</p>
              </div>
            </Card>
            
            {results.findings_by_stage && (
              <Card>
                <div className="p-4 text-center">
                  <Workflow className="h-8 w-8 text-green-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-gray-900">
                    {Object.keys(results.findings_by_stage).length}
                  </p>
                  <p className="text-sm text-gray-600">Pipeline Stages</p>
                </div>
              </Card>
            )}
            
            <Card>
              <div className="p-4 text-center">
                <AlertTriangle className="h-8 w-8 text-red-600 mx-auto mb-2" />
                <p className="text-2xl font-bold text-gray-900">
                  {(results.findings_by_severity?.critical || 0) + (results.findings_by_severity?.high || 0)}
                </p>
                <p className="text-sm text-gray-600">Critical/High Issues</p>
              </div>
            </Card>
            
            <Card>
              <div className="p-4 text-center">
                <CheckCircle className="h-8 w-8 text-green-600 mx-auto mb-2" />
                <p className="text-2xl font-bold text-gray-900">
                  {results.findings?.filter((f: any) => f.autofixable).length || 0}
                </p>
                <p className="text-sm text-gray-600">Auto-fixable</p>
              </div>
            </Card>
          </div>

          {/* Export Options */}
          <Card>
            <div className="p-4 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900">Export Results</h3>
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  onClick={() => exportResults('json')}
                  size="sm"
                >
                  <Download className="h-4 w-4 mr-2" />
                  JSON
                </Button>
                <Button
                  variant="outline"
                  onClick={() => exportResults('markdown')}
                  size="sm"
                >
                  <FileText className="h-4 w-4 mr-2" />
                  Markdown
                </Button>
              </div>
            </div>
          </Card>

          {/* Findings by Category */}
          {results.findings_by_category && (
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Findings by Category</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(results.findings_by_category).map(([category, count]) => (
                    <div key={category} className="flex items-center p-3 bg-gray-50 rounded-lg">
                      {getCategoryIcon(category)}
                      <div className="ml-3">
                        <p className="font-medium text-gray-900 capitalize">{category}</p>
                        <p className="text-sm text-gray-600">{count} findings</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}

          {/* Findings by Severity */}
          {results.findings_by_severity && (
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Findings by Severity</h3>
                <div className="space-y-3">
                  {Object.entries(results.findings_by_severity).map(([severity, count]) => (
                    <div key={severity} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center">
                        <span className={cn(
                          "px-2 py-1 text-xs font-medium rounded-full border",
                          getSeverityColor(severity)
                        )}>
                          {severity.toUpperCase()}
                        </span>
                        <span className="ml-3 text-gray-700">{count} findings</span>
                      </div>
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div 
                          className={cn(
                            "h-2 rounded-full",
                            severity === 'critical' ? 'bg-red-500' :
                            severity === 'high' ? 'bg-orange-500' :
                            severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                          )}
                          style={{ width: `${(count as number / results.total_findings) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}

          {/* Detailed Findings */}
          {results.findings && (
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Findings</h3>
                <div className="space-y-4">
                  {results.findings.map((finding: any, index: number) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <span className={cn(
                            "px-2 py-1 text-xs font-medium rounded-full border",
                            getSeverityColor(finding.severity)
                          )}>
                            {finding.severity.toUpperCase()}
                          </span>
                          <span className="text-sm text-gray-600 capitalize">
                            {finding.category}
                          </span>
                        </div>
                        <div className="text-sm text-gray-500">
                          {finding.file}:{finding.line}
                        </div>
                      </div>
                      
                      <h4 className="font-medium text-gray-900 mb-2">{finding.message}</h4>
                      
                      {finding.suggestion && (
                        <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                          <p className="text-sm text-blue-800">
                            <strong>💡 Suggestion:</strong> {finding.suggestion}
                          </p>
                        </div>
                      )}
                      
                      {finding.code_snippet && (
                        <div className="mb-3">
                          <p className="text-sm font-medium text-gray-700 mb-2">Code Snippet:</p>
                          <pre className="bg-gray-100 p-3 rounded-lg text-sm overflow-x-auto">
                            <code>{finding.code_snippet}</code>
                          </pre>
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between text-sm text-gray-600">
                        <div className="flex items-center space-x-4">
                          <span>Confidence: {Math.round((finding.confidence || 0) * 100)}%</span>
                          <span>Impact: {finding.impact}</span>
                          <span>Effort: {finding.effort}</span>
                        </div>
                        {finding.autofixable && (
                          <span className="text-green-600 font-medium">🔄 Auto-fixable</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}

          {/* Pipeline Information */}
          {results.metadata?.pipeline_stages && (
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Information</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-700">Analysis Type:</span>
                    <span className="font-medium capitalize">{results.metadata.analysis_type}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-700">Pipeline Stages:</span>
                    <span className="font-medium">{results.metadata.pipeline_stages.join(' → ')}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-700">Tools Used:</span>
                    <span className="font-medium">{results.metadata.tools_used.join(', ')}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-700">Stages Completed:</span>
                    <span className="font-medium">{results.metadata.stages_completed.join(', ')}</span>
                  </div>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Feature Overview */}
      <Card>
        <div className="p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Code Review Agent Capabilities</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
              <Code className="h-5 w-5 text-blue-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-blue-900">Code Quality</h3>
                <p className="text-sm text-blue-700">Style, imports, best practices, consistency</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
              <RefreshCw className="h-5 w-5 text-green-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-green-900">Refactoring</h3>
                <p className="text-sm text-green-700">Long functions, large classes, nested logic</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg">
              <GitBranch className="h-5 w-5 text-purple-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-purple-900">Reusability</h3>
                <p className="text-sm text-purple-700">Code duplication, similar patterns, inheritance</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-orange-50 rounded-lg">
              <Zap className="h-5 w-5 text-orange-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-orange-900">Efficiency</h3>
                <p className="text-sm text-orange-700">Performance patterns, algorithm optimization</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
              <Settings className="h-5 w-5 text-yellow-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-yellow-900">Configuration</h3>
                <p className="text-sm text-yellow-700">Hardcoded values, environment variables</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-red-50 rounded-lg">
              <Cpu className="h-5 w-5 text-red-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-red-900">Complexity</h3>
                <p className="text-sm text-red-700">Cyclomatic complexity, maintainability</p>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}
