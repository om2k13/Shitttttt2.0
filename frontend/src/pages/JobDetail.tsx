import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { useParams } from 'react-router-dom'

export default function JobDetail(){
  const { id } = useParams()
  const [job,setJob] = useState<any>()
  const [report,setReport] = useState<any>({summary:{by_tool:{},by_severity:{}}, findings:[]})

  const load = async()=>{
    const j = await api.get(`/api/jobs/${id}`); setJob(j.data)
    const r = await api.get(`/api/reports/${id}`); setReport(r.data)
  }
  useEffect(()=>{ const t=setInterval(load, 1500); return ()=>clearInterval(t) }, [id])

  const applyFix = async()=>{
    await api.post('/api/actions/apply-fix',{job_id:id})
    alert('Safe autofixes applied! Check the fixed version in .workspaces/[job-id]-fix/')
  }

  // Helper function to get severity color and icon
  const getSeverityInfo = (severity: string) => {
    switch(severity.toLowerCase()) {
      case 'critical': return { color: '#dc3545', icon: '🚨', label: 'Critical' }
      case 'high': return { color: '#fd7e14', icon: '⚠️', label: 'High' }
      case 'medium': return { color: '#ffc107', icon: '⚡', label: 'Medium' }
      case 'low': return { color: '#28a745', icon: '💡', label: 'Low' }
      default: return { color: '#6c757d', icon: 'ℹ️', label: severity }
    }
  }

  // Helper function to get tool description
  const getToolDescription = (tool: string) => {
    switch(tool.toLowerCase()) {
      case 'ruff': return 'Python code linter - finds style and quality issues'
      case 'mypy': return 'Python type checker - finds type-related problems'
      case 'semgrep': return 'Security scanner - finds security vulnerabilities'
      case 'bandit': return 'Security linter - finds common security issues'
      case 'radon': return 'Code complexity analyzer - finds complex functions'
      default: return 'Code analysis tool'
    }
  }

  // Group findings by tool for better organization
  const groupedFindings = report.findings?.reduce((acc: any, finding: any) => {
    if (!acc[finding.tool]) acc[finding.tool] = []
    acc[finding.tool].push(finding)
    return acc
  }, {}) || {}

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      {/* Header Section */}
      <div style={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
        color: 'white', 
        padding: '30px', 
        borderRadius: '15px',
        marginBottom: '30px',
        textAlign: 'center'
      }}>
        <h1 style={{ margin: '0 0 10px 0', fontSize: '2.5em' }}>🔍 Code Review Report</h1>
        <p style={{ fontSize: '1.2em', margin: '0 0 20px 0', opacity: 0.9 }}>
          Job ID: {id?.slice(0,8)}...{id?.slice(-8)}
        </p>
        
        {job && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', flexWrap: 'wrap' }}>
            <div style={{ background: 'rgba(255,255,255,0.2)', padding: '15px', borderRadius: '10px' }}>
              <strong>Status:</strong> {job.status === 'completed' ? '✅ Completed' : job.status}
            </div>
            <div style={{ background: 'rgba(255,255,255,0.2)', padding: '15px', borderRadius: '10px' }}>
              <strong>Stage:</strong> {job.current_stage}
            </div>
            <div style={{ background: 'rgba(255,255,255,0.2)', padding: '15px', borderRadius: '10px' }}>
              <strong>Progress:</strong> {job.progress}%
            </div>
          </div>
        )}
      </div>

      {/* Action Button */}
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <button 
          onClick={applyFix}
          style={{
            background: '#28a745',
            color: 'white',
            border: 'none',
            padding: '15px 30px',
            borderRadius: '25px',
            fontSize: '1.1em',
            cursor: 'pointer',
            boxShadow: '0 4px 15px rgba(40, 167, 69, 0.3)',
            transition: 'all 0.3s ease'
          }}
          onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
          onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
        >
          🚀 Apply Safe Auto-Fixes
        </button>
        <p style={{ color: '#666', marginTop: '10px', fontSize: '0.9em' }}>
          Automatically fixes code style and formatting issues
        </p>
      </div>

      {/* Summary Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '30px' }}>
        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <h3 style={{ color: '#333', margin: '0 0 15px 0' }}>📊 Total Issues</h3>
          <div style={{ fontSize: '3em', fontWeight: 'bold', color: '#667eea' }}>
            {report.summary?.total || 0}
          </div>
          <p style={{ color: '#666', margin: '0' }}>Issues found in your code</p>
        </div>

        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <h3 style={{ color: '#333', margin: '0 0 15px 0' }}>🛠️ Tools Used</h3>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#28a745' }}>
            {Object.keys(report.summary?.by_tool || {}).length}
          </div>
          <p style={{ color: '#666', margin: '0' }}>Analysis tools</p>
        </div>

        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <h3 style={{ color: '#333', margin: '0 0 15px 0' }}>📁 Repository</h3>
          <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#fd7e14', wordBreak: 'break-word' }}>
            {job?.repo_url || 'N/A'}
          </div>
          <p style={{ color: '#666', margin: '0' }}>Analyzed repository</p>
        </div>
      </div>

      {/* Severity Breakdown */}
      {report.summary?.by_severity && Object.keys(report.summary.by_severity).length > 0 && (
        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', marginBottom: '30px' }}>
          <h3 style={{ color: '#333', margin: '0 0 20px 0' }}>🚨 Issues by Severity</h3>
          <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
            {Object.entries(report.summary.by_severity).map(([severity, count]: [string, any]) => {
              const severityInfo = getSeverityInfo(severity)
              return (
                <div key={severity} style={{ 
                  background: severityInfo.color, 
                  color: 'white', 
                  padding: '15px 20px', 
                  borderRadius: '10px',
                  textAlign: 'center',
                  minWidth: '120px'
                }}>
                  <div style={{ fontSize: '2em', marginBottom: '5px' }}>{severityInfo.icon}</div>
                  <div style={{ fontSize: '1.5em', fontWeight: 'bold' }}>{count}</div>
                  <div style={{ fontSize: '0.9em', opacity: 0.9 }}>{severityInfo.label}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Tool Breakdown */}
      {report.summary?.by_tool && Object.keys(report.summary.by_tool).length > 0 && (
        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', marginBottom: '30px' }}>
          <h3 style={{ color: '#333', margin: '0 0 20px 0' }}>🔧 Issues by Tool</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
            {Object.entries(report.summary.by_tool).map(([tool, count]: [string, any]) => (
              <div key={tool} style={{ 
                background: '#f8f9fa', 
                padding: '20px', 
                borderRadius: '10px',
                border: '2px solid #e9ecef',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '2em', marginBottom: '10px' }}>
                  {tool === 'ruff' ? '🐍' : tool === 'mypy' ? '🔍' : tool === 'semgrep' ? '🛡️' : '⚙️'}
                </div>
                <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#333' }}>{count}</div>
                <div style={{ fontSize: '1em', color: '#666', marginBottom: '10px' }}>{tool.toUpperCase()}</div>
                <div style={{ fontSize: '0.8em', color: '#888' }}>{getToolDescription(tool)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Findings */}
      <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
        <h3 style={{ color: '#333', margin: '0 0 20px 0' }}>📋 Detailed Findings</h3>
        
        {Object.keys(groupedFindings).length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
            <div style={{ fontSize: '3em', marginBottom: '20px' }}>🎉</div>
            <h4>No issues found!</h4>
            <p>Your code looks great! No problems detected.</p>
          </div>
        ) : (
          Object.entries(groupedFindings).map(([tool, findings]: [string, any]) => (
            <div key={tool} style={{ marginBottom: '30px' }}>
              <h4 style={{ 
                color: '#333', 
                borderBottom: '2px solid #e9ecef', 
                paddingBottom: '10px',
                marginBottom: '20px'
              }}>
                {tool === 'ruff' ? '🐍 Ruff (Code Style)' : 
                 tool === 'mypy' ? '🔍 MyPy (Type Checking)' : 
                 tool === 'semgrep' ? '🛡️ Semgrep (Security)' : 
                 `${tool} Issues`}
              </h4>
              
              <div style={{ display: 'grid', gap: '15px' }}>
                {findings.map((finding: any, idx: number) => {
                  const severityInfo = getSeverityInfo(finding.severity)
                  const fileName = finding.file?.split('/').pop() || finding.file
                  const filePath = finding.file?.replace(/^.*\/backend\//, 'backend/') || finding.file
                  
                  return (
                    <div key={idx} style={{ 
                      background: '#f8f9fa', 
                      padding: '20px', 
                      borderRadius: '10px',
                      border: `2px solid ${severityInfo.color}`,
                      borderLeft: `5px solid ${severityInfo.color}`
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
                        <div style={{ 
                          background: severityInfo.color, 
                          color: 'white', 
                          padding: '5px 12px', 
                          borderRadius: '15px',
                          fontSize: '0.8em',
                          fontWeight: 'bold'
                        }}>
                          {severityInfo.icon} {severityInfo.label}
                        </div>
                        <div style={{ color: '#666', fontSize: '0.9em' }}>
                          Line {finding.line}
                        </div>
                      </div>
                      
                      <div style={{ marginBottom: '10px' }}>
                        <strong style={{ color: '#333' }}>File:</strong> 
                        <span style={{ 
                          background: '#e9ecef', 
                          padding: '3px 8px', 
                          borderRadius: '5px', 
                          marginLeft: '10px',
                          fontFamily: 'monospace',
                          fontSize: '0.9em'
                        }}>
                          {fileName}
                        </span>
                      </div>
                      
                      <div style={{ marginBottom: '10px' }}>
                        <strong style={{ color: '#333' }}>Path:</strong> 
                        <span style={{ 
                          color: '#666', 
                          marginLeft: '10px',
                          fontFamily: 'monospace',
                          fontSize: '0.9em'
                        }}>
                          {filePath}
                        </span>
                      </div>
                      
                      {finding.rule_id && (
                        <div style={{ marginBottom: '10px' }}>
                          <strong style={{ color: '#333' }}>Rule:</strong> 
                          <span style={{ 
                            background: '#007bff', 
                            color: 'white', 
                            padding: '3px 8px', 
                            borderRadius: '5px', 
                            marginLeft: '10px',
                            fontSize: '0.8em'
                          }}>
                            {finding.rule_id}
                          </span>
                        </div>
                      )}
                      
                      {finding.message && (
                        <div style={{ 
                          background: 'white', 
                          padding: '15px', 
                          borderRadius: '8px',
                          border: '1px solid #e9ecef'
                        }}>
                          <strong style={{ color: '#333' }}>Issue:</strong>
                          <p style={{ margin: '10px 0 0 0', color: '#555', lineHeight: '1.5' }}>
                            {finding.message}
                          </p>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Download Section */}
      <div style={{ 
        background: 'linear-gradient(135deg, #28a745 0%, #20c997 100%)', 
        color: 'white', 
        padding: '25px', 
        borderRadius: '15px',
        marginTop: '30px',
        textAlign: 'center'
      }}>
        <h3 style={{ margin: '0 0 15px 0' }}>💾 Download Report</h3>
        <p style={{ margin: '0 0 20px 0', opacity: 0.9 }}>
          Get a detailed report of all findings in JSON format
        </p>
        <button 
          onClick={() => {
            const dataStr = JSON.stringify(report, null, 2)
            const dataBlob = new Blob([dataStr], {type: 'application/json'})
            const url = URL.createObjectURL(dataBlob)
            const link = document.createElement('a')
            link.href = url
            link.download = `code-review-report-${id?.slice(0,8)}.json`
            link.click()
            URL.revokeObjectURL(url)
          }}
          style={{
            background: 'rgba(255,255,255,0.2)',
            color: 'white',
            border: '2px solid white',
            padding: '12px 25px',
            borderRadius: '25px',
            fontSize: '1em',
            cursor: 'pointer',
            transition: 'all 0.3s ease'
          }}
          onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.3)'}
          onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
        >
          📥 Download JSON Report
        </button>
      </div>
    </div>
  )
}
