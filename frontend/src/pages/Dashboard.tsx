import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Link } from 'react-router-dom'

export default function Dashboard(){
  const [jobs,setJobs] = useState<any>({items:[]})
  const [repoUrl,setRepoUrl] = useState('')
  const [branch,setBranch] = useState('')

  const load = async()=>{
    const r = await api.get('/api/jobs')
    setJobs(r.data)
  }
  useEffect(()=>{ load() }, [])

  const create = async()=>{
    if (!repoUrl.trim()) {
      alert('Please enter a repository URL')
      return
    }
    await api.post('/api/jobs',{repo_url:repoUrl, branch})
    setRepoUrl(''); setBranch('')
    setTimeout(load, 800)
  }

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'completed': return '✅'
      case 'running': return '🔄'
      case 'queued': return '⏳'
      case 'failed': return '❌'
      default: return '❓'
    }
  }

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'completed': return '#28a745'
      case 'running': return '#007bff'
      case 'queued': return '#ffc107'
      case 'failed': return '#dc3545'
      default: return '#6c757d'
    }
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      {/* Header */}
      <div style={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
        color: 'white', 
        padding: '40px', 
        borderRadius: '20px',
        marginBottom: '40px',
        textAlign: 'center'
      }}>
        <h1 style={{ margin: '0 0 15px 0', fontSize: '3em' }}>🔍 Code Review Agent</h1>
        <p style={{ fontSize: '1.3em', margin: '0', opacity: 0.9 }}>
          Automatically analyze your GitHub repositories for code quality, security, and best practices
        </p>
      </div>

      {/* New Job Form */}
      <div style={{ 
        background: 'white', 
        padding: '30px', 
        borderRadius: '15px', 
        boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        marginBottom: '40px'
      }}>
        <h2 style={{ color: '#333', margin: '0 0 25px 0', textAlign: 'center' }}>🚀 Start New Code Review</h2>
        
        <div style={{ display: 'flex', gap: '15px', marginBottom: '20px', flexWrap: 'wrap' }}>
          <div style={{ flex: '1', minWidth: '300px' }}>
            <label style={{ display: 'block', marginBottom: '8px', color: '#555', fontWeight: 'bold' }}>
              Repository URL *
            </label>
            <input 
              placeholder="https://github.com/username/repository.git" 
              value={repoUrl} 
              onChange={e=>setRepoUrl(e.target.value)} 
              style={{
                width: '100%',
                padding: '15px',
                border: '2px solid #e9ecef',
                borderRadius: '10px',
                fontSize: '1em',
                transition: 'border-color 0.3s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#667eea'}
              onBlur={(e) => e.target.style.borderColor = '#e9ecef'}
            />
          </div>
          
          <div style={{ minWidth: '200px' }}>
            <label style={{ display: 'block', marginBottom: '8px', color: '#555', fontWeight: 'bold' }}>
              Branch (optional)
            </label>
            <input 
              placeholder="main" 
              value={branch} 
              onChange={e=>setBranch(e.target.value)} 
              style={{
                width: '100%',
                padding: '15px',
                border: '2px solid #e9ecef',
                borderRadius: '10px',
                fontSize: '1em',
                transition: 'border-color 0.3s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#667eea'}
              onBlur={(e) => e.target.style.borderColor = '#e9ecef'}
            />
          </div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <button 
            onClick={create}
            style={{
              background: '#28a745',
              color: 'white',
              border: 'none',
              padding: '18px 40px',
              borderRadius: '25px',
              fontSize: '1.2em',
              cursor: 'pointer',
              boxShadow: '0 4px 15px rgba(40, 167, 69, 0.3)',
              transition: 'all 0.3s ease',
              fontWeight: 'bold'
            }}
            onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
          >
            🔍 Start Code Review
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <div style={{ fontSize: '2.5em', marginBottom: '10px' }}>📊</div>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#667eea' }}>{jobs.total || 0}</div>
          <div style={{ color: '#666' }}>Total Reviews</div>
        </div>
        
        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <div style={{ fontSize: '2.5em', marginBottom: '10px' }}>✅</div>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#28a745' }}>
            {jobs.items?.filter((j: any) => j.status === 'completed').length || 0}
          </div>
          <div style={{ color: '#666' }}>Completed</div>
        </div>
        
        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <div style={{ fontSize: '2.5em', marginBottom: '10px' }}>🔄</div>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#007bff' }}>
            {jobs.items?.filter((j: any) => j.status === 'running').length || 0}
          </div>
          <div style={{ color: '#666' }}>Running</div>
        </div>
        
        <div style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <div style={{ fontSize: '2.5em', marginBottom: '10px' }}>⏳</div>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#ffc107' }}>
            {jobs.items?.filter((j: any) => j.status === 'queued').length || 0}
          </div>
          <div style={{ color: '#666' }}>Queued</div>
        </div>
      </div>

      {/* Recent Jobs */}
      <div style={{ background: 'white', padding: '30px', borderRadius: '15px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
        <h2 style={{ color: '#333', margin: '0 0 25px 0', textAlign: 'center' }}>📋 Recent Code Reviews</h2>
        
        {jobs.items?.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
            <div style={{ fontSize: '3em', marginBottom: '20px' }}>🚀</div>
            <h3>No reviews yet!</h3>
            <p>Start your first code review by entering a GitHub repository URL above.</p>
          </div>
        ) : (
          <div style={{ display: 'grid', gap: '15px' }}>
            {jobs.items.map((j: any) => (
              <div key={j.id} style={{ 
                background: '#f8f9fa', 
                padding: '20px', 
                borderRadius: '10px',
                border: '2px solid #e9ecef',
                transition: 'all 0.3s ease'
              }}
              onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                    <Link 
                      to={`/jobs/${j.id}`}
                      style={{
                        background: '#667eea',
                        color: 'white',
                        padding: '8px 15px',
                        borderRadius: '20px',
                        textDecoration: 'none',
                        fontWeight: 'bold',
                        fontSize: '0.9em'
                      }}
                    >
                      {j.id.slice(0,8)}...{j.id.slice(-8)}
                    </Link>
                    
                    <div style={{ 
                      background: getStatusColor(j.status), 
                      color: 'white', 
                      padding: '5px 12px', 
                      borderRadius: '15px',
                      fontSize: '0.8em',
                      fontWeight: 'bold'
                    }}>
                      {getStatusIcon(j.status)} {j.status}
                    </div>
                  </div>
                  
                  <div style={{ color: '#666', fontSize: '0.9em' }}>
                    {new Date(j.created_at).toLocaleDateString()}
                  </div>
                </div>
                
                <div style={{ marginBottom: '10px' }}>
                  <strong style={{ color: '#333' }}>Repository:</strong>
                  <span style={{ 
                    background: '#e9ecef', 
                    padding: '5px 10px', 
                    borderRadius: '5px', 
                    marginLeft: '10px',
                    fontFamily: 'monospace',
                    fontSize: '0.9em',
                    wordBreak: 'break-all'
                  }}>
                    {j.repo_url}
                  </span>
                </div>
                
                {j.branch && (
                  <div style={{ marginBottom: '10px' }}>
                    <strong style={{ color: '#333' }}>Branch:</strong>
                    <span style={{ 
                      background: '#007bff', 
                      color: 'white', 
                      padding: '3px 8px', 
                      borderRadius: '5px', 
                      marginLeft: '10px',
                      fontSize: '0.8em'
                    }}>
                      {j.branch}
                    </span>
                  </div>
                )}
                
                {j.status === 'completed' && (
                  <div style={{ 
                    background: '#d4edda', 
                    color: '#155724', 
                    padding: '10px', 
                    borderRadius: '8px',
                    fontSize: '0.9em'
                  }}>
                    ✅ Review completed successfully! Click the job ID to view detailed results.
                  </div>
                )}
                
                {j.status === 'running' && (
                  <div style={{ 
                    background: '#d1ecf1', 
                    color: '#0c5460', 
                    padding: '10px', 
                    borderRadius: '8px',
                    fontSize: '0.9em'
                  }}>
                    🔄 Code review in progress... This may take a few minutes.
                  </div>
                )}
                
                {j.status === 'queued' && (
                  <div style={{ 
                    background: '#fff3cd', 
                    color: '#856404', 
                    padding: '10px', 
                    borderRadius: '8px',
                    fontSize: '0.9em'
                  }}>
                    ⏳ Waiting in queue... Your review will start soon.
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
