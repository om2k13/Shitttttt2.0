import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import JobDetail from './pages/JobDetail'
import Reports from './pages/Reports'
import Analytics from './pages/Analytics'
import Settings from './pages/Settings'

function App() {
  return (
    <Router>
      <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
        {/* Navigation Header */}
        <nav style={{
          background: 'white',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          padding: '15px 0',
          marginBottom: '20px'
        }}>
          <div style={{ 
            maxWidth: '1200px', 
            margin: '0 auto', 
            padding: '0 20px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <div style={{ 
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                padding: '8px 15px',
                borderRadius: '20px',
                fontWeight: 'bold',
                fontSize: '1.2em'
              }}>
                🔍 Code Review Agent
              </div>
            </div>
            
            <div style={{ display: 'flex', gap: '20px' }}>
              <a href="/" style={{ 
                textDecoration: 'none', 
                color: '#333', 
                padding: '8px 15px',
                borderRadius: '20px',
                transition: 'all 0.3s ease',
                fontWeight: '500'
              }}
              onMouseOver={(e) => e.currentTarget.style.background = '#f8f9fa'}
              onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                📊 Dashboard
              </a>
              <a href="/reports" style={{ 
                textDecoration: 'none', 
                color: '#333', 
                padding: '8px 15px',
                borderRadius: '20px',
                transition: 'all 0.3s ease',
                fontWeight: '500'
              }}
              onMouseOver={(e) => e.currentTarget.style.background = '#f8f9fa'}
              onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                📋 Reports
              </a>
              <a href="/analytics" style={{ 
                textDecoration: 'none', 
                color: '#333', 
                padding: '8px 15px',
                borderRadius: '20px',
                transition: 'all 0.3s ease',
                fontWeight: '500'
              }}
              onMouseOver={(e) => e.currentTarget.style.background = '#f8f9fa'}
              onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                📈 Analytics
              </a>
              <a href="/settings" style={{ 
                textDecoration: 'none', 
                color: '#333', 
                padding: '8px 15px',
                borderRadius: '20px',
                transition: 'all 0.3s ease',
                fontWeight: '500'
              }}
              onMouseOver={(e) => e.currentTarget.style.background = '#f8f9fa'}
              onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                ⚙️ Settings
              </a>
            </div>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/jobs/:id" element={<JobDetail />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
