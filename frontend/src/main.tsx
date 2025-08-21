import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link, Outlet } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Reports from './pages/Reports'
import Settings from './pages/Settings'
import JobDetail from './pages/JobDetail'
import Analytics from './pages/Analytics'

function Layout(){
  return (<div style={{padding:'16px', fontFamily:'system-ui'}}>
    <h1>Code Review Agent</h1>
    <nav style={{display:'flex', gap:'12px', marginBottom:'12px'}}>
      <Link to="/">Dashboard</Link>
      <Link to="/reports">Reports</Link>
      <Link to="/analytics">Analytics</Link>
      <Link to="/settings">Settings</Link>
    </nav>
    <Outlet/>
  </div>)
}

createRoot(document.getElementById('root')!).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Layout/>}>
        <Route index element={<Dashboard/>}/>
        <Route path="jobs/:id" element={<JobDetail/>}/>
        <Route path="reports" element={<Reports/>}/>
        <Route path="analytics" element={<Analytics/>}/>
        <Route path="settings" element={<Settings/>}/>
      </Route>
    </Routes>
  </BrowserRouter>
)
