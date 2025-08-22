import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import { Dashboard } from '@/pages/Dashboard'
import { Jobs } from '@/pages/Jobs'
import { JobDetail } from '@/pages/JobDetail'
import { Reports } from '@/pages/Reports'
import { Analytics } from '@/pages/Analytics'
import { Settings } from '@/pages/Settings'
import { WorkspaceManagement } from '@/pages/WorkspaceManagement'
import { CodeReview } from '@/pages/CodeReview'
import { UserManagement } from '@/pages/UserManagement'
import { ErrorBoundary } from '@/components/ErrorBoundary'

function App() {
  return (
    <ErrorBoundary>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/jobs/:id" element={<JobDetail />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/reports/:id" element={<Reports />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/workspace" element={<WorkspaceManagement />} />
          <Route path="/code-review" element={<CodeReview />} />
          <Route path="/users" element={<UserManagement />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </ErrorBoundary>
  )
}

export default App
