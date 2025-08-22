import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  BarChart3, 
  FileText, 
  Settings, 
  GitBranch, 
  Home, 
  Menu, 
  X,
  Shield,
  Zap,
  TestTube,
  Github,
  Workflow,
  Code,
  Users
} from 'lucide-react'
import { cn } from '@/lib/utils'


const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Jobs', href: '/jobs', icon: Workflow },
  { name: 'Reports', href: '/reports', icon: FileText },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Code Review', href: '/code-review', icon: Code },
  { name: 'Workspace', href: '/workspace', icon: GitBranch },
  { name: 'Users', href: '/users', icon: Users },
  { name: 'Settings', href: '/settings', icon: Settings },
]

const featureHighlights = [
  {
    name: 'Security Analysis',
    description: 'OWASP Top 10 vulnerability detection',
    icon: Shield,
    color: 'text-red-600 bg-red-50',
  },
  {
    name: 'Code Review',
    description: 'AI-powered code quality analysis',
    icon: Code,
    color: 'text-blue-600 bg-blue-50',
  },
  {
    name: 'PR Analysis',
    description: 'Focused analysis on pull request changes',
    icon: GitBranch,
    color: 'text-green-600 bg-green-50',
  },
  {
    name: 'Test Generation',
    description: 'Automatic test plan and test case generation',
    icon: TestTube,
    color: 'text-purple-600 bg-purple-50',
  },
  {
    name: 'Performance Analysis',
    description: 'Code optimization and performance insights',
    icon: Zap,
    color: 'text-orange-600 bg-orange-50',
  },
  {
    name: 'User Management',
    description: 'Organization and token management',
    icon: Users,
    color: 'text-indigo-600 bg-indigo-50',
  },
]

interface LayoutProps {
  children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile sidebar */}
      <div className={cn(
        "fixed inset-0 z-50 lg:hidden",
        sidebarOpen ? "block" : "hidden"
      )}>
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
        <div className="fixed inset-y-0 left-0 flex w-64 flex-col bg-white">
          <div className="flex h-16 items-center justify-between px-4">
            <h1 className="text-xl font-bold text-gray-900">Code Review Agent</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-6 w-6" />
            </button>
          </div>
          <nav className="flex-1 space-y-1 px-2 py-4">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={cn(
                    "group flex items-center px-2 py-2 text-sm font-medium rounded-md",
                    isActive
                      ? "bg-blue-100 text-blue-900"
                      : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                  )}
                >
                  <item.icon
                    className={cn(
                      "mr-3 h-5 w-5 flex-shrink-0",
                      isActive ? "text-blue-500" : "text-gray-400 group-hover:text-gray-500"
                    )}
                  />
                  {item.name}
                </Link>
              )
            })}
          </nav>
          

        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex flex-col flex-grow bg-white border-r border-gray-200">
          <div className="flex h-16 items-center px-4">
            <h1 className="text-xl font-bold text-gray-900">Code Review Agent</h1>
          </div>
          <nav className="flex-1 space-y-1 px-2 py-4">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={cn(
                    "group flex items-center px-2 py-2 text-sm font-medium rounded-md",
                    isActive
                      ? "bg-blue-100 text-blue-900"
                      : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                  )}
                >
                  <item.icon
                    className={cn(
                      "mr-3 h-5 w-5 flex-shrink-0",
                      isActive ? "text-blue-500" : "text-gray-400 group-hover:text-gray-500"
                    )}
                  />
                  {item.name}
                </Link>
              )
            })}
          </nav>
          
          {/* Feature highlights */}
          <div className="p-4 border-t border-gray-200">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Features
            </h3>
            <div className="space-y-2">
              {featureHighlights.map((feature) => (
                <div key={feature.name} className="flex items-center space-x-2 text-xs">
                  <div className={cn("p-1 rounded", feature.color)}>
                    <feature.icon className="h-3 w-3" />
                  </div>
                  <span className="text-gray-600">{feature.name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Mobile header */}
        <div className="sticky top-0 z-40 flex h-16 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:hidden">
          <button
            type="button"
            className="-m-2.5 p-2.5 text-gray-700 lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-6 w-6" />
          </button>
          <div className="flex-1 text-sm font-semibold leading-6 text-gray-900">
            Code Review Agent
          </div>
        </div>

        {/* Page content */}
        <main className="py-6">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
