import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Users, 
  UserPlus, 
  Building, 
  Key, 
  Shield, 
  Settings, 
  Edit,
  Trash2,
  Eye,
  EyeOff,
  CheckCircle,
  XCircle,
  AlertTriangle,
  GitBranch,
  Lock,
  Unlock,
  Activity,
  Calendar,
  Mail,
  Phone,
  Globe,
  Plus,
  Search,
  Filter,
  Download,
  Upload,
  RefreshCw
} from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card } from '@/components/ui/Card'
import { Select } from '@/components/ui/Select'
import { cn, formatDate } from '@/lib/utils'

// Mock API calls - replace with real API integration
const userManagementApi = {
  getUsers: async () => {
    await new Promise(resolve => setTimeout(resolve, 1000))
    return {
      users: [
        {
          id: 1,
          username: 'john_doe',
          email: 'john@example.com',
          full_name: 'John Doe',
          role: 'admin',
          organization: 'Tech Corp',
          github_username: 'johndoe',
          is_active: true,
          created_at: '2024-01-15T10:00:00Z',
          last_login: '2024-01-20T14:30:00Z'
        },
        {
          id: 2,
          username: 'jane_smith',
          email: 'jane@example.com',
          full_name: 'Jane Smith',
          role: 'user',
          organization: 'Tech Corp',
          github_username: 'janesmith',
          is_active: true,
          created_at: '2024-01-16T11:00:00Z',
          last_login: '2024-01-19T16:45:00Z'
        },
        {
          id: 3,
          username: 'bob_wilson',
          email: 'bob@example.com',
          full_name: 'Bob Wilson',
          role: 'org_admin',
          organization: 'Startup Inc',
          github_username: 'bobwilson',
          is_active: false,
          created_at: '2024-01-10T09:00:00Z',
          last_login: '2024-01-15T12:20:00Z'
        }
      ]
    }
  },

  getOrganizations: async () => {
    await new Promise(resolve => setTimeout(resolve, 800))
    return {
      organizations: [
        {
          id: 1,
          name: 'Tech Corp',
          github_org: 'techcorp',
          org_token_scopes: ['repo', 'read:org'],
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-20T10:00:00Z',
          member_count: 15,
          active_projects: 8
        },
        {
          id: 2,
          name: 'Startup Inc',
          github_org: 'startupinc',
          org_token_scopes: ['repo'],
          created_at: '2024-01-05T00:00:00Z',
          updated_at: '2024-01-18T15:30:00Z',
          member_count: 6,
          active_projects: 3
        }
      ]
    }
  },

  getUserTokens: async (userId: number) => {
    await new Promise(resolve => setTimeout(resolve, 600))
    return {
      tokens: [
        {
          id: 1,
          token_name: 'Personal Access Token',
          scopes: ['repo', 'read:user'],
          expires_at: '2025-01-20T00:00:00Z',
          last_used: '2024-01-20T14:30:00Z',
          is_active: true
        },
        {
          id: 2,
          token_name: 'CI/CD Token',
          scopes: ['repo'],
          expires_at: '2024-12-31T00:00:00Z',
          last_used: '2024-01-19T10:15:00Z',
          is_active: true
        }
      ]
    }
  },

  createUser: async (userData: any) => {
    await new Promise(resolve => setTimeout(resolve, 1500))
    return { success: true, user_id: Math.floor(Math.random() * 1000) + 4 }
  },

  updateUser: async (userId: number, userData: any) => {
    await new Promise(resolve => setTimeout(resolve, 1000))
    return { success: true }
  },

  deleteUser: async (userId: number) => {
    await new Promise(resolve => setTimeout(resolve, 800))
    return { success: true }
  },

  addGitHubToken: async (userId: number, tokenData: any) => {
    await new Promise(resolve => setTimeout(resolve, 1200))
    return { success: true, token_id: Math.floor(Math.random() * 100) + 1 }
  },

  testGitHubToken: async (userId: number) => {
    await new Promise(resolve => setTimeout(resolve, 2000))
    return { success: true, scopes: ['repo', 'read:user'], rate_limit: 5000 }
  }
}

export function UserManagement() {
  const [activeTab, setActiveTab] = useState<'users' | 'organizations' | 'tokens'>('users')
  const [selectedUser, setSelectedUser] = useState<any>(null)
  const [showCreateUser, setShowCreateUser] = useState(false)
  const [showEditUser, setShowEditUser] = useState(false)
  const [showAddToken, setShowAddToken] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [roleFilter, setRoleFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')

  const queryClient = useQueryClient()

  // Queries
  const { data: usersData, isLoading: usersLoading } = useQuery({
    queryKey: ['users'],
    queryFn: userManagementApi.getUsers
  })

  const { data: orgsData, isLoading: orgsLoading } = useQuery({
    queryKey: ['organizations'],
    queryFn: userManagementApi.getOrganizations
  })

  const { data: tokensData, isLoading: tokensLoading } = useQuery({
    queryKey: ['user-tokens', selectedUser?.id],
    queryFn: () => selectedUser ? userManagementApi.getUserTokens(selectedUser.id) : null,
    enabled: !!selectedUser
  })

  // Mutations
  const createUserMutation = useMutation({
    mutationFn: userManagementApi.createUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
      toast.success('User created successfully!')
      setShowCreateUser(false)
    },
    onError: () => {
      toast.error('Failed to create user')
    }
  })

  const updateUserMutation = useMutation({
    mutationFn: ({ userId, userData }: { userId: number; userData: any }) => 
      userManagementApi.updateUser(userId, userData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
      toast.success('User updated successfully!')
      setShowEditUser(false)
    },
    onError: () => {
      toast.error('Failed to update user')
    }
  })

  const deleteUserMutation = useMutation({
    mutationFn: userManagementApi.deleteUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
      toast.success('User deleted successfully!')
    },
    onError: () => {
      toast.error('Failed to delete user')
    }
  })

  const addTokenMutation = useMutation({
    mutationFn: ({ userId, tokenData }: { userId: number; tokenData: any }) =>
      userManagementApi.addGitHubToken(userId, tokenData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user-tokens'] })
      toast.success('GitHub token added successfully!')
      setShowAddToken(false)
    },
    onError: () => {
      toast.error('Failed to add GitHub token')
    }
  })

  const testTokenMutation = useMutation({
    mutationFn: userManagementApi.testGitHubToken,
    onSuccess: (data) => {
      toast.success(`Token test successful! Scopes: ${data.scopes.join(', ')}`)
    },
    onError: () => {
      toast.error('Token test failed')
    }
  })

  // Filtered users
  const filteredUsers = usersData?.users?.filter(user => {
    const matchesSearch = user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.full_name?.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesRole = roleFilter === 'all' || user.role === roleFilter
    const matchesStatus = statusFilter === 'all' || 
                         (statusFilter === 'active' && user.is_active) ||
                         (statusFilter === 'inactive' && !user.is_active)
    
    return matchesSearch && matchesRole && matchesStatus
  }) || []

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-red-100 text-red-800 border-red-200'
      case 'org_admin': return 'bg-purple-100 text-purple-800 border-purple-200'
      case 'user': return 'bg-blue-100 text-blue-800 border-blue-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getStatusIcon = (isActive: boolean) => {
    return isActive ? 
      <CheckCircle className="h-4 w-4 text-green-600" /> : 
      <XCircle className="h-4 w-4 text-red-600" />
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">User Management</h1>
        <p className="mt-2 text-gray-600">
          Manage users, organizations, and GitHub tokens for the Code Review Agent
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'users', label: 'Users', icon: Users, count: usersData?.users?.length || 0 },
            { id: 'organizations', label: 'Organizations', icon: Building, count: orgsData?.organizations?.length || 0 },
            { id: 'tokens', label: 'GitHub Tokens', icon: Key, count: tokensData?.tokens?.length || 0 }
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
              <span className="bg-gray-100 text-gray-900 py-0.5 px-2 rounded-full text-xs">
                {tab.count}
              </span>
            </button>
          ))}
        </nav>
      </div>

      {/* Users Tab */}
      {activeTab === 'users' && (
        <div className="space-y-6">
          {/* Actions Bar */}
          <div className="flex justify-between items-center">
            <div className="flex space-x-4">
              <div className="relative">
                <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <Input
                  type="text"
                  placeholder="Search users..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64"
                />
              </div>
              <Select
                value={roleFilter}
                onChange={(e) => setRoleFilter(e.target.value)}
                className="w-32"
              >
                <option value="all">All Roles</option>
                <option value="user">User</option>
                <option value="org_admin">Org Admin</option>
                <option value="admin">Admin</option>
              </Select>
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="w-32"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </Select>
            </div>
            <Button onClick={() => setShowCreateUser(true)}>
              <UserPlus className="h-4 w-4 mr-2" />
              Add User
            </Button>
          </div>

          {/* Users Table */}
          <Card>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      User
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Role
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Organization
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last Login
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredUsers.map((user) => (
                    <tr key={user.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                            <span className="text-sm font-medium text-blue-600">
                              {user.full_name?.split(' ').map(n => n[0]).join('') || user.username[0].toUpperCase()}
                            </span>
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">{user.full_name}</div>
                            <div className="text-sm text-gray-500">{user.email}</div>
                            <div className="text-sm text-gray-400">@{user.username}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={cn(
                          "px-2 py-1 text-xs font-medium rounded-full border",
                          getRoleColor(user.role)
                        )}>
                          {user.role.replace('_', ' ').toUpperCase()}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {user.organization || '—'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {getStatusIcon(user.is_active)}
                          <span className="ml-2 text-sm text-gray-900">
                            {user.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {user.last_login ? formatDate(user.last_login) : 'Never'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <div className="flex space-x-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setSelectedUser(user)
                              setShowEditUser(true)
                            }}
                          >
                            <Edit className="h-3 w-3 mr-1" />
                            Edit
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setSelectedUser(user)
                              setActiveTab('tokens')
                            }}
                          >
                            <Key className="h-3 w-3 mr-1" />
                            Tokens
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              if (confirm(`Are you sure you want to delete ${user.username}?`)) {
                                deleteUserMutation.mutate(user.id)
                              }
                            }}
                            className="text-red-600 border-red-300 hover:bg-red-50"
                          >
                            <Trash2 className="h-3 w-3 mr-1" />
                            Delete
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

      {/* Organizations Tab */}
      {activeTab === 'organizations' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-semibold text-gray-900">Organizations</h2>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Organization
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {orgsData?.organizations?.map((org) => (
              <Card key={org.id}>
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="h-12 w-12 rounded-lg bg-purple-100 flex items-center justify-center">
                      <Building className="h-6 w-6 text-purple-600" />
                    </div>
                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm">
                        <Edit className="h-3 w-3 mr-1" />
                        Edit
                      </Button>
                      <Button variant="outline" size="sm">
                        <Settings className="h-3 w-3 mr-1" />
                        Settings
                      </Button>
                    </div>
                  </div>
                  
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">{org.name}</h3>
                  <p className="text-sm text-gray-600 mb-4">GitHub: @{org.github_org}</p>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Members:</span>
                      <span className="font-medium">{org.member_count}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Active Projects:</span>
                      <span className="font-medium">{org.active_projects}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Token Scopes:</span>
                      <span className="font-medium">{org.org_token_scopes.join(', ')}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Created:</span>
                      <span className="font-medium">{formatDate(org.created_at)}</span>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* GitHub Tokens Tab */}
      {activeTab === 'tokens' && (
        <div className="space-y-6">
          {selectedUser ? (
            <div>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    GitHub Tokens for {selectedUser.full_name}
                  </h2>
                  <p className="text-sm text-gray-600">@{selectedUser.username}</p>
                </div>
                <div className="flex space-x-2">
                  <Button
                    variant="outline"
                    onClick={() => setSelectedUser(null)}
                  >
                    Back to Users
                  </Button>
                  <Button onClick={() => setShowAddToken(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Token
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {tokensData?.tokens?.map((token) => (
                  <Card key={token.id}>
                    <div className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-medium text-gray-900">{token.token_name}</h3>
                        <div className="flex space-x-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => testTokenMutation.mutate(selectedUser.id)}
                            disabled={testTokenMutation.isPending}
                          >
                            {testTokenMutation.isPending ? (
                              <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                            ) : (
                              <Eye className="h-3 w-3 mr-1" />
                            )}
                            Test
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-red-600 border-red-300 hover:bg-red-50"
                          >
                            <Trash2 className="h-3 w-3 mr-1" />
                            Revoke
                          </Button>
                        </div>
                      </div>
                      
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Scopes:</span>
                          <span className="font-medium">{token.scopes.join(', ')}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Expires:</span>
                          <span className="font-medium">{formatDate(token.expires_at)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Last Used:</span>
                          <span className="font-medium">{formatDate(token.last_used)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Status:</span>
                          <span className={cn(
                            "font-medium",
                            token.is_active ? "text-green-600" : "text-red-600"
                          )}>
                            {token.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Key className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No User Selected</h3>
              <p className="text-gray-600 mb-4">
                Select a user from the Users tab to view their GitHub tokens
              </p>
              <Button onClick={() => setActiveTab('users')}>
                Go to Users
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Create User Modal */}
      {showCreateUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Create New User</h2>
            <form className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <Input type="text" placeholder="username" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <Input type="email" placeholder="email@example.com" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
                <Input type="text" placeholder="Full Name" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <Select>
                  <option value="user">User</option>
                  <option value="org_admin">Organization Admin</option>
                  <option value="admin">Admin</option>
                </Select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Organization</label>
                <Input type="text" placeholder="Organization Name" />
              </div>
              <div className="flex space-x-2 pt-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowCreateUser(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  className="flex-1"
                  onClick={(e) => {
                    e.preventDefault()
                    createUserMutation.mutate({})
                  }}
                >
                  Create User
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit User Modal */}
      {showEditUser && selectedUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Edit User</h2>
            <form className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <Input type="text" value={selectedUser.username} disabled />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <Input type="email" value={selectedUser.email} />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
                <Input type="text" value={selectedUser.full_name || ''} />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <Select value={selectedUser.role}>
                  <option value="user">User</option>
                  <option value="org_admin">Organization Admin</option>
                  <option value="admin">Admin</option>
                </Select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Organization</label>
                <Input type="text" value={selectedUser.organization || ''} />
              </div>
              <div className="flex space-x-2 pt-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowEditUser(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  className="flex-1"
                  onClick={(e) => {
                    e.preventDefault()
                    updateUserMutation.mutate({
                      userId: selectedUser.id,
                      userData: {}
                    })
                  }}
                >
                  Update User
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Add Token Modal */}
      {showAddToken && selectedUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Add GitHub Token</h2>
            <form className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Token Name</label>
                <Input type="text" placeholder="e.g., Personal Access Token" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">GitHub Token</label>
                <Input type="password" placeholder="ghp_..." />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Scopes</label>
                <div className="space-y-2">
                  {['repo', 'read:user', 'read:org', 'workflow'].map((scope) => (
                    <label key={scope} className="flex items-center">
                      <input type="checkbox" className="rounded mr-2" />
                      <span className="text-sm text-gray-700">{scope}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="flex space-x-2 pt-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowAddToken(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  className="flex-1"
                  onClick={(e) => {
                    e.preventDefault()
                    addTokenMutation.mutate({
                      userId: selectedUser.id,
                      tokenData: {}
                    })
                  }}
                >
                  Add Token
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
