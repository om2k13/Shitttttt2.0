import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Settings as SettingsIcon, 
  Github, 
  User, 
  Shield, 
  Key,
  Eye,
  EyeOff,
  Plus,
  Trash2,
  CheckCircle,
  AlertCircle,
  Info,
  Save,
  Loader2
} from 'lucide-react'
import { toast } from 'sonner'
import { usersApi } from '@/lib/api'
import { cn } from '@/lib/utils'

interface UserProfile {
  id: number
  username: string
  email: string
  full_name: string
  organization: string
  github_username: string
  created_at: string
  last_login: string
}

interface GitHubToken {
  id: number
  token_name: string
  created_at: string
  last_used: string
  is_active: boolean
}

export function Settings() {
  const [showToken, setShowToken] = useState(false)
  const [newToken, setNewToken] = useState('')
  const [tokenName, setTokenName] = useState('Code Review Agent Token')
  const [activeTab, setActiveTab] = useState<'profile' | 'github' | 'security'>('profile')
  const [profileForm, setProfileForm] = useState({
    full_name: '',
    organization: '',
    github_username: ''
  })
  const [isEditing, setIsEditing] = useState(false)
  
  const queryClient = useQueryClient()

  // Fetch user profile
  const { data: userProfile, isLoading } = useQuery({
    queryKey: ['user-profile'],
    queryFn: async () => {
      try {
        const response = await usersApi.getProfile()
        const profile = response.data
        setProfileForm({
          full_name: profile.full_name || '',
          organization: profile.organization || '',
          github_username: profile.github_username || ''
        })
        return profile
      } catch (error) {
        console.error('Failed to fetch profile:', error)
        // Return default profile if API fails
        return {
          id: 1,
          username: 'demo_user',
          email: 'demo@example.com',
          full_name: 'Demo User',
          organization: 'Demo Organization',
          github_username: 'demo_user',
          created_at: new Date().toISOString(),
          last_login: new Date().toISOString()
        }
      }
    },
  })

  // Mutations
  const updateProfileMutation = useMutation({
    mutationFn: (data: { full_name: string; organization: string; github_username: string }) =>
      usersApi.updateUserProfile(1, data),
    onSuccess: () => {
      toast.success('Profile updated successfully')
      setIsEditing(false)
      queryClient.invalidateQueries({ queryKey: ['user-profile'] })
    },
    onError: (error) => {
      toast.error('Failed to update profile')
      console.error('Update error:', error)
    },
  })

  const addTokenMutation = useMutation({
    mutationFn: (data: { token: string; name: string }) =>
      usersApi.addGitHubToken(1, data.token, data.name),
    onSuccess: () => {
      toast.success('GitHub token added successfully')
      setNewToken('')
      setTokenName('Code Review Agent Token')
      queryClient.invalidateQueries({ queryKey: ['user-profile'] })
    },
    onError: (error) => {
      toast.error('Failed to add GitHub token')
      console.error('Add token error:', error)
    },
  })

  const revokeTokenMutation = useMutation({
    mutationFn: (tokenId: number) =>
      usersApi.revokeGitHubToken(1, tokenId),
    onSuccess: () => {
      toast.success('GitHub token revoked successfully')
      queryClient.invalidateQueries({ queryKey: ['user-profile'] })
    },
    onError: (error) => {
      toast.error('Failed to revoke GitHub token')
      console.error('Revoke token error:', error)
    },
  })

  const testTokenMutation = useMutation({
    mutationFn: () => usersApi.testGitHubToken(1),
    onSuccess: (data) => {
      if (data.data?.token_valid) {
        toast.success('GitHub token is valid and working')
      } else {
        toast.error('GitHub token is invalid or expired')
      }
    },
    onError: (error) => {
      toast.error('Failed to test GitHub token')
      console.error('Test token error:', error)
    },
  })

  const handleUpdateProfile = () => {
    updateProfileMutation.mutate(profileForm)
  }

  const handleAddToken = () => {
    if (!newToken.trim()) {
      toast.error('Please enter a GitHub token')
      return
    }
    if (!tokenName.trim()) {
      toast.error('Please enter a token name')
      return
    }
    addTokenMutation.mutate({ token: newToken, name: tokenName })
  }

  const handleRevokeToken = (tokenId: number) => {
    if (confirm('Are you sure you want to revoke this token? This action cannot be undone.')) {
      revokeTokenMutation.mutate(tokenId)
    }
  }

  const handleTestToken = () => {
    testTokenMutation.mutate()
  }

  const handleProfileChange = (field: string, value: string) => {
    setProfileForm(prev => ({
      ...prev,
      [field]: value
    }))
  }

  if (isLoading) {
    return (
      <div className="text-center py-8">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
        <p className="text-gray-500">Loading settings...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="mt-2 text-gray-600">
          Manage your profile, GitHub tokens, and security settings
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'profile', name: 'Profile', icon: User },
            { id: 'github', name: 'GitHub', icon: Github },
            { id: 'security', name: 'Security', icon: Shield },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                'flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm',
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              )}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.name}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'profile' && (
        <div className="space-y-6">
          {/* Profile Information */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="card-title">Profile Information</h2>
                  <p className="card-description">
                    Update your personal information and organization details
                  </p>
                </div>
                <div className="flex space-x-2">
                  {isEditing ? (
                    <>
                      <button
                        onClick={() => setIsEditing(false)}
                        className="btn-outline btn-sm"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleUpdateProfile}
                        disabled={updateProfileMutation.isPending}
                        className="btn-primary btn-sm"
                      >
                        <Save className="h-4 w-4 mr-2" />
                        {updateProfileMutation.isPending ? 'Saving...' : 'Save'}
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => setIsEditing(true)}
                      className="btn-outline btn-sm"
                    >
                      Edit Profile
                    </button>
                  )}
                </div>
              </div>
            </div>
            <div className="card-content">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="full-name" className="block text-sm font-medium text-gray-700 mb-2">
                    Full Name
                  </label>
                  <input
                    id="full-name"
                    type="text"
                    value={profileForm.full_name}
                    onChange={(e) => handleProfileChange('full_name', e.target.value)}
                    disabled={!isEditing}
                    className="input w-full disabled:bg-gray-50"
                    placeholder="Enter your full name"
                  />
                </div>

                <div>
                  <label htmlFor="organization" className="block text-sm font-medium text-gray-700 mb-2">
                    Organization
                  </label>
                  <input
                    id="organization"
                    type="text"
                    value={profileForm.organization}
                    onChange={(e) => handleProfileChange('organization', e.target.value)}
                    disabled={!isEditing}
                    className="input w-full disabled:bg-gray-50"
                    placeholder="Enter your organization"
                  />
                </div>

                <div>
                  <label htmlFor="github-username" className="block text-sm font-medium text-gray-700 mb-2">
                    GitHub Username
                  </label>
                  <input
                    id="github-username"
                    type="text"
                    value={profileForm.github_username}
                    onChange={(e) => handleProfileChange('github_username', e.target.value)}
                    disabled={!isEditing}
                    className="input w-full disabled:bg-gray-50"
                    placeholder="Enter your GitHub username"
                  />
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                    Email Address
                  </label>
                  <input
                    id="email"
                    type="email"
                    value={userProfile?.email || ''}
                    disabled
                    className="input w-full bg-gray-50"
                  />
                  <p className="text-xs text-gray-500 mt-1">Email cannot be changed</p>
                </div>
              </div>

              {isEditing && (
                <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center space-x-2">
                    <Info className="h-4 w-4 text-blue-600" />
                    <span className="text-sm text-blue-800">
                      Click Save to apply your changes
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Account Statistics */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Account Statistics</h2>
              <p className="card-description">
                Overview of your code review activity
              </p>
            </div>
            <div className="card-content">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {userProfile?.statistics?.total_jobs || 0}
                  </div>
                  <div className="text-sm text-gray-600">Total Jobs</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {userProfile?.statistics?.completed_jobs || 0}
                  </div>
                  <div className="text-sm text-gray-600">Completed Jobs</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {userProfile?.statistics?.total_findings || 0}
                  </div>
                  <div className="text-sm text-gray-600">Total Findings</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {userProfile?.github_tokens?.length || 0}
                  </div>
                  <div className="text-sm text-gray-600">Active Tokens</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'github' && (
        <div className="space-y-6">
          {/* Add New Token */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Add GitHub Token</h2>
              <p className="card-description">
                Add a new GitHub personal access token for repository access
              </p>
            </div>
            <div className="card-content">
              <div className="space-y-4">
                <div>
                  <label htmlFor="token-name" className="block text-sm font-medium text-gray-700 mb-2">
                    Token Name
                  </label>
                  <input
                    id="token-name"
                    type="text"
                    value={tokenName}
                    onChange={(e) => setTokenName(e.target.value)}
                    className="input w-full"
                    placeholder="Enter a name for this token"
                  />
                </div>

                <div>
                  <label htmlFor="github-token" className="block text-sm font-medium text-gray-700 mb-2">
                    GitHub Personal Access Token
                  </label>
                  <div className="relative">
                    <input
                      id="github-token"
                      type={showToken ? 'text' : 'password'}
                      value={newToken}
                      onChange={(e) => setNewToken(e.target.value)}
                      className="input w-full pr-10"
                      placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                    />
                    <button
                      type="button"
                      onClick={() => setShowToken(!showToken)}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    >
                      {showToken ? (
                        <EyeOff className="h-4 w-4 text-gray-400" />
                      ) : (
                        <Eye className="h-4 w-4 text-gray-400" />
                      )}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Token should have 'repo' scope for private repository access
                  </p>
                </div>

                <button
                  onClick={handleAddToken}
                  disabled={!newToken.trim() || !tokenName.trim() || addTokenMutation.isPending}
                  className="btn-primary"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  {addTokenMutation.isPending ? 'Adding...' : 'Add Token'}
                </button>
              </div>

              <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-yellow-600" />
                  <span className="text-sm text-yellow-800">
                    <strong>Security Note:</strong> GitHub tokens are encrypted and stored securely. 
                    Never share your tokens with anyone.
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Existing Tokens */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">GitHub Tokens</h2>
              <p className="card-description">
                Manage your existing GitHub tokens
              </p>
            </div>
            <div className="card-content">
              {userProfile?.github_tokens && userProfile.github_tokens.length > 0 ? (
                <div className="space-y-4">
                  {userProfile.github_tokens.map((token: GitHubToken) => (
                    <div key={token.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3">
                          <Github className="h-5 w-5 text-gray-600" />
                          <div>
                            <h3 className="font-medium text-gray-900">{token.token_name}</h3>
                            <p className="text-sm text-gray-500">
                              Added {new Date(token.created_at).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={handleTestToken}
                          disabled={testTokenMutation.isPending}
                          className="btn-outline btn-sm"
                        >
                          <CheckCircle className="h-4 w-4 mr-2" />
                          {testTokenMutation.isPending ? 'Testing...' : 'Test'}
                        </button>
                        
                        <button
                          onClick={() => handleRevokeToken(token.id)}
                          disabled={revokeTokenMutation.isPending}
                          className="btn-outline btn-sm border-red-200 text-red-700 hover:bg-red-50"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          {revokeTokenMutation.isPending ? 'Revoking...' : 'Revoke'}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Github className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No GitHub tokens found</p>
                  <p className="text-gray-400 mt-1">Add a token above to get started</p>
                </div>
              )}
            </div>
          </div>

          {/* GitHub Integration Info */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">GitHub Integration</h2>
              <p className="card-description">
                How GitHub tokens are used in the code review agent
              </p>
            </div>
            <div className="card-content">
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium text-gray-900">Repository Access</h4>
                    <p className="text-sm text-gray-600">
                      Tokens provide access to private repositories and organization repositories
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium text-gray-900">PR Integration</h4>
                    <p className="text-sm text-gray-600">
                      Post findings as comments and create comprehensive PR reviews
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium text-gray-900">Security</h4>
                    <p className="text-sm text-gray-600">
                      All tokens are encrypted and stored securely in the database
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'security' && (
        <div className="space-y-6">
          {/* Security Overview */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Security Settings</h2>
              <p className="card-description">
                Manage your account security and access controls
              </p>
            </div>
            <div className="card-content">
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                  <div>
                    <h3 className="font-medium text-gray-900">Two-Factor Authentication</h3>
                    <p className="text-sm text-gray-600">Add an extra layer of security to your account</p>
                  </div>
                  <button className="btn-outline btn-sm" disabled>
                    <Shield className="h-4 w-4 mr-2" />
                    Coming Soon
                  </button>
                </div>
                
                <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                  <div>
                    <h3 className="font-medium text-gray-900">Session Management</h3>
                    <p className="text-sm text-gray-600">View and manage active sessions</p>
                  </div>
                  <button className="btn-outline btn-sm" disabled>
                    <Key className="h-4 w-4 mr-2" />
                    Coming Soon
                  </button>
                </div>
                
                <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                  <div>
                    <h3 className="font-medium text-gray-900">API Access</h3>
                    <p className="text-sm text-gray-600">Manage API keys and access tokens</p>
                  </div>
                  <button className="btn-outline btn-sm" disabled>
                    <SettingsIcon className="h-4 w-4 mr-2" />
                    Coming Soon
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Security Recommendations */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Security Recommendations</h2>
              <p className="card-description">
                Best practices to keep your account secure
              </p>
            </div>
            <div className="card-content">
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-gray-900">Use Strong Passwords</h4>
                    <p className="text-sm text-gray-600">
                      Ensure your password is unique and contains a mix of characters
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-gray-900">Enable Two-Factor Authentication</h4>
                    <p className="text-sm text-gray-600">
                      Add an extra layer of security with 2FA (coming soon)
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-gray-900">Regular Token Rotation</h4>
                    <p className="text-sm text-gray-600">
                      Rotate GitHub tokens periodically for better security
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-gray-900">Monitor Account Activity</h4>
                    <p className="text-sm text-gray-600">
                      Regularly review your account activity and sessions
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
