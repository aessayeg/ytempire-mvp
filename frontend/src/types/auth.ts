/**
 * Authentication Types
 * Owner: React Engineer
 */

export interface User {
  id: string
  email: string
  firstName: string
  lastName: string
  fullName?: string
  companyName?: string
  roles: string[]
  isActive: boolean
  isEmailVerified: boolean
  avatar?: string
  createdAt: string
  updatedAt: string
  lastLoginAt?: string
  
  // Subscription info
  subscriptionPlan?: 'free' | 'pro' | 'enterprise'
  subscriptionStatus?: 'active' | 'canceled' | 'past_due' | 'trialing'
  subscriptionExpiresAt?: string
  
  // Usage limits
  maxChannels: number
  maxVideosPerDay: number
  currentChannelCount: number
  
  // Preferences
  timezone?: string
  language?: string
  emailNotifications?: boolean
  marketingEmails?: boolean
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterCredentials {
  firstName: string
  lastName: string
  email: string
  password: string
  companyName?: string
  marketingEmails?: boolean
}

export interface AuthTokens {
  accessToken: string
  refreshToken: string
  tokenType: 'Bearer'
  expiresIn: number
}

export interface AuthResponse {
  user: User
  access_token: string
  refresh_token: string
  token_type: 'Bearer'
  expires_in: number
}

export interface ForgotPasswordRequest {
  email: string
}

export interface ResetPasswordRequest {
  token: string
  password: string
}

export interface UpdateProfileRequest {
  firstName?: string
  lastName?: string
  companyName?: string
  timezone?: string
  language?: string
  emailNotifications?: boolean
  marketingEmails?: boolean
}

export interface AuthError {
  message: string
  code?: string
  details?: Record<string, any>
}

export interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  isLoading: boolean
  error: string | null
  isInitialized: boolean
}

// OAuth provider types
export type OAuthProvider = 'google' | 'github' | 'microsoft'

export interface OAuthCallbackData {
  code: string
  state?: string
  error?: string
  error_description?: string
}

// JWT token payload interface
export interface JWTPayload {
  sub: string // user id
  email: string
  roles: string[]
  exp: number
  iat: number
  iss: string
}

// Permission and role types
export type UserRole = 'user' | 'admin' | 'moderator'

export interface Permission {
  resource: string
  actions: string[]
}

export interface RolePermissions {
  [role: string]: Permission[]
}