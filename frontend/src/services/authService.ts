/**
 * Authentication Service
 * Owner: Frontend Team Lead
 */

import { apiClient } from '@/utils/api'
import { User, LoginCredentials, RegisterCredentials, AuthResponse } from '@/types/auth'

export class AuthService {
  /**
   * Login user with email and password
   */
  static async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await apiClient.login(credentials.email, credentials.password)
    return response.data
  }

  /**
   * Register new user
   */
  static async register(credentials: RegisterCredentials): Promise<AuthResponse> {
    const response = await apiClient.register(credentials)
    return response.data
  }

  /**
   * Logout user
   */
  static async logout(): Promise<void> {
    await apiClient.logout()
  }

  /**
   * Refresh access token
   */
  static async refreshToken(refreshToken: string): Promise<{ access_token: string; refresh_token: string }> {
    const response = await apiClient.refreshToken(refreshToken)
    return response.data
  }

  /**
   * Get current user profile
   */
  static async getCurrentUser(): Promise<User> {
    const response = await apiClient.getCurrentUser()
    return response.data
  }

  /**
   * Update user profile
   */
  static async updateProfile(updates: Partial<User>): Promise<User> {
    const response = await apiClient.updateCurrentUser(updates)
    return response.data
  }

  /**
   * Request password reset
   */
  static async forgotPassword(email: string): Promise<void> {
    await apiClient.post('/auth/forgot-password', { email })
  }

  /**
   * Reset password with token
   */
  static async resetPassword(token: string, password: string): Promise<void> {
    await apiClient.post('/auth/reset-password', { token, password })
  }

  /**
   * Verify email address
   */
  static async verifyEmail(token: string): Promise<void> {
    await apiClient.post('/auth/verify-email', { token })
  }

  /**
   * Resend email verification
   */
  static async resendVerification(email: string): Promise<void> {
    await apiClient.post('/auth/resend-verification', { email })
  }

  /**
   * Change password (authenticated user)
   */
  static async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    await apiClient.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword
    })
  }

  /**
   * Get user usage statistics
   */
  static async getUserUsage(): Promise<{
    current_channels: number
    current_videos_today: number
    total_videos_this_month: number
    total_cost_this_month: number
  }> {
    const response = await apiClient.getUserUsage()
    return response.data
  }

  /**
   * Check if email is available
   */
  static async checkEmailAvailability(email: string): Promise<{ available: boolean }> {
    const response = await apiClient.get('/auth/check-email', { params: { email } })
    return response.data
  }

  /**
   * Get user permissions
   */
  static async getUserPermissions(): Promise<string[]> {
    const response = await apiClient.get('/auth/permissions')
    return response.data.permissions
  }

  /**
   * Enable 2FA
   */
  static async enable2FA(): Promise<{ qr_code: string; backup_codes: string[] }> {
    const response = await apiClient.post('/auth/2fa/enable')
    return response.data
  }

  /**
   * Disable 2FA
   */
  static async disable2FA(token: string): Promise<void> {
    await apiClient.post('/auth/2fa/disable', { token })
  }

  /**
   * Verify 2FA token
   */
  static async verify2FA(token: string): Promise<void> {
    await apiClient.post('/auth/2fa/verify', { token })
  }
}

export default AuthService