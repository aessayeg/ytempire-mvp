import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { authApi } from '../services/api'

interface User {
  id: string
  email: string
  username: string
  full_name?: string
  is_active: boolean
  is_superuser: boolean
}

interface AuthContextType {
  user: User | null
  token: string | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, username: string, password: string, full_name?: string) => Promise<void>
  logout: () => void
  updateUser: (userData: Partial<User>) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const [isLoading, setIsLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    const checkAuth = async () => {
      const storedToken = localStorage.getItem('token')
      if (storedToken) {
        try {
          const userData = await authApi.getCurrentUser(storedToken)
          setUser(userData)
          setToken(storedToken)
        } catch (error) {
          console.error('Token validation failed:', error)
          localStorage.removeItem('token')
          setToken(null)
        }
      }
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  const login = async (email: string, password: string) => {
    try {
      const response = await authApi.login(email, password)
      const { access_token, user: userData } = response
      
      localStorage.setItem('token', access_token)
      setToken(access_token)
      setUser(userData)
      
      toast.success('Login successful!')
      navigate('/dashboard')
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Login failed'
      toast.error(message)
      throw error
    }
  }

  const register = async (email: string, username: string, password: string, full_name?: string) => {
    try {
      const response = await authApi.register({
        email,
        username,
        password,
        full_name
      })
      
      toast.success('Registration successful! Please login.')
      navigate('/login')
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Registration failed'
      toast.error(message)
      throw error
    }
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
    navigate('/login')
    toast.success('Logged out successfully')
  }

  const updateUser = (userData: Partial<User>) => {
    setUser(prev => prev ? { ...prev, ...userData } : null)
  }

  const value = {
    user,
    token,
    isLoading,
    login,
    register,
    logout,
    updateUser
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}