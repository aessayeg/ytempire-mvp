import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import axios from 'axios';

interface User {
  id: number;
  email: string;
  full_name: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  _error: string | null;
  
  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  register: (email: string, password: string, fullName: string) => Promise<void>;
  refreshToken: () => Promise<void>;
  clearError: () => void;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      _error: null,
      
      login: async (email: string, password: string) => {
        set({ isLoading: true, _error: null });
        
        try {
          const response = await axios.post(`${API_URL}/api/v1/auth/login`, {
            username: email,
            password,
          });
          
          const { access_token, user } = response.data;
          
          // Set axios default header
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
            _error: null,
          });
        } catch (_error: unknown) {
          const errorMessage = error && typeof error === 'object' && 'response' in error 
            ? (error as any).response?.data?.detail || 'Login failed'
            : 'Login failed';
          set({
            isLoading: false,
            _error: errorMessage,
            isAuthenticated: false,
          });
          throw error;
        }
      },
      
      logout: () => {
        delete axios.defaults.headers.common['Authorization'];
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          _error: null,
        });
      },
      
      register: async (email: string, password: string, fullName: string) => {
        set({ isLoading: true, _error: null });
        
        try {
          await axios.post(`${API_URL}/api/v1/auth/register`, {
            email,
            password,
            full_name: fullName,
          });
          
          // Auto-login after registration
          await get().login(email, password);
        } catch (_error: unknown) {
          const errorMessage = error && typeof error === 'object' && 'response' in error 
            ? (error as any).response?.data?.detail || 'Registration failed'
            : 'Registration failed';
          set({
            isLoading: false,
            _error: errorMessage,
          });
          throw error;
        }
      },
      
      refreshToken: async () => {
        const token = get().token;
        if (!token) return;
        
        try {
          const response = await axios.post(`${API_URL}/api/v1/auth/refresh`, {
            token,
          });
          
          const { access_token } = response.data;
          
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({ token: access_token });
        } catch (_error) {
          get().logout();
        }
      },
      
      clearError: () => set({ _error: null }),
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// Initialize axios interceptor for token refresh
axios.interceptors.response.use(
  (response) => response,
  async (_error) => {
    const originalRequest = error.config;
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        await useAuthStore.getState().refreshToken();
        return axios(originalRequest);
      } catch (_refreshError) {
        useAuthStore.getState().logout();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(_error);
  }
);