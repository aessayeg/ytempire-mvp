import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from 'axios';

interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  is_verified: boolean;
  is_superuser: boolean;
  subscription_tier: string;
  channels_limit: number;
  videos_per_day_limit: number;
  total_videos_generated: number;
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string, full_name?: string) => Promise<void>;
  logout: () => void;
  refreshAccessToken: () => Promise<void>;
  fetchUser: () => Promise<void>;
  clearError: () => void;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await axios.post(`${API_URL}/api/v1/auth/login`, 
            new URLSearchParams({
              username: email,
              password: password,
            }),
            {
              headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
              },
            }
          );

          const { access_token, refresh_token } = response.data;
          
          set({
            accessToken: access_token,
            refreshToken: refresh_token,
            isAuthenticated: true,
            isLoading: false,
          });

          // Set default auth header
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

          // Fetch user data
          await get().fetchUser();
        } catch (error: any) {
          set({
            error: error.response?.data?.detail || 'Login failed',
            isLoading: false,
          });
          throw error;
        }
      },

      register: async (email: string, username: string, password: string, full_name?: string) => {
        set({ isLoading: true, error: null });
        try {
          await axios.post(`${API_URL}/api/v1/auth/register`, {
            email,
            username,
            password,
            full_name,
          });

          // Auto-login after registration
          await get().login(email, password);
        } catch (error: any) {
          set({
            error: error.response?.data?.detail || 'Registration failed',
            isLoading: false,
          });
          throw error;
        }
      },

      logout: () => {
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null,
        });
        delete axios.defaults.headers.common['Authorization'];
      },

      refreshAccessToken: async () => {
        const refreshToken = get().refreshToken;
        if (!refreshToken) {
          get().logout();
          return;
        }

        try {
          const response = await axios.post(`${API_URL}/api/v1/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token, refresh_token } = response.data;
          
          set({
            accessToken: access_token,
            refreshToken: refresh_token,
          });

          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
        } catch (error) {
          get().logout();
          throw error;
        }
      },

      fetchUser: async () => {
        const accessToken = get().accessToken;
        if (!accessToken) return;

        try {
          const response = await axios.get(`${API_URL}/api/v1/auth/me`, {
            headers: {
              Authorization: `Bearer ${accessToken}`,
            },
          });

          set({
            user: response.data,
            isAuthenticated: true,
          });
        } catch (error: any) {
          if (error.response?.status === 401) {
            // Try to refresh token
            try {
              await get().refreshAccessToken();
              await get().fetchUser();
            } catch {
              get().logout();
            }
          }
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
      }),
    }
  )
);