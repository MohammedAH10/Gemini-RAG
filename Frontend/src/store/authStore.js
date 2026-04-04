import { create } from 'zustand';
import { authService } from '../services/auth';

// Initialize auth state from localStorage synchronously
const getStoredToken = () => localStorage.getItem('access_token');
const getStoredUser = () => {
  const userStr = localStorage.getItem('user');
  return userStr ? JSON.parse(userStr) : null;
};

export const useAuthStore = create((set) => ({
  user: getStoredUser(),
  token: getStoredToken(),
  isAuthenticated: !!getStoredToken(),
  loading: false,
  error: null,

  // Re-check localStorage on demand (call this on app mount)
  initializeAuth: () => {
    const token = getStoredToken();
    const user = getStoredUser();
    set({
      user,
      token,
      isAuthenticated: !!token,
    });
  },

  login: async (email, password) => {
    set({ loading: true, error: null });
    try {
      const data = await authService.signin(email, password);
      if (!data.access_token) {
        throw new Error('No access token returned from server');
      }
      localStorage.setItem('access_token', data.access_token);
      if (data.user) {
        localStorage.setItem('user', JSON.stringify(data.user));
      }
      set({
        user: data.user || { email: data.email, id: data.user_id },
        token: data.access_token,
        isAuthenticated: true,
        loading: false,
        error: null,
      });
      return data;
    } catch (error) {
      set({ error: error.response?.data?.detail || 'Login failed', loading: false });
      throw error;
    }
  },

  signup: async (email, password) => {
    set({ loading: true, error: null });
    try {
      const data = await authService.signup(email, password);
      if (!data.access_token) {
        // Email confirmation required - don't set auth
        set({ loading: false });
        return data;
      }
      localStorage.setItem('access_token', data.access_token);
      if (data.user) {
        localStorage.setItem('user', JSON.stringify(data.user));
      }
      set({
        user: data.user || { email: data.email, id: data.user_id },
        token: data.access_token,
        isAuthenticated: true,
        loading: false,
        error: null,
      });
      return data;
    } catch (error) {
      set({ error: error.response?.data?.detail || 'Signup failed', loading: false });
      throw error;
    }
  },

  googleAuth: async () => {
    set({ loading: true, error: null });
    try {
      // This will redirect to Google's OAuth page
      await authService.googleAuth();
      // The page will redirect back after authentication
      return null;
    } catch (error) {
      set({ error: error.response?.data?.detail || 'Google authentication failed', loading: false });
      throw error;
    }
  },

  logout: async () => {
    await authService.signout();
    set({ user: null, token: null, isAuthenticated: false });
  },

  updateUser: (user) => {
    localStorage.setItem('user', JSON.stringify(user));
    set({ user });
  },

  clearError: () => set({ error: null }),
}));
