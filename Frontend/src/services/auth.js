import api from './api';

export const authService = {
  signup: async (email, password) => {
    const response = await api.post('/auth/signup', { email, password });
    if (response.data.access_token) {
      localStorage.setItem('access_token', response.data.access_token);
    }
    return response.data;
  },

  signin: async (email, password) => {
    const response = await api.post('/auth/signin', { email, password });
    if (response.data.access_token) {
      localStorage.setItem('access_token', response.data.access_token);
    }
    return response.data;
  },

  signout: async () => {
    try {
      await api.post('/auth/signout');
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
    }
  },

  getCurrentUser: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  googleAuth: () => {
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
    const FRONTEND_URL = import.meta.env.VITE_FRONTEND_URL || window.location.origin;

    // Redirect to Google auth endpoint with callback URL
    const googleAuthUrl = `${API_BASE_URL}/auth/google?redirect_uri=${encodeURIComponent(FRONTEND_URL + '/auth/google/callback')}`;
    window.location.href = googleAuthUrl;
  },

  handleGoogleCallback: async (code) => {
    const response = await api.post('/auth/google/callback', { code });
    if (response.data.access_token) {
      localStorage.setItem('access_token', response.data.access_token);
      if (response.data.user) {
        localStorage.setItem('user', JSON.stringify(response.data.user));
      }
    }
    return response.data;
  },
};
