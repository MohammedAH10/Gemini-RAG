import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { authService } from '../services/auth';
import { useAuthStore } from '../store/authStore';
import toast from 'react-hot-toast';

export default function GoogleCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleCallback = async () => {
      const code = searchParams.get('code');
      const accessToken = searchParams.get('access_token');
      const callbackError = searchParams.get('error');

      if (callbackError) {
        setError('Authentication cancelled');
        toast.error('Google authentication cancelled');
        setTimeout(() => navigate('/login'), 2000);
        return;
      }

      // Flow 1: Backend redirect with tokens in URL (Google OAuth redirect flow)
      if (accessToken) {
        const userId = searchParams.get('user_id') || '';
        const email = searchParams.get('email') || '';

        const user = { id: userId, email };
        localStorage.setItem('access_token', accessToken);
        localStorage.setItem('user', JSON.stringify(user));

        // Update Zustand store so the current SPA session is authenticated immediately
        useAuthStore.setState({
          user,
          token: accessToken,
          isAuthenticated: true,
          loading: false,
          error: null,
        });

        toast.success('Signed in with Google successfully!');
        navigate('/dashboard', { replace: true });
        return;
      }

      // Flow 2: SPA POST flow with code
      if (code) {
        try {
          await authService.handleGoogleCallback(code);
          toast.success('Signed in with Google successfully!');
          navigate('/dashboard', { replace: true });
        } catch (err) {
          setError(err.response?.data?.detail || 'Google authentication failed');
          toast.error('Google authentication failed');
          setTimeout(() => navigate('/login'), 2000);
        }
        return;
      }

      setError('Invalid authentication response');
      toast.error('Invalid authentication response');
      setTimeout(() => navigate('/login'), 2000);
    };

    handleCallback();
  }, [searchParams, navigate]);

  return (
    <div className="min-h-screen bg-dark flex items-center justify-center">
      <div className="text-center">
        {error ? (
          <>
            <div className="text-red-400 text-xl mb-4">{error}</div>
            <div className="text-white/60">Redirecting...</div>
          </>
        ) : (
          <>
            <div className="w-16 h-16 mx-auto mb-4 border-4 border-neon-blue border-t-transparent rounded-full animate-spin" />
            <div className="text-white/80 text-lg">Completing Google authentication...</div>
          </>
        )}
      </div>
    </div>
  );
}
