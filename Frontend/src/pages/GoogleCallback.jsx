import { useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import useAuthStore from '../context/authStore';
import styles from '../styles/Auth.module.css';

export default function GoogleCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { setAuth } = useAuthStore();

  useEffect(() => {
    const accessToken = searchParams.get('access_token');
    const refreshToken = searchParams.get('refresh_token');
    const userId = searchParams.get('user_id');
    const email = searchParams.get('email');

    if (accessToken && userId) {
      setAuth({ user_id: userId, email }, accessToken, refreshToken);
      navigate('/dashboard');
    } else {
      navigate('/auth/signin');
    }
  }, [searchParams, setAuth, navigate]);

  return (
    <div className={styles.authPage}>
      <div className={styles.loadingContainer}>
        <div className={styles.spinner} />
        <p>Completing authentication...</p>
      </div>
    </div>
  );
}
