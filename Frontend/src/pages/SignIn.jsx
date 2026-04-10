import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FiMail, FiLock, FiEye, FiEyeOff } from 'react-icons/fi';
import styles from '../styles/Auth.module.css';

export default function SignIn() {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: Wire up API endpoint
    console.log('Sign in:', formData);
  };

  return (
    <div className={styles.authPage}>
      <div className={styles.authCard}>
        <div className={styles.authHeader}>
          <div className={styles.logoIcon}>G</div>
          <h1>Welcome Back</h1>
          <p>Sign in to continue your session</p>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.inputGroup}>
            <FiMail className={styles.inputIcon} />
            <input
              type="email"
              placeholder="Email address"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              required
            />
          </div>

          <div className={styles.inputGroup}>
            <FiLock className={styles.inputIcon} />
            <input
              type={showPassword ? 'text' : 'password'}
              placeholder="Password"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              required
            />
            <button
              type="button"
              className={styles.eyeBtn}
              onClick={() => setShowPassword(!showPassword)}
            >
              {showPassword ? <FiEyeOff /> : <FiEye />}
            </button>
          </div>

          <div className={styles.forgotRow}>
            <Link to="/auth/forgot-password" className={styles.forgotLink}>
              Forgot password?
            </Link>
          </div>

          <button type="submit" className={styles.submitBtn}>
            Sign In
          </button>
        </form>

        <div className={styles.authDivider}>
          <span>or</span>
        </div>

        <button className={styles.googleBtn}>
          Continue with Google
        </button>

        <p className={styles.authFooter}>
          Don&apos;t have an account?{' '}
          <Link to="/auth/signup" className={styles.link}>
            Sign Up
          </Link>
        </p>
      </div>
    </div>
  );
}
