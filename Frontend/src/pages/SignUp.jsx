import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FiMail, FiLock, FiUser, FiEye, FiEyeOff } from 'react-icons/fi';
import styles from '../styles/Auth.module.css';

export default function SignUp() {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: Wire up API endpoint
    console.log('Sign up:', formData);
  };

  return (
    <div className={styles.authPage}>
      <div className={styles.authCard}>
        <div className={styles.authHeader}>
          <div className={styles.logoIcon}>G</div>
          <h1>Create Account</h1>
          <p>Start your intelligent document journey</p>
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

          <div className={styles.inputGroup}>
            <FiLock className={styles.inputIcon} />
            <input
              type={showPassword ? 'text' : 'password'}
              placeholder="Confirm password"
              value={formData.confirmPassword}
              onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
              required
            />
          </div>

          <button type="submit" className={styles.submitBtn}>
            Create Account
          </button>
        </form>

        <div className={styles.authDivider}>
          <span>or</span>
        </div>

        <button className={styles.googleBtn}>
          Continue with Google
        </button>

        <p className={styles.authFooter}>
          Already have an account?{' '}
          <Link to="/auth/signin" className={styles.link}>
            Sign In
          </Link>
        </p>
      </div>
    </div>
  );
}
