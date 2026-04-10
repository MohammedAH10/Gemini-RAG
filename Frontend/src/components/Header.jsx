import { useNavigate } from 'react-router-dom';
import { FiBell, FiLogOut } from 'react-icons/fi';
import useAuthStore from '../context/authStore';
import styles from '../styles/Header.module.css';

export default function Header() {
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/auth/signin');
  };

  return (
    <header className={styles.header}>
      <div className={styles.headerLeft}>
        <h2 className={styles.pageTitle}>
          {/* Page title will be set by individual pages */}
        </h2>
      </div>

      <div className={styles.headerRight}>
        <button className={styles.iconBtn} aria-label="Notifications">
          <FiBell />
        </button>

        <div className={styles.userMenu}>
          <div className={styles.userAvatar}>
            {user?.email?.charAt(0).toUpperCase() || 'U'}
          </div>
          <span className={styles.userName}>{user?.email || 'User'}</span>
        </div>

        <button className={styles.logoutBtn} onClick={handleLogout} aria-label="Logout">
          <FiLogOut />
        </button>
      </div>
    </header>
  );
}
