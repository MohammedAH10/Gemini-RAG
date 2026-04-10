import { NavLink } from 'react-router-dom';
import {
  FiHome,
  FiMessageSquare,
  FiFileText,
  FiUploadCloud,
  FiSearch,
  FiDatabase,
  FiSettings,
  FiClock,
  FiUser,
  FiActivity,
} from 'react-icons/fi';
import styles from '../styles/Sidebar.module.css';

export default function Sidebar() {
  const navItems = [
    { to: '/dashboard', icon: FiHome, label: 'Dashboard' },
    { to: '/chat', icon: FiMessageSquare, label: 'Chat' },
    { to: '/documents', icon: FiFileText, label: 'Documents' },
    { to: '/documents/upload', icon: FiUploadCloud, label: 'Upload' },
    { to: '/explore', icon: FiSearch, label: 'Explore' },
    { to: '/vector-store', icon: FiDatabase, label: 'Vector Store' },
    { to: '/history', icon: FiClock, label: 'History' },
    { to: '/settings', icon: FiSettings, label: 'Settings' },
    { to: '/profile', icon: FiUser, label: 'Profile' },
    { to: '/health', icon: FiActivity, label: 'Health' },
  ];

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>
          <span className={styles.logoText}>G</span>
        </div>
        <span className={styles.logoTitle}>Gemini RAG</span>
      </div>

      <nav className={styles.nav}>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `${styles.navItem} ${isActive ? styles.navItemActive : ''}`
            }
          >
            <item.icon className={styles.navIcon} />
            <span className={styles.navLabel}>{item.label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
