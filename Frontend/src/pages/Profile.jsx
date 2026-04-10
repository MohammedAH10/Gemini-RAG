import { useState } from 'react';
import { FiUser, FiMail, FiShield, FiKey, FiLogOut, FiEdit2 } from 'react-icons/fi';
import useAuthStore from '../context/authStore';
import styles from '../styles/Profile.module.css';

export default function Profile() {
  const { user, logout } = useAuthStore();
  const [editing, setEditing] = useState(false);

  return (
    <div className={styles.profilePage}>
      <div className={styles.profileHeader}>
        <h1>Profile</h1>
        <button className={styles.editBtn} onClick={() => setEditing(!editing)}>
          <FiEdit2 /> {editing ? 'Cancel' : 'Edit Profile'}
        </button>
      </div>

      <div className={styles.profileGrid}>
        <div className={styles.profileCard}>
          <div className={styles.avatarSection}>
            <div className={styles.avatarLarge}>
              {user?.email?.charAt(0).toUpperCase() || 'U'}
            </div>
            <h2>{user?.email || 'User'}</h2>
          </div>

          <div className={styles.infoList}>
            <div className={styles.infoRow}>
              <FiMail />
              <span>Email</span>
              <strong>{user?.email || 'N/A'}</strong>
            </div>
            <div className={styles.infoRow}>
              <FiShield />
              <span>Role</span>
              <strong>User</strong>
            </div>
          </div>
        </div>

        <div className={styles.profileCard}>
          <h3>Security</h3>
          <div className={styles.securityActions}>
            <button className={styles.securityBtn}>
              <FiKey /> Change Password
            </button>
          </div>
        </div>

        <div className={styles.profileCard}>
          <h3>Usage Statistics</h3>
          <div className={styles.statsGrid}>
            <div className={styles.miniStat}>
              <span className={styles.miniStatLabel}>Queries</span>
              <span className={styles.miniStatValue}>0</span>
            </div>
            <div className={styles.miniStat}>
              <span className={styles.miniStatLabel}>Documents</span>
              <span className={styles.miniStatValue}>0</span>
            </div>
            <div className={styles.miniStat}>
              <span className={styles.miniStatLabel}>Storage Used</span>
              <span className={styles.miniStatValue}>0 MB</span>
            </div>
          </div>
        </div>

        <div className={styles.profileCard}>
          <h3>Rate Limits</h3>
          <div className={styles.rateLimitInfo}>
            <p>Current rate limit usage will be displayed here</p>
          </div>
        </div>
      </div>

      <div className={styles.dangerZone}>
        <h3>Danger Zone</h3>
        <button className={styles.dangerBtn}>
          <FiLogOut /> Sign Out All Devices
        </button>
      </div>
    </div>
  );
}
