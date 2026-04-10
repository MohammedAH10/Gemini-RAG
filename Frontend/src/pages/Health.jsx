import { useState, useEffect } from 'react';
import { FiActivity, FiCheckCircle, FiXCircle, FiLoader, FiDatabase, FiCpu, FiMessageSquare } from 'react-icons/fi';
import styles from '../styles/Health.module.css';

export default function Health() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // TODO: Wire up API endpoint
    console.log('Fetch health status');
    setLoading(false);
  }, []);

  return (
    <div className={styles.healthPage}>
      <div className={styles.healthHeader}>
        <h1>System Health</h1>
        <p>Monitor backend services and component status</p>
      </div>

      {loading ? (
        <div className={styles.loadingState}>
          <FiLoader className={styles.spin} />
          <p>Checking system health...</p>
        </div>
      ) : (
        <>
          <div className={styles.overallStatus}>
            <FiCheckCircle className={styles.statusIcon} />
            <h2>System Operational</h2>
          </div>

          <div className={styles.componentsGrid}>
            <div className={styles.componentCard}>
              <div className={styles.componentHeader}>
                <FiActivity className={styles.componentIcon} />
                <h3>API Server</h3>
              </div>
              <div className={`${styles.statusBadge} ${styles.statusHealthy}`}>
                Healthy
              </div>
            </div>

            <div className={styles.componentCard}>
              <div className={styles.componentHeader}>
                <FiDatabase className={styles.componentIcon} />
                <h3>Vector Store</h3>
              </div>
              <div className={`${styles.statusBadge} ${styles.statusHealthy}`}>
                Healthy
              </div>
            </div>

            <div className={styles.componentCard}>
              <div className={styles.componentHeader}>
                <FiCpu className={styles.componentIcon} />
                <h3>Embedding Service</h3>
              </div>
              <div className={`${styles.statusBadge} ${styles.statusHealthy}`}>
                Healthy
              </div>
            </div>

            <div className={styles.componentCard}>
              <div className={styles.componentHeader}>
                <FiMessageSquare className={styles.componentIcon} />
                <h3>LLM Client</h3>
              </div>
              <div className={`${styles.statusBadge} ${styles.statusHealthy}`}>
                Healthy
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
