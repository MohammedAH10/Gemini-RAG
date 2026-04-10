import { useState, useEffect } from 'react';
import { FiDatabase, FiTrash2, FiRefreshCw, FiCheckCircle } from 'react-icons/fi';
import styles from '../styles/VectorStore.module.css';

export default function VectorStore() {
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // TODO: Wire up API endpoints
    console.log('Fetch vector store stats');
    setLoading(false);
  }, []);

  return (
    <div className={styles.vsPage}>
      <div className={styles.vsHeader}>
        <h1>Vector Store Manager</h1>
        <p>Manage your indexed documents and embeddings</p>
      </div>

      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <FiDatabase className={styles.statIcon} />
          <div className={styles.statInfo}>
            <span className={styles.statLabel}>Total Chunks</span>
            <span className={styles.statValue}>{stats?.total_chunks || 0}</span>
          </div>
        </div>
        <div className={styles.statCard}>
          <FiCheckCircle className={styles.statIcon} />
          <div className={styles.statInfo}>
            <span className={styles.statLabel}>Documents</span>
            <span className={styles.statValue}>{documents.length || 0}</span>
          </div>
        </div>
      </div>

      <div className={styles.docsSection}>
        <h2>Indexed Documents</h2>
        {documents.length === 0 ? (
          <div className={styles.emptyState}>
            <FiDatabase className={styles.emptyIcon} />
            <p>No indexed documents yet. Upload documents to populate the vector store.</p>
          </div>
        ) : (
          <div className={styles.docsTable}>
            <table>
              <thead>
                <tr>
                  <th>Document ID</th>
                  <th>Chunks</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.document_id}>
                    <td>{doc.document_id}</td>
                    <td>{doc.chunk_count}</td>
                    <td>
                      <button className={styles.tableActionBtn} aria-label="Delete">
                        <FiTrash2 />
                      </button>
                      <button className={styles.tableActionBtn} aria-label="Re-index">
                        <FiRefreshCw />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
