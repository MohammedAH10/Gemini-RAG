import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  FiFileText, FiCalendar, FiHash, FiDatabase,
  FiArrowLeft, FiTrash2, FiEdit2,
} from 'react-icons/fi';
import styles from '../styles/DocumentViewer.module.css';

export default function DocumentViewer() {
  const { id } = useParams();
  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // TODO: Wire up API endpoint
    console.log('Fetch document:', id);
    setLoading(false);
  }, [id]);

  if (loading) {
    return <div className={styles.loading}>Loading document...</div>;
  }

  return (
    <div className={styles.viewerPage}>
      <div className={styles.viewerHeader}>
        <Link to="/documents" className={styles.backBtn}>
          <FiArrowLeft /> Back to Documents
        </Link>
        <div className={styles.titleActions}>
          <h1>{document?.title || 'Document Details'}</h1>
          <div className={styles.actionBtns}>
            <button className={styles.actionBtn}><FiEdit2 /> Edit</button>
            <button className={`${styles.actionBtn} ${styles.deleteBtn}`}><FiTrash2 /> Delete</button>
          </div>
        </div>
      </div>

      <div className={styles.viewerGrid}>
        <div className={styles.infoCard}>
          <h3>Document Info</h3>
          <div className={styles.infoRow}>
            <FiFileText />
            <span>Type</span>
            <strong>{document?.file_type?.toUpperCase() || 'PDF'}</strong>
          </div>
          <div className={styles.infoRow}>
            <FiDatabase />
            <span>Size</span>
            <strong>{document?.file_size ? `${(document.file_size / 1024).toFixed(2)} KB` : 'N/A'}</strong>
          </div>
          <div className={styles.infoRow}>
            <FiHash />
            <span>Chunks</span>
            <strong>{document?.chunk_count || 0}</strong>
          </div>
          <div className={styles.infoRow}>
            <FiCalendar />
            <span>Created</span>
            <strong>{document?.created_at ? new Date(document.created_at).toLocaleDateString() : 'N/A'}</strong>
          </div>
        </div>

        <div className={styles.previewCard}>
          <h3>Chunks Preview</h3>
          <div className={styles.chunksPlaceholder}>
            <p>Document chunks will be displayed here</p>
          </div>
        </div>
      </div>
    </div>
  );
}
