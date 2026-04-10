import { useState } from 'react';
import { Link } from 'react-router-dom';
import { FiFileText, FiPlus, FiTrash2, FiEdit2, FiSearch, FiMoreVertical } from 'react-icons/fi';
import styles from '../styles/Documents.module.css';

export default function Documents() {
  const [documents, setDocuments] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);

  return (
    <div className={styles.docsPage}>
      <div className={styles.docsHeader}>
        <h1>Documents</h1>
        <Link to="/documents/upload" className={styles.uploadBtn}>
          <FiPlus /> Upload Document
        </Link>
      </div>

      <div className={styles.docsToolbar}>
        <div className={styles.searchBox}>
          <FiSearch />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {documents.length === 0 ? (
        <div className={styles.emptyState}>
          <FiFileText className={styles.emptyIcon} />
          <h2>No Documents Yet</h2>
          <p>Upload your first document to get started</p>
          <Link to="/documents/upload" className={styles.uploadLink}>
            <FiPlus /> Upload Document
          </Link>
        </div>
      ) : (
        <div className={styles.docsGrid}>
          {documents.map((doc) => (
            <div key={doc.id} className={styles.docCard}>
              <Link to={`/documents/${doc.id}`} className={styles.docLink}>
                <div className={styles.docIcon}>
                  <FiFileText />
                </div>
                <h3>{doc.title}</h3>
                <p className={styles.docMeta}>
                  {doc.chunk_count} chunks &middot; {doc.file_type?.toUpperCase()}
                </p>
              </Link>
              <div className={styles.docActions}>
                <button className={styles.actionBtn} aria-label="Edit"><FiEdit2 /></button>
                <button className={styles.actionBtn} aria-label="Delete"><FiTrash2 /></button>
                <button className={styles.actionBtn} aria-label="More"><FiMoreVertical /></button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
