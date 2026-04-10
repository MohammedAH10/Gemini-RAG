import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  FiFileText, FiMessageSquare, FiDatabase, FiUploadCloud,
  FiArrowRight, FiTrendingUp, FiClock,
} from 'react-icons/fi';
import styles from '../styles/Dashboard.module.css';

export default function Dashboard() {
  const [stats, setStats] = useState({
    totalDocuments: 0,
    totalQueries: 0,
    totalChunks: 0,
    storageUsed: '0 MB',
  });
  const [recentDocs, setRecentDocs] = useState([]);
  const [recentQueries, setRecentQueries] = useState([]);

  return (
    <div className={styles.dashboard}>
      <div className={styles.header}>
        <h1>Dashboard</h1>
        <p>Welcome back! Here&apos;s an overview of your RAG system.</p>
      </div>

      {/* Stats Cards */}
      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>
            <FiFileText />
          </div>
          <div className={styles.statInfo}>
            <span className={styles.statValue}>{stats.totalDocuments}</span>
            <span className={styles.statLabel}>Documents</span>
          </div>
        </div>

        <div className={styles.statCard}>
          <div className={styles.statIcon}>
            <FiMessageSquare />
          </div>
          <div className={styles.statInfo}>
            <span className={styles.statValue}>{stats.totalQueries}</span>
            <span className={styles.statLabel}>Queries</span>
          </div>
        </div>

        <div className={styles.statCard}>
          <div className={styles.statIcon}>
            <FiDatabase />
          </div>
          <div className={styles.statInfo}>
            <span className={styles.statValue}>{stats.totalChunks}</span>
            <span className={styles.statLabel}>Chunks Indexed</span>
          </div>
        </div>

        <div className={styles.statCard}>
          <div className={styles.statIcon}>
            <FiTrendingUp />
          </div>
          <div className={styles.statInfo}>
            <span className={styles.statValue}>{stats.storageUsed}</span>
            <span className={styles.statLabel}>Storage Used</span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className={styles.quickActions}>
        <h2>Quick Actions</h2>
        <div className={styles.actionsGrid}>
          <Link to="/chat" className={styles.actionCard}>
            <FiMessageSquare />
            <h3>Ask a Question</h3>
            <p>Query your documents with AI</p>
            <FiArrowRight className={styles.actionArrow} />
          </Link>

          <Link to="/documents/upload" className={styles.actionCard}>
            <FiUploadCloud />
            <h3>Upload Document</h3>
            <p>Add new documents to index</p>
            <FiArrowRight className={styles.actionArrow} />
          </Link>

          <Link to="/explore" className={styles.actionCard}>
            <FiDatabase />
            <h3>Explore</h3>
            <p>Search your knowledge base</p>
            <FiArrowRight className={styles.actionArrow} />
          </Link>
        </div>
      </div>

      {/* Recent Activity */}
      <div className={styles.recentActivity}>
        <div className={styles.activityColumn}>
          <h2>
            <FiClock /> Recent Documents
          </h2>
          {recentDocs.length === 0 ? (
            <div className={styles.emptyActivity}>
              <p>No documents yet. Upload your first one!</p>
              <Link to="/documents/upload" className={styles.link}>
                Upload Document
              </Link>
            </div>
          ) : (
            recentDocs.map((doc) => (
              <div key={doc.id} className={styles.activityItem}>
                <h4>{doc.title}</h4>
                <p className={styles.activityMeta}>
                  {new Date(doc.created_at).toLocaleDateString()}
                </p>
              </div>
            ))
          )}
        </div>

        <div className={styles.activityColumn}>
          <h2>
            <FiMessageSquare /> Recent Queries
          </h2>
          {recentQueries.length === 0 ? (
            <div className={styles.emptyActivity}>
              <p>No queries yet. Start asking questions!</p>
              <Link to="/chat" className={styles.link}>
                Go to Chat
              </Link>
            </div>
          ) : (
            recentQueries.map((q) => (
              <div key={q.id} className={styles.activityItem}>
                <h4>{q.query}</h4>
                <p className={styles.activityMeta}>
                  {new Date(q.created_at).toLocaleString()}
                </p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
