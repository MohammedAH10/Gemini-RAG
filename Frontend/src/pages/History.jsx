import { useState, useEffect } from 'react';
import { FiClock, FiMessageSquare, FiFileText, FiBarChart2 } from 'react-icons/fi';
import styles from '../styles/History.module.css';

export default function History() {
  const [activeTab, setActiveTab] = useState('chats');
  const [chats, setChats] = useState([]);
  const [docHistory, setDocHistory] = useState([]);

  return (
    <div className={styles.historyPage}>
      <div className={styles.historyHeader}>
        <h1>History</h1>
        <p>Your activity timeline</p>
      </div>

      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'chats' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('chats')}
        >
          <FiMessageSquare /> Chat History
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'documents' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('documents')}
        >
          <FiFileText /> Document History
        </button>
      </div>

      {activeTab === 'chats' && (
        <div className={styles.historyList}>
          {chats.length === 0 ? (
            <div className={styles.emptyState}>
              <FiClock className={styles.emptyIcon} />
              <h2>No Chat History</h2>
              <p>Your chat conversations will be saved here</p>
            </div>
          ) : (
            chats.map((chat) => (
              <div key={chat.id} className={styles.historyItem}>
                <div className={styles.historyItemContent}>
                  <h4>{chat.query}</h4>
                  <p className={styles.historyMeta}>
                    {new Date(chat.created_at).toLocaleString()} &middot; {chat.chunks_retrieved} chunks
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'documents' && (
        <div className={styles.historyList}>
          {docHistory.length === 0 ? (
            <div className={styles.emptyState}>
              <FiFileText className={styles.emptyIcon} />
              <h2>No Document History</h2>
              <p>Document upload and processing history will appear here</p>
            </div>
          ) : (
            docHistory.map((doc) => (
              <div key={doc.id} className={styles.historyItem}>
                <div className={styles.historyItemContent}>
                  <h4>{doc.title}</h4>
                  <p className={styles.historyMeta}>
                    {doc.status} &middot; {doc.chunk_count} chunks
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
