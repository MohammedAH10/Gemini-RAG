import { useState, useEffect } from 'react';
import { FiSettings, FiSliders, FiCpu, FiLayers } from 'react-icons/fi';
import styles from '../styles/Settings.module.css';

export default function Settings() {
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    // TODO: Wire up API endpoint
    console.log('Save settings:', { chunkSize, chunkOverlap });
  };

  return (
    <div className={styles.settingsPage}>
      <div className={styles.settingsHeader}>
        <h1>Settings</h1>
        <p>Configure your RAG pipeline parameters</p>
      </div>

      <div className={styles.settingsGrid}>
        <div className={styles.settingsCard}>
          <div className={styles.cardHeader}>
            <FiLayers className={styles.cardIcon} />
            <h2>Chunking Configuration</h2>
          </div>

          <div className={styles.settingItem}>
            <label>Chunk Size (tokens)</label>
            <input
              type="range"
              min="100"
              max="2000"
              step="50"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
            />
            <span className={styles.settingValue}>{chunkSize}</span>
          </div>

          <div className={styles.settingItem}>
            <label>Chunk Overlap (tokens)</label>
            <input
              type="range"
              min="0"
              max="500"
              step="10"
              value={chunkOverlap}
              onChange={(e) => setChunkOverlap(Number(e.target.value))}
            />
            <span className={styles.settingValue}>{chunkOverlap}</span>
          </div>
        </div>

        <div className={styles.settingsCard}>
          <div className={styles.cardHeader}>
            <FiCpu className={styles.cardIcon} />
            <h2>Model Configuration</h2>
          </div>

          <div className={styles.settingItem}>
            <label>Temperature</label>
            <input type="range" min="0" max="1" step="0.1" defaultValue="0.7" />
            <span className={styles.settingValue}>0.7</span>
          </div>

          <div className={styles.settingItem}>
            <label>Max Tokens</label>
            <input type="number" defaultValue="1024" min="1" max="4096" />
          </div>
        </div>

        <div className={styles.settingsCard}>
          <div className={styles.cardHeader}>
            <FiSliders className={styles.cardIcon} />
            <h2>Query Configuration</h2>
          </div>

          <div className={styles.settingItem}>
            <label>Top-K Results</label>
            <input type="number" defaultValue="5" min="1" max="20" />
          </div>

          <div className={styles.settingItem}>
            <label>Min Similarity Threshold</label>
            <input type="range" min="0" max="1" step="0.05" defaultValue="0.5" />
            <span className={styles.settingValue}>0.5</span>
          </div>
        </div>
      </div>

      <div className={styles.actions}>
        <button className={styles.saveBtn} onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  );
}
