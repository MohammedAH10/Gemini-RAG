import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiUploadCloud, FiFileText, FiCheck, FiX } from 'react-icons/fi';
import styles from '../styles/DocumentUpload.module.css';

export default function DocumentUpload() {
  const navigate = useNavigate();
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles((prev) => [...prev, ...droppedFiles]);
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles((prev) => [...prev, ...selectedFiles]);
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    // TODO: Wire up API endpoint
    console.log('Uploading:', files);
  };

  return (
    <div className={styles.uploadPage}>
      <div className={styles.uploadHeader}>
        <h1>Upload Documents</h1>
        <p>Supported formats: PDF, EPUB, TXT, DOCX, MOBI, AZW, AZW3</p>
      </div>

      <div
        className={`${styles.dropZone} ${dragActive ? styles.dragActive : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <FiUploadCloud className={styles.dropIcon} />
        <h3>Drop files here or click to browse</h3>
        <p>Maximum file size: 50MB per file</p>
        <input
          type="file"
          multiple
          accept=".pdf,.epub,.txt,.docx,.mobi,.azw,.azw3"
          onChange={handleFileSelect}
          className={styles.fileInput}
        />
      </div>

      {files.length > 0 && (
        <div className={styles.fileList}>
          <h2>Selected Files ({files.length})</h2>
          {files.map((file, idx) => (
            <div key={idx} className={styles.fileItem}>
              <div className={styles.fileInfo}>
                <FiFileText className={styles.fileIcon} />
                <div>
                  <span className={styles.fileName}>{file.name}</span>
                  <span className={styles.fileSize}>
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              </div>
              <button className={styles.removeBtn} onClick={() => removeFile(idx)}>
                <FiX />
              </button>
            </div>
          ))}

          <button
            className={styles.uploadBtn}
            onClick={handleUpload}
            disabled={uploading}
          >
            {uploading ? (
              <>Processing...</>
            ) : (
              <>
                <FiCheck /> Upload & Index
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
