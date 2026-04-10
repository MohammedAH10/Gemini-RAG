import { Link } from 'react-router-dom';
import { FiArrowRight, FiMessageSquare, FiFileText, FiSearch, FiShield, FiZap } from 'react-icons/fi';
import styles from '../styles/Landing.module.css';

export default function Landing() {
  return (
    <div className={styles.landing}>
      {/* Hero Section */}
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <div className={styles.badge}>Powered by Gemini AI</div>
          <h1 className={styles.title}>
            Intelligent Document
            <span className={styles.gradientText}> Query Engine</span>
          </h1>
          <p className={styles.subtitle}>
            Upload your documents and get instant, AI-powered answers with precise citations.
            Experience the future of information retrieval.
          </p>
          <div className={styles.heroActions}>
            <Link to="/auth/signup" className={styles.primaryBtn}>
              Get Started <FiArrowRight />
            </Link>
            <Link to="/auth/signin" className={styles.secondaryBtn}>
              Sign In
            </Link>
          </div>
        </div>
        <div className={styles.heroVisual}>
          <div className={styles.orb} />
          <div className={styles.orbSecondary} />
        </div>
      </section>

      {/* Features Section */}
      <section className={styles.features}>
        <h2 className={styles.sectionTitle}>Core Capabilities</h2>
        <div className={styles.featuresGrid}>
          <div className={styles.featureCard}>
            <FiMessageSquare className={styles.featureIcon} />
            <h3>RAG-Powered Chat</h3>
            <p>Ask questions and get intelligent answers sourced directly from your documents.</p>
          </div>
          <div className={styles.featureCard}>
            <FiFileText className={styles.featureIcon} />
            <h3>Document Processing</h3>
            <p>Upload PDFs, EPUBs, DOCX and more. Automatic text extraction and indexing.</p>
          </div>
          <div className={styles.featureCard}>
            <FiSearch className={styles.featureIcon} />
            <h3>Smart Search</h3>
            <p>Semantic search across your entire document collection with similarity matching.</p>
          </div>
          <div className={styles.featureCard}>
            <FiZap className={styles.featureIcon} />
            <h3>Instant Indexing</h3>
            <p>Documents are chunked, embedded and indexed in real-time for immediate querying.</p>
          </div>
          <div className={styles.featureCard}>
            <FiShield className={styles.featureIcon} />
            <h3>Secure & Private</h3>
            <p>User-level isolation ensures your documents are accessible only to you.</p>
          </div>
          <div className={styles.featureCard}>
            <FiDatabase className={styles.featureIcon} />
            <h3>Vector Store</h3>
            <p>Advanced vector database for high-dimensional semantic similarity search.</p>
          </div>
        </div>
      </section>

      {/* Supported Formats */}
      <section className={styles.formats}>
        <h2 className={styles.sectionTitle}>Supported Formats</h2>
        <div className={styles.formatsList}>
          {['PDF', 'EPUB', 'TXT', 'DOCX', 'MOBI', 'AZW', 'AZW3'].map((format) => (
            <span key={format} className={styles.formatBadge}>{format}</span>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <p>Gemini RAG &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}
