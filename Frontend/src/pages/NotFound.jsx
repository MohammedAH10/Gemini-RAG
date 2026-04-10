import { Link } from 'react-router-dom';
import { FiAlertTriangle, FiHome } from 'react-icons/fi';
import styles from '../styles/NotFound.module.css';

export default function NotFound() {
  return (
    <div className={styles.notFound}>
      <FiAlertTriangle className={styles.notFoundIcon} />
      <h1>404</h1>
      <h2>Page Not Found</h2>
      <p>The page you are looking for does not exist.</p>
      <Link to="/" className={styles.homeBtn}>
        <FiHome /> Back to Home
      </Link>
    </div>
  );
}
