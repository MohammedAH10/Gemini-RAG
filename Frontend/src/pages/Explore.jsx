import { useState } from 'react';
import { FiSearch, FiFilter } from 'react-icons/fi';
import styles from '../styles/Explore.module.css';

export default function Explore() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    // TODO: Wire up API endpoint
    console.log('Explore search:', query);
  };

  return (
    <div className={styles.explorePage}>
      <div className={styles.exploreHeader}>
        <h1>Explore Documents</h1>
        <p>Semantic search across your entire document collection</p>
      </div>

      <div className={styles.searchArea}>
        <div className={styles.searchBox}>
          <FiSearch className={styles.searchIcon} />
          <input
            type="text"
            placeholder="Search your documents..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button className={styles.searchBtn} onClick={handleSearch}>
            Search
          </button>
        </div>

        <div className={styles.filters}>
          <button className={styles.filterBtn}>
            <FiFilter /> All Documents
          </button>
        </div>
      </div>

      {results.length === 0 && !searching ? (
        <div className={styles.emptyState}>
          <FiSearch className={styles.emptyIcon} />
          <h2>Search Your Knowledge Base</h2>
          <p>Enter a query to find relevant information across all your documents</p>
        </div>
      ) : (
        <div className={styles.results}>
          {results.map((result, idx) => (
            <div key={idx} className={styles.resultCard}>
              <h3>{result.title}</h3>
              <p className={styles.resultSnippet}>{result.snippet}</p>
              <div className={styles.resultMeta}>
                <span>Similarity: {(result.similarity * 100).toFixed(1)}%</span>
                <span>Document: {result.document_id}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
