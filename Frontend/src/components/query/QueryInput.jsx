import { useState } from 'react';
import { motion } from 'framer-motion';
import { FiSend, FiSettings } from 'react-icons/fi';

export default function QueryInput({ onResponse, compact = false }) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [nResults, setNResults] = useState(5);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    try {
      await onResponse(query, { nResults });
      setQuery('');
    } catch (error) {
      console.error('Query failed:', error);
    } finally {
      setLoading(false);
    }
  };

  if (compact) {
    return (
      <form onSubmit={handleSubmit} className="relative">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
          className="w-full px-6 py-4 pr-16 bg-white/5 border border-white/20 rounded-xl text-white placeholder-white/40 focus:outline-none focus:border-neon-blue focus:ring-2 focus:ring-neon-blue/20 transition-all duration-300 resize-none"
          rows={3}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="absolute right-4 bottom-4 p-3 bg-gradient-to-r from-neon-blue to-neon-purple rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          {loading ? (
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <FiSend className="text-white" />
          )}
        </button>
      </form>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="relative">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
          className="w-full px-6 py-6 pr-16 bg-white/5 border border-white/20 rounded-xl text-white placeholder-white/40 focus:outline-none focus:border-neon-blue focus:ring-2 focus:ring-neon-blue/20 transition-all duration-300 resize-none text-lg"
          rows={4}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="absolute right-4 bottom-4 p-4 bg-gradient-to-r from-neon-blue to-neon-purple rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          {loading ? (
            <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <FiSend className="text-white text-xl" />
          )}
        </button>
      </div>

      {/* Options */}
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setShowOptions(!showOptions)}
          className="flex items-center gap-2 text-white/60 hover:text-white transition-colors"
        >
          <FiSettings />
          <span className="text-sm">Options</span>
        </button>

        <p className="text-sm text-white/40">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>

      {showOptions && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="glass rounded-lg p-4 space-y-4"
        >
          <div>
            <label className="block text-sm font-medium mb-2 text-white/80">
              Number of Results: {nResults}
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={nResults}
              onChange={(e) => setNResults(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </motion.div>
      )}
    </form>
  );
}
