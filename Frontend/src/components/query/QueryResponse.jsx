import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { FiFileText, FiChevronDown, FiChevronUp } from 'react-icons/fi';
import { useState } from 'react';

export default function QueryResponse({ response }) {
  const [expandedSources, setExpandedSources] = useState({});

  if (!response) return null;

  const toggleSource = (index) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Answer */}
      <div>
        <h3 className="text-lg font-semibold mb-3 gradient-text">Answer</h3>
        <div className="glass rounded-xl p-6 prose prose-invert max-w-none">
          <ReactMarkdown
            components={{
              p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
              strong: ({ children }) => <strong className="text-neon-blue">{children}</strong>,
              code: ({ children }) => (
                <code className="px-2 py-1 bg-dark-800 rounded text-sm font-mono text-neon-green">
                  {children}
                </code>
              ),
              pre: ({ children }) => (
                <pre className="p-4 bg-dark-800 rounded-lg overflow-x-auto my-3">
                  {children}
                </pre>
              ),
              ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
            }}
          >
            {response.answer || ''}
          </ReactMarkdown>
        </div>
      </div>

      {/* Sources */}
      {response.sources && response.sources.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3 gradient-text">
            Sources ({response.sources.length})
          </h3>
          <div className="space-y-3">
            {response.sources.map((source, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="glass rounded-lg overflow-hidden"
              >
                <button
                  onClick={() => toggleSource(index)}
                  className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 flex items-center justify-center">
                      <FiFileText className="text-neon-blue" />
                    </div>
                    <div className="text-left">
                      <p className="font-medium text-sm">
                        {source.document_title || `Source ${index + 1}`}
                      </p>
                      <p className="text-xs text-white/60">
                        Similarity: {(source.similarity * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  {expandedSources[index] ? <FiChevronUp /> : <FiChevronDown />}
                </button>
                
                {expandedSources[index] && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-white/10 p-4 text-sm text-white/80">
                    
                    <p className="line-clamp-4">{source.content || source.text}</p>
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="flex flex-wrap gap-4 text-sm text-white/60">
        {response.chunks_retrieved && (
          <div className="px-3 py-2 glass rounded-lg">
            <span className="text-neon-blue">{response.chunks_retrieved}</span> chunks retrieved
          </div>
        )}
        {response.response_time && (
          <div className="px-3 py-2 glass rounded-lg">
            <span className="text-neon-green">{response.response_time.toFixed(2)}s</span> response time
          </div>
        )}
        {response.model && (
          <div className="px-3 py-2 glass rounded-lg">
            Model: <span className="text-neon-purple">{response.model}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}