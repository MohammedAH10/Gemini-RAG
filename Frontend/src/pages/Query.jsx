import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiSend, FiClock, FiDatabase, FiZap } from 'react-icons/fi';
import { useQueryStore } from '../store/queryStore';
import QueryInput from '../components/query/QueryInput';
import QueryResponse from '../components/query/QueryResponse';
import toast from 'react-hot-toast';

export default function Query() {
  const { queries, loading, askQuestion } = useQueryStore();
  const [currentResponse, setCurrentResponse] = useState(null);
  const [responseTime, setResponseTime] = useState(null);

  const handleQuery = async (query) => {
    const startTime = Date.now();
    try {
      const response = await askQuestion(query, { nResults: 5 });
      const time = Date.now() - startTime;
      setResponseTime(time);
      setCurrentResponse(response);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Query failed');
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">Ask Questions</h1>
        <p className="text-white/60">Query your documents using AI-powered retrieval</p>
      </motion.div>

      {/* Query Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <QueryInput onResponse={handleQuery} />
      </motion.div>

      {/* Current Response */}
      <AnimatePresence>
        {currentResponse && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="card"
          >
            {responseTime && (
              <div className="flex items-center gap-4 mb-4 text-sm text-white/60">
                <div className="flex items-center gap-2">
                  <FiClock className="text-neon-blue" />
                  <span>{responseTime}ms</span>
                </div>
                <div className="flex items-center gap-2">
                  <FiDatabase className="text-neon-purple" />
                  <span>{currentResponse.chunks_retrieved || 0} chunks retrieved</span>
                </div>
                {currentResponse.response_time && (
                  <div className="flex items-center gap-2">
                    <FiZap className="text-neon-green" />
                    <span>{currentResponse.response_time.toFixed(2)}s generation time</span>
                  </div>
                )}
              </div>
            )}
            <QueryResponse response={currentResponse} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Query History */}
      {queries.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <h2 className="text-2xl font-bold mb-4 gradient-text">Recent Queries</h2>
          <div className="space-y-3 max-h-96 overflow-y-auto scrollbar-custom">
            <AnimatePresence>
              {queries.slice(0, 10).map((query) => (
                <motion.div
                  key={query.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="glass rounded-lg p-4 hover:bg-white/10 transition-all cursor-pointer"
                  onClick={() => {
                    setCurrentResponse(query.response);
                    setResponseTime(query.responseTime);
                  }}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <p className="font-medium mb-1">{query.query}</p>
                      <p className="text-sm text-white/60 line-clamp-2">
                        {query.response.answer?.substring(0, 150)}...
                      </p>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <p className="text-xs text-white/40">{query.responseTime}ms</p>
                      <p className="text-xs text-white/40">
                        {new Date(query.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </motion.div>
      )}
    </div>
  );
}