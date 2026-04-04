import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { FiFileText, FiMessageSquare, FiUpload, FiTrendingUp } from 'react-icons/fi';
import { useDocumentStore } from '../store/documentStore';
import { useQueryStore } from '../store/queryStore';
import QueryInput from '../components/query/QueryInput';
import QueryResponse from '../components/query/QueryResponse';

export default function Dashboard() {
  const navigate = useNavigate();
  const { documents, fetchDocuments } = useDocumentStore();
  const { queries, fetchStats, stats } = useQueryStore();
  const [queryResponse, setQueryResponse] = useState(null);

  useEffect(() => {
    fetchDocuments(1, 5);
    fetchStats();
  }, [fetchDocuments, fetchStats]);

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">Dashboard</h1>
        <p className="text-white/60">Welcome to N.O.V.A.R</p>
      </motion.div>

      {/* Stats Cards */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <motion.div variants={item} className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-blue/5 flex items-center justify-center">
              <FiFileText className="text-2xl text-neon-blue" />
            </div>
            <FiTrendingUp className="text-neon-green text-xl" />
          </div>
          <h3 className="text-3xl font-bold mb-1">{documents.length}</h3>
          <p className="text-white/60 text-sm">Documents</p>
        </motion.div>

        <motion.div variants={item} className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-neon-purple/20 to-neon-purple/5 flex items-center justify-center">
              <FiMessageSquare className="text-2xl text-neon-purple" />
            </div>
            <FiTrendingUp className="text-neon-green text-xl" />
          </div>
          <h3 className="text-3xl font-bold mb-1">{queries.length}</h3>
          <p className="text-white/60 text-sm">Queries</p>
        </motion.div>

        <motion.div variants={item} className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-neon-pink/20 to-neon-pink/5 flex items-center justify-center">
              <FiUpload className="text-2xl text-neon-pink" />
            </div>
            <FiTrendingUp className="text-neon-green text-xl" />
          </div>
          <h3 className="text-3xl font-bold mb-1">{stats?.total_uploads || 0}</h3>
          <p className="text-white/60 text-sm">Uploads</p>
        </motion.div>

        <motion.div variants={item} className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-neon-green/20 to-neon-green/5 flex items-center justify-center">
              <FiTrendingUp className="text-2xl text-neon-green" />
            </div>
            <FiTrendingUp className="text-neon-green text-xl" />
          </div>
          <h3 className="text-3xl font-bold mb-1">{stats?.avg_response_time || '0s'}</h3>
          <p className="text-white/60 text-sm">Avg Response</p>
        </motion.div>
      </motion.div>

      {/* Quick Query Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4 gradient-text">Quick Query</h2>
        <QueryInput onResponse={setQueryResponse} compact />
        {queryResponse && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-6"
          >
            <QueryResponse response={queryResponse} />
          </motion.div>
        )}
      </motion.div>

      {/* Recent Documents */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold gradient-text">Recent Documents</h2>
          <button
            onClick={() => navigate('/documents')}
            className="text-neon-blue hover:underline text-sm"
          >
            View All
          </button>
        </div>
        {documents.length === 0 ? (
          <div className="text-center py-8">
            <FiFileText className="text-4xl text-white/20 mx-auto mb-3" />
            <p className="text-white/60 mb-4">No documents uploaded yet</p>
            <button
              onClick={() => navigate('/documents')}
              className="btn-primary"
            >
              Upload Documents
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {documents.slice(0, 5).map((doc) => (
              <div
                key={doc.id}
                className="p-4 glass rounded-lg hover:bg-white/10 transition-all cursor-pointer"
                onClick={() => navigate('/documents')}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 flex items-center justify-center">
                      <FiFileText className="text-neon-blue" />
                    </div>
                    <div>
                      <h4 className="font-medium">{doc.title || doc.filename}</h4>
                      <p className="text-sm text-white/60">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
}