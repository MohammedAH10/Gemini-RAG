import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiUpload, FiFileText, FiTrash2, FiSearch, FiX } from 'react-icons/fi';
import { useDocumentStore } from '../store/documentStore';
import toast from 'react-hot-toast';

export default function Documents() {
  const { documents, fetchDocuments, uploadDocument, deleteDocument, loading, pagination } = useDocumentStore();
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchDocuments(pagination.page, pagination.pageSize);
  }, [fetchDocuments, pagination.page, pagination.pageSize]);

  const handleFileUpload = async (file) => {
    if (!file) return;
    
    setUploading(true);
    try {
      await uploadDocument(file, file.name.split('.')[0], []);
      toast.success('Document uploaded successfully');
    } catch (error) {
      toast.error(error.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  const handleDelete = async (docId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;
    
    try {
      await deleteDocument(docId);
      toast.success('Document deleted');
    } catch (error) {
      toast.error('Delete failed');
    }
  };

  const filteredDocuments = documents.filter(doc => 
    (doc.title || doc.filename || '').toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">Documents</h1>
        <p className="text-white/60">Manage your knowledge base</p>
      </motion.div>

      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4 gradient-text">Upload Document</h2>
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
            dragActive
              ? 'border-neon-blue bg-neon-blue/10'
              : 'border-white/20 hover:border-white/40'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.docx,.epub"
            onChange={(e) => handleFileUpload(e.target.files[0])}
            className="hidden"
          />
          <FiUpload className={`text-5xl mx-auto mb-4 ${dragActive ? 'text-neon-blue' : 'text-white/40'}`} />
          <p className="text-lg mb-2">
            {dragActive ? 'Drop your file here' : 'Drag & drop your document here'}
          </p>
          <p className="text-white/60 mb-4">Supports PDF, TXT, DOCX, EPUB</p>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="btn-primary disabled:opacity-50"
          >
            {uploading ? (
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Uploading...</span>
              </div>
            ) : (
              'Browse Files'
            )}
          </button>
        </div>
      </motion.div>

      {/* Documents List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card"
      >
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
          <h2 className="text-2xl font-bold gradient-text">
            Your Documents ({filteredDocuments.length})
          </h2>
          <div className="relative w-full sm:w-64">
            <FiSearch className="absolute left-4 top-1/2 transform -translate-y-1/2 text-white/40" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search documents..."
              className="input-field pl-12 py-2"
            />
          </div>
        </div>

        {filteredDocuments.length === 0 ? (
          <div className="text-center py-12">
            <FiFileText className="text-5xl text-white/20 mx-auto mb-4" />
            <p className="text-white/60 mb-2">No documents found</p>
            <p className="text-sm text-white/40">Upload your first document to get started</p>
          </div>
        ) : (
          <motion.div
            variants={{
              hidden: { opacity: 0 },
              show: {
                opacity: 1,
                transition: { staggerChildren: 0.05 }
              }
            }}
            initial="hidden"
            animate="show"
            className="grid gap-4"
          >
            <AnimatePresence>
              {filteredDocuments.map((doc) => (
                <motion.div
                  key={doc.id}
                  variants={{
                    hidden: { opacity: 0, x: -20 },
                    show: { opacity: 1, x: 0 }
                  }}
                  exit={{ opacity: 0, x: 20 }}
                  className="glass rounded-lg p-4 hover:bg-white/10 transition-all group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 flex items-center justify-center flex-shrink-0">
                        <FiFileText className="text-neon-blue text-xl" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold truncate">{doc.title || doc.filename}</h3>
                        <div className="flex items-center gap-4 text-sm text-white/60">
                          <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                          {doc.file_size && (
                            <span>{(doc.file_size / 1024).toFixed(2)} KB</span>
                          )}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleDelete(doc.id)}
                      className="p-2 rounded-lg hover:bg-red-500/10 text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                    >
                      <FiTrash2 />
                    </button>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Pagination */}
        {pagination.total > pagination.pageSize && (
          <div className="flex items-center justify-center gap-2 mt-6">
            <button
              onClick={() => fetchDocuments(pagination.page - 1, pagination.pageSize)}
              disabled={pagination.page === 1}
              className="px-4 py-2 glass rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/10 transition-all"
            >
              Previous
            </button>
            <span className="px-4 py-2">
              Page {pagination.page} of {Math.ceil(pagination.total / pagination.pageSize)}
            </span>
            <button
              onClick={() => fetchDocuments(pagination.page + 1, pagination.pageSize)}
              disabled={pagination.page >= Math.ceil(pagination.total / pagination.pageSize)}
              className="px-4 py-2 glass rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/10 transition-all"
            >
              Next
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
}