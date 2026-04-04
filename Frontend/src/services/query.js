import api from './api';

export const queryService = {
  ask: async (query, options = {}) => {
    const response = await api.post('/query/ask', {
      query,
      n_results: options.nResults || 5,
      document_ids: options.documentIds,
      temperature: options.temperature || 0.7,
      include_citations: options.includeCitations !== false,
      min_similarity: options.minSimilarity,
    });
    return response.data;
  },

  batch: async (queries) => {
    const response = await api.post('/query/batch', {
      queries,
      n_results: 5,
    });
    return response.data;
  },

  stats: async () => {
    const response = await api.get('/query/stats');
    return response.data;
  },

  health: async () => {
    const response = await api.get('/query/health');
    return response.data;
  },
};
