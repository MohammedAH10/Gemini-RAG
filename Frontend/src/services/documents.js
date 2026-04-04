import api from './api';

export const documentService = {
  upload: async (file, title, tags) => {
    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);
    if (tags) formData.append('tags', JSON.stringify(tags));

    const response = await api.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  list: async (page = 1, pageSize = 20) => {
    const response = await api.get('/documents', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  get: async (documentId) => {
    const response = await api.get(`/documents/${documentId}`);
    return response.data;
  },

  update: async (documentId, data) => {
    const response = await api.patch(`/documents/${documentId}`, data);
    return response.data;
  },

  delete: async (documentId) => {
    const response = await api.delete(`/documents/${documentId}`);
    return response.data;
  },

  getStats: async (documentId) => {
    const response = await api.get(`/documents/${documentId}/stats`);
    return response.data;
  },
};
