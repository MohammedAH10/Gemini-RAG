import { create } from 'zustand';
import { documentService } from '../services/documents';

export const useDocumentStore = create((set, get) => ({
  documents: [],
  currentDocument: null,
  loading: false,
  error: null,
  pagination: {
    page: 1,
    pageSize: 20,
    total: 0,
  },

  fetchDocuments: async (page = 1, pageSize = 20) => {
    set({ loading: true, error: null });
    try {
      const data = await documentService.list(page, pageSize);
      set({
        documents: data.documents || [],
        pagination: {
          page: data.page || page,
          pageSize: data.page_size || pageSize,
          total: data.total || 0,
        },
        loading: false,
      });
    } catch (error) {
      set({ error: error.message, loading: false });
    }
  },

  uploadDocument: async (file, title, tags) => {
    set({ loading: true, error: null });
    try {
      const data = await documentService.upload(file, title, tags);
      // Refresh document list
      await get().fetchDocuments(get().pagination.page, get().pagination.pageSize);
      set({ loading: false });
      return data;
    } catch (error) {
      set({ error: error.message, loading: false });
      throw error;
    }
  },

  deleteDocument: async (documentId) => {
    set({ loading: true, error: null });
    try {
      await documentService.delete(documentId);
      // Remove from local state
      set((state) => ({
        documents: state.documents.filter((doc) => doc.id !== documentId),
        loading: false,
      }));
    } catch (error) {
      set({ error: error.message, loading: false });
      throw error;
    }
  },

  fetchDocument: async (documentId) => {
    set({ loading: true, error: null });
    try {
      const data = await documentService.get(documentId);
      set({ currentDocument: data, loading: false });
      return data;
    } catch (error) {
      set({ error: error.message, loading: false });
      throw error;
    }
  },

  updateDocument: async (documentId, data) => {
    set({ loading: true, error: null });
    try {
      const updated = await documentService.update(documentId, data);
      // Update in local state
      set((state) => ({
        documents: state.documents.map((doc) =>
          doc.id === documentId ? { ...doc, ...updated } : doc
        ),
        loading: false,
      }));
      return updated;
    } catch (error) {
      set({ error: error.message, loading: false });
      throw error;
    }
  },

  clearCurrentDocument: () => set({ currentDocument: null }),
}));
