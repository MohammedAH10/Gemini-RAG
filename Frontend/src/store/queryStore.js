import { create } from 'zustand';
import { queryService } from '../services/query';

export const useQueryStore = create((set) => ({
  queries: [],
  currentQuery: null,
  loading: false,
  error: null,
  stats: null,

  askQuestion: async (query, options = {}) => {
    set({ loading: true, error: null });
    const startTime = Date.now();
    try {
      const data = await queryService.ask(query, options);
      const responseTime = Date.now() - startTime;
      
      const queryRecord = {
        id: Date.now(),
        query,
        response: data,
        responseTime,
        timestamp: new Date().toISOString(),
      };

      set((state) => ({
        queries: [queryRecord, ...state.queries].slice(0, 50), // Keep last 50 queries
        currentQuery: queryRecord,
        loading: false,
      }));

      return data;
    } catch (error) {
      set({ error: error.message, loading: false });
      throw error;
    }
  },

  fetchStats: async () => {
    try {
      const data = await queryService.stats();
      set({ stats: data });
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  },

  clearCurrentQuery: () => set({ currentQuery: null }),
  
  clearQueries: () => set({ queries: [] }),
}));
