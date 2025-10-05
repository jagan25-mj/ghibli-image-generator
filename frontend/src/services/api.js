import axios from 'axios';

// Create axios instance with default config
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 120000, // 2 minutes for image generation
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Response from ${response.config.url}:`, response.status);
    return response;
  },
  (error) => {
    console.error('[API] Response error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Fetch available configuration options
 */
export const getConfig = async () => {
  const response = await api.get('/api/config/');
  return response.data;
};

/**
 * Generate image with provided parameters
 * @param {Object} data - Generation parameters
 * @param {Function} onProgress - Progress callback (optional)
 */
export const generateImage = async (data, onProgress) => {
  const formData = new FormData();
  
  // Append all fields to FormData
  Object.entries(data).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      if (key === 'init_image' && value instanceof File) {
        formData.append(key, value);
      } else {
        formData.append(key, value);
      }
    }
  });

  const response = await api.post('/api/generate/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress({ phase: 'upload', progress: percentCompleted });
      }
    },
  });

  return response.data;
};

export default api;