import { useMutation } from '@tanstack/react-query';
import { generateImage } from '../services/api';
import toast from 'react-hot-toast';

/**
 * Custom hook for image generation with optimized state management
 */
export const useImageGeneration = () => {
  const mutation = useMutation({
    mutationFn: ({ data, onProgress }) => generateImage(data, onProgress),
    onSuccess: (data) => {
      console.log('[Generation] Success:', data);
      toast.success('Image generated successfully!');
    },
    onError: (error) => {
      console.error('[Generation] Error:', error);
      const errorMessage = error.response?.data?.error || error.message || 'Failed to generate image';
      toast.error(errorMessage);
    },
  });

  return {
    generate: mutation.mutate,
    isGenerating: mutation.isPending,
    data: mutation.data,
    error: mutation.error,
    reset: mutation.reset,
  };
};

export default useImageGeneration;