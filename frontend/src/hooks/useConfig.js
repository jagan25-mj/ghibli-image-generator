import { useQuery } from '@tanstack/react-query';
import { getConfig } from '../services/api';

/**
 * Custom hook to fetch and cache configuration
 */
export const useConfig = () => {
  return useQuery({
    queryKey: ['config'],
    queryFn: getConfig,
    staleTime: Infinity, // Config rarely changes
    cacheTime: Infinity,
  });
};

export default useConfig;