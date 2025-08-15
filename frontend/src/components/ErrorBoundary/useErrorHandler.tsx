import { useCallback, useState } from 'react';
import { toast } from 'react-hot-toast';

interface ErrorHandlerOptions {
  showToast?: boolean;
  fallbackMessage?: string;
  retryCount?: number;
  retryDelay?: number;
  onError?: (error: Error) => void;
  onRetry?: () => void;
}

export const useErrorHandler = (options: ErrorHandlerOptions = {}) => {
  const {
    showToast = true,
    fallbackMessage = 'An error occurred',
    retryCount = 3,
    retryDelay = 1000,
    onError,
    onRetry,
  } = options;

  const [error, setError] = useState<Error | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);
  const [attemptCount, setAttemptCount] = useState(0);

  const handleError = useCallback(
    (error: Error | unknown) => {
      const errorObj = error instanceof Error ? error : new Error(String(error));
      
      setError(errorObj);
      
      // Log error
      console.error('Error handled:', errorObj);
      
      // Show toast notification
      if (showToast) {
        toast.error(errorObj.message || fallbackMessage);
      }
      
      // Call custom error handler
      if (onError) {
        onError(errorObj);
      }
      
      // Log to error service in production
      if (process.env.NODE_ENV === 'production') {
        // sendToErrorService(errorObj);
      }
    },
    [showToast, fallbackMessage, onError]
  );

  const retry = useCallback(
    async (fn: () => Promise<any>) => {
      if (attemptCount >= retryCount) {
        toast.error('Maximum retry attempts reached');
        return;
      }

      setIsRetrying(true);
      setAttemptCount((prev) => prev + 1);

      try {
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
        const result = await fn();
        
        // Reset on success
        setError(null);
        setAttemptCount(0);
        setIsRetrying(false);
        
        if (onRetry) {
          onRetry();
        }
        
        return result;
      } catch (_err) {
        setIsRetrying(false);
        handleError(err);
        throw err;
      }
    },
    [attemptCount, retryCount, retryDelay, handleError, onRetry]
  );

  const reset = useCallback(() => {
    setError(null);
    setAttemptCount(0);
    setIsRetrying(false);
  }, []);

  const throwError = useCallback((error: Error | string) => {
    const errorObj = error instanceof Error ? error : new Error(error);
    handleError(errorObj);
    throw errorObj;
  }, [handleError]);

  return {
    error,
    isRetrying,
    attemptCount,
    handleError,
    retry,
    reset,
    throwError,
  };
};

// Async error handler wrapper
export const withErrorHandling = async <T,>(
  fn: () => Promise<T>,
  options?: ErrorHandlerOptions
): Promise<T | null> => {
  try {
    return await fn();
  } catch (_error) {
    const errorMessage = error instanceof Error ? error.message : 'An error occurred';
    
    if (options?.showToast !== false) {
      toast.error(options?.fallbackMessage || errorMessage);
    }
    
    if (options?.onError) {
      options.onError(error instanceof Error ? error : new Error(String(error)));
    }
    
    console.error('Error in async operation:', error);
    return null;
  }
};