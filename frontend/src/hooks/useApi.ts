/**
 * API Hooks
 * Owner: Frontend Team Lead
 */

import { useState, useEffect, useCallback } from 'react'
import { AxiosError } from 'axios'

// Generic API state interface
interface ApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

// Hook for basic API calls
export const useApi = <T>(
  apiCall: () => Promise<T>,
  deps: React.DependencyList = []
) => {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: true,
    error: null,
  })

  const execute = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    try {
      const result = await apiCall()
      setState({ data: result, loading: false, error: null })
      return result
    } catch (error) {
      const errorMessage = error instanceof AxiosError 
        ? error.response?.data?.detail || error.message 
        : 'An unexpected error occurred'
      
      setState({ data: null, loading: false, error: errorMessage })
      throw error
    }
  }, deps)

  useEffect(() => {
    execute()
  }, [execute])

  return {
    ...state,
    refetch: execute,
  }
}

// Hook for mutations (POST, PUT, DELETE)
export const useMutation = <T, P = void>(
  mutationFn: (params: P) => Promise<T>
) => {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
  })

  const mutate = useCallback(async (params: P) => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    try {
      const result = await mutationFn(params)
      setState({ data: result, loading: false, error: null })
      return result
    } catch (error) {
      const errorMessage = error instanceof AxiosError 
        ? error.response?.data?.detail || error.message 
        : 'An unexpected error occurred'
      
      setState({ data: null, loading: false, error: errorMessage })
      throw error
    }
  }, [mutationFn])

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null })
  }, [])

  return {
    ...state,
    mutate,
    reset,
  }
}

// Hook for paginated data
export const usePagination = <T>(
  apiCall: (page: number, limit: number, params?: any) => Promise<{
    data: T[]
    total: number
    page: number
    limit: number
    total_pages: number
  }>,
  initialPage = 1,
  initialLimit = 10,
  params?: any
) => {
  const [page, setPage] = useState(initialPage)
  const [limit, setLimit] = useState(initialLimit)
  const [state, setState] = useState<{
    data: T[]
    total: number
    totalPages: number
    loading: boolean
    error: string | null
  }>({
    data: [],
    total: 0,
    totalPages: 0,
    loading: true,
    error: null,
  })

  const fetchData = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    try {
      const result = await apiCall(page, limit, params)
      setState({
        data: result.data,
        total: result.total,
        totalPages: result.total_pages,
        loading: false,
        error: null,
      })
    } catch (error) {
      const errorMessage = error instanceof AxiosError 
        ? error.response?.data?.detail || error.message 
        : 'An unexpected error occurred'
      
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
      }))
    }
  }, [apiCall, page, limit, params])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const goToPage = useCallback((newPage: number) => {
    setPage(newPage)
  }, [])

  const changeLimit = useCallback((newLimit: number) => {
    setLimit(newLimit)
    setPage(1) // Reset to first page when changing limit
  }, [])

  return {
    ...state,
    page,
    limit,
    goToPage,
    changeLimit,
    refetch: fetchData,
  }
}

// Hook for real-time data with polling
export const usePolling = <T>(
  apiCall: () => Promise<T>,
  interval = 5000,
  enabled = true
) => {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: true,
    error: null,
  })

  const fetchData = useCallback(async () => {
    try {
      const result = await apiCall()
      setState(prev => ({ 
        data: result, 
        loading: prev.data === null, // Only show loading on first fetch
        error: null 
      }))
    } catch (error) {
      const errorMessage = error instanceof AxiosError 
        ? error.response?.data?.detail || error.message 
        : 'An unexpected error occurred'
      
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: errorMessage 
      }))
    }
  }, [apiCall])

  useEffect(() => {
    if (!enabled) return

    fetchData()
    const intervalId = setInterval(fetchData, interval)

    return () => clearInterval(intervalId)
  }, [fetchData, interval, enabled])

  return {
    ...state,
    refetch: fetchData,
  }
}

// Hook for optimistic updates
export const useOptimisticUpdate = <T>(
  data: T[],
  updateFn: (item: T) => Promise<T>,
  keyField: keyof T = 'id' as keyof T
) => {
  const [optimisticData, setOptimisticData] = useState<T[]>(data)
  const [loading, setLoading] = useState<Record<string, boolean>>({})

  useEffect(() => {
    setOptimisticData(data)
  }, [data])

  const updateItem = useCallback(async (item: T, updates: Partial<T>) => {
    const itemId = String(item[keyField])
    
    // Optimistic update
    setOptimisticData(prev => 
      prev.map(i => 
        i[keyField] === item[keyField] 
          ? { ...i, ...updates } 
          : i
      )
    )
    
    setLoading(prev => ({ ...prev, [itemId]: true }))

    try {
      const result = await updateFn({ ...item, ...updates })
      
      // Update with server response
      setOptimisticData(prev => 
        prev.map(i => 
          i[keyField] === item[keyField] 
            ? result 
            : i
        )
      )
    } catch (error) {
      // Revert optimistic update on error
      setOptimisticData(prev => 
        prev.map(i => 
          i[keyField] === item[keyField] 
            ? item 
            : i
        )
      )
      throw error
    } finally {
      setLoading(prev => ({ ...prev, [itemId]: false }))
    }
  }, [updateFn, keyField])

  return {
    data: optimisticData,
    loading,
    updateItem,
  }
}

// Hook for debounced search
export const useDebounceSearch = <T>(
  searchFn: (query: string) => Promise<T[]>,
  delay = 300
) => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!query.trim()) {
      setResults([])
      return
    }

    const timer = setTimeout(async () => {
      setLoading(true)
      setError(null)
      
      try {
        const searchResults = await searchFn(query)
        setResults(searchResults)
      } catch (error) {
        const errorMessage = error instanceof AxiosError 
          ? error.response?.data?.detail || error.message 
          : 'Search failed'
        
        setError(errorMessage)
        setResults([])
      } finally {
        setLoading(false)
      }
    }, delay)

    return () => clearTimeout(timer)
  }, [query, searchFn, delay])

  return {
    query,
    setQuery,
    results,
    loading,
    error,
  }
}

export default useApi