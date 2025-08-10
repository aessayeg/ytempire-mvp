/**
 * Cache Service - Client-side caching and storage management
 * Owner: Frontend Team Lead
 */

export interface CacheOptions {
  ttl?: number // Time to live in milliseconds
  strategy?: 'memory' | 'localStorage' | 'sessionStorage' | 'indexedDB'
  maxSize?: number // Maximum cache size
  compress?: boolean
  serialize?: boolean
}

export interface CacheItem<T = any> {
  data: T
  timestamp: number
  ttl: number
  hits: number
  size: number
}

export interface CacheStats {
  totalItems: number
  totalSize: number
  hitRate: number
  missRate: number
  oldestItem?: string
  newestItem?: string
}

class CacheService {
  private memoryCache: Map<string, CacheItem> = new Map()
  private cacheStats = {
    hits: 0,
    misses: 0,
    sets: 0,
    deletes: 0
  }

  private defaultOptions: Required<CacheOptions> = {
    ttl: 5 * 60 * 1000, // 5 minutes
    strategy: 'memory',
    maxSize: 100, // 100 items max
    compress: false,
    serialize: true
  }

  /**
   * Set cache item
   */
  set<T>(key: string, data: T, options: CacheOptions = {}): void {
    const mergedOptions = { ...this.defaultOptions, ...options }
    
    const item: CacheItem<T> = {
      data,
      timestamp: Date.now(),
      ttl: mergedOptions.ttl,
      hits: 0,
      size: this.calculateSize(data)
    }

    this.cacheStats.sets++

    switch (mergedOptions.strategy) {
      case 'memory':
        this.setMemory(key, item, mergedOptions)
        break
      case 'localStorage':
        this.setLocalStorage(key, item, mergedOptions)
        break
      case 'sessionStorage':
        this.setSessionStorage(key, item, mergedOptions)
        break
      case 'indexedDB':
        this.setIndexedDB(key, item, mergedOptions)
        break
    }
  }

  /**
   * Get cache item
   */
  async get<T>(key: string, options: CacheOptions = {}): Promise<T | null> {
    const mergedOptions = { ...this.defaultOptions, ...options }
    
    let item: CacheItem<T> | null = null

    switch (mergedOptions.strategy) {
      case 'memory':
        item = this.getMemory<T>(key)
        break
      case 'localStorage':
        item = this.getLocalStorage<T>(key)
        break
      case 'sessionStorage':
        item = this.getSessionStorage<T>(key)
        break
      case 'indexedDB':
        item = await this.getIndexedDB<T>(key)
        break
    }

    if (!item) {
      this.cacheStats.misses++
      return null
    }

    // Check if expired
    if (this.isExpired(item)) {
      this.delete(key, options)
      this.cacheStats.misses++
      return null
    }

    // Update hit count
    item.hits++
    this.cacheStats.hits++

    return item.data
  }

  /**
   * Delete cache item
   */
  delete(key: string, options: CacheOptions = {}): void {
    const mergedOptions = { ...this.defaultOptions, ...options }
    
    this.cacheStats.deletes++

    switch (mergedOptions.strategy) {
      case 'memory':
        this.memoryCache.delete(key)
        break
      case 'localStorage':
        localStorage.removeItem(this.getStorageKey(key))
        break
      case 'sessionStorage':
        sessionStorage.removeItem(this.getStorageKey(key))
        break
      case 'indexedDB':
        this.deleteIndexedDB(key)
        break
    }
  }

  /**
   * Check if key exists in cache
   */
  async has(key: string, options: CacheOptions = {}): Promise<boolean> {
    const item = await this.get(key, options)
    return item !== null
  }

  /**
   * Clear all cache items
   */
  clear(options: CacheOptions = {}): void {
    const strategy = options.strategy || this.defaultOptions.strategy

    switch (strategy) {
      case 'memory':
        this.memoryCache.clear()
        break
      case 'localStorage':
        this.clearPrefixedStorage(localStorage)
        break
      case 'sessionStorage':
        this.clearPrefixedStorage(sessionStorage)
        break
      case 'indexedDB':
        this.clearIndexedDB()
        break
    }
  }

  /**
   * Get cache statistics
   */
  getStats(strategy?: CacheOptions['strategy']): CacheStats {
    let totalItems = 0
    let totalSize = 0
    let oldestTimestamp = Infinity
    let newestTimestamp = 0
    let oldestKey = ''
    let newestKey = ''

    if (!strategy || strategy === 'memory') {
      this.memoryCache.forEach((item, key) => {
        totalItems++
        totalSize += item.size
        
        if (item.timestamp < oldestTimestamp) {
          oldestTimestamp = item.timestamp
          oldestKey = key
        }
        
        if (item.timestamp > newestTimestamp) {
          newestTimestamp = item.timestamp
          newestKey = key
        }
      })
    }

    const totalRequests = this.cacheStats.hits + this.cacheStats.misses
    
    return {
      totalItems,
      totalSize,
      hitRate: totalRequests > 0 ? this.cacheStats.hits / totalRequests : 0,
      missRate: totalRequests > 0 ? this.cacheStats.misses / totalRequests : 0,
      oldestItem: oldestKey || undefined,
      newestItem: newestKey || undefined
    }
  }

  /**
   * Cache with fallback function
   */
  async remember<T>(
    key: string,
    fallback: () => Promise<T> | T,
    options: CacheOptions = {}
  ): Promise<T> {
    let cached = await this.get<T>(key, options)
    
    if (cached === null) {
      cached = await fallback()
      this.set(key, cached, options)
    }
    
    return cached
  }

  /**
   * Cache multiple items
   */
  setMany<T>(items: Record<string, T>, options: CacheOptions = {}): void {
    Object.entries(items).forEach(([key, data]) => {
      this.set(key, data, options)
    })
  }

  /**
   * Get multiple items
   */
  async getMany<T>(keys: string[], options: CacheOptions = {}): Promise<Record<string, T | null>> {
    const results: Record<string, T | null> = {}
    
    await Promise.all(
      keys.map(async (key) => {
        results[key] = await this.get<T>(key, options)
      })
    )
    
    return results
  }

  /**
   * Delete multiple items
   */
  deleteMany(keys: string[], options: CacheOptions = {}): void {
    keys.forEach(key => this.delete(key, options))
  }

  /**
   * Get all keys
   */
  getAllKeys(options: CacheOptions = {}): string[] {
    const strategy = options.strategy || this.defaultOptions.strategy
    
    switch (strategy) {
      case 'memory':
        return Array.from(this.memoryCache.keys())
      case 'localStorage':
        return this.getStorageKeys(localStorage)
      case 'sessionStorage':
        return this.getStorageKeys(sessionStorage)
      default:
        return []
    }
  }

  /**
   * Cleanup expired items
   */
  cleanup(options: CacheOptions = {}): number {
    const strategy = options.strategy || this.defaultOptions.strategy
    let cleaned = 0
    
    if (strategy === 'memory') {
      const expiredKeys: string[] = []
      
      this.memoryCache.forEach((item, key) => {
        if (this.isExpired(item)) {
          expiredKeys.push(key)
        }
      })
      
      expiredKeys.forEach(key => {
        this.memoryCache.delete(key)
        cleaned++
      })
    }
    
    return cleaned
  }

  // Memory cache methods
  private setMemory<T>(key: string, item: CacheItem<T>, options: Required<CacheOptions>): void {
    // Enforce max size
    if (this.memoryCache.size >= options.maxSize) {
      this.evictLRU()
    }
    
    this.memoryCache.set(key, item)
  }

  private getMemory<T>(key: string): CacheItem<T> | null {
    return this.memoryCache.get(key) as CacheItem<T> || null
  }

  // Local storage methods
  private setLocalStorage<T>(key: string, item: CacheItem<T>, options: Required<CacheOptions>): void {
    try {
      const serialized = options.serialize ? JSON.stringify(item) : item
      const compressed = options.compress ? this.compress(serialized) : serialized
      localStorage.setItem(this.getStorageKey(key), compressed as string)
    } catch (error) {
      console.warn('Failed to set localStorage cache:', error)
    }
  }

  private getLocalStorage<T>(key: string): CacheItem<T> | null {
    try {
      const stored = localStorage.getItem(this.getStorageKey(key))
      if (!stored) return null
      
      const decompressed = this.decompress(stored)
      return JSON.parse(decompressed) as CacheItem<T>
    } catch (error) {
      console.warn('Failed to get localStorage cache:', error)
      return null
    }
  }

  // Session storage methods  
  private setSessionStorage<T>(key: string, item: CacheItem<T>, options: Required<CacheOptions>): void {
    try {
      const serialized = options.serialize ? JSON.stringify(item) : item
      const compressed = options.compress ? this.compress(serialized) : serialized
      sessionStorage.setItem(this.getStorageKey(key), compressed as string)
    } catch (error) {
      console.warn('Failed to set sessionStorage cache:', error)
    }
  }

  private getSessionStorage<T>(key: string): CacheItem<T> | null {
    try {
      const stored = sessionStorage.getItem(this.getStorageKey(key))
      if (!stored) return null
      
      const decompressed = this.decompress(stored)
      return JSON.parse(decompressed) as CacheItem<T>
    } catch (error) {
      console.warn('Failed to get sessionStorage cache:', error)
      return null
    }
  }

  // IndexedDB methods
  private async setIndexedDB<T>(key: string, item: CacheItem<T>, options: Required<CacheOptions>): Promise<void> {
    // IndexedDB implementation would go here
    // For now, fallback to memory cache
    this.setMemory(key, item, options)
  }

  private async getIndexedDB<T>(key: string): Promise<CacheItem<T> | null> {
    // IndexedDB implementation would go here
    // For now, fallback to memory cache
    return this.getMemory<T>(key)
  }

  private async deleteIndexedDB(key: string): Promise<void> {
    // IndexedDB implementation would go here
    // For now, fallback to memory cache
    this.memoryCache.delete(key)
  }

  private async clearIndexedDB(): Promise<void> {
    // IndexedDB implementation would go here
    // For now, fallback to memory cache
    this.memoryCache.clear()
  }

  // Utility methods
  private isExpired(item: CacheItem): boolean {
    return Date.now() - item.timestamp > item.ttl
  }

  private calculateSize(data: any): number {
    // Rough size calculation
    try {
      return JSON.stringify(data).length
    } catch {
      return 0
    }
  }

  private getStorageKey(key: string): string {
    return `ytempire_cache_${key}`
  }

  private getStorageKeys(storage: Storage): string[] {
    const keys: string[] = []
    const prefix = 'ytempire_cache_'
    
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i)
      if (key?.startsWith(prefix)) {
        keys.push(key.substring(prefix.length))
      }
    }
    
    return keys
  }

  private clearPrefixedStorage(storage: Storage): void {
    const keysToDelete: string[] = []
    const prefix = 'ytempire_cache_'
    
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i)
      if (key?.startsWith(prefix)) {
        keysToDelete.push(key)
      }
    }
    
    keysToDelete.forEach(key => storage.removeItem(key))
  }

  private evictLRU(): void {
    let lruKey = ''
    let lruTime = Infinity
    
    this.memoryCache.forEach((item, key) => {
      const lastAccess = item.timestamp + (item.hits * 1000) // Factor in usage
      if (lastAccess < lruTime) {
        lruTime = lastAccess
        lruKey = key
      }
    })
    
    if (lruKey) {
      this.memoryCache.delete(lruKey)
    }
  }

  private compress(data: any): string {
    // Simple compression placeholder
    // In production, use a proper compression library
    return typeof data === 'string' ? data : JSON.stringify(data)
  }

  private decompress(data: string): string {
    // Simple decompression placeholder
    return data
  }
}

// Create singleton instance
export const cacheService = new CacheService()

// Export convenience methods
export const cache = {
  set: cacheService.set.bind(cacheService),
  get: cacheService.get.bind(cacheService),
  delete: cacheService.delete.bind(cacheService),
  has: cacheService.has.bind(cacheService),
  clear: cacheService.clear.bind(cacheService),
  remember: cacheService.remember.bind(cacheService),
  setMany: cacheService.setMany.bind(cacheService),
  getMany: cacheService.getMany.bind(cacheService),
  deleteMany: cacheService.deleteMany.bind(cacheService),
  getAllKeys: cacheService.getAllKeys.bind(cacheService),
  cleanup: cacheService.cleanup.bind(cacheService),
  getStats: cacheService.getStats.bind(cacheService)
}

export default cacheService