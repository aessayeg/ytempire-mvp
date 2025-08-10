/**
 * Upload Service - File upload utilities and management
 * Owner: Frontend Team Lead
 */

import { apiService } from './apiService'

export interface UploadOptions {
  fieldName?: string
  allowedTypes?: string[]
  maxSize?: number
  chunkSize?: number
  onProgress?: (progress: UploadProgress) => void
  onError?: (error: UploadError) => void
  onSuccess?: (result: UploadResult) => void
  metadata?: Record<string, any>
}

export interface UploadProgress {
  loaded: number
  total: number
  percentage: number
  speed?: number
  estimatedTime?: number
}

export interface UploadError {
  code: string
  message: string
  details?: any
}

export interface UploadResult {
  id: string
  url: string
  filename: string
  size: number
  mimeType: string
  metadata?: Record<string, any>
}

export interface ChunkedUploadSession {
  id: string
  url: string
  chunkSize: number
  totalChunks: number
  uploadedChunks: Set<number>
  file: File
  options: UploadOptions
}

class UploadService {
  private activeUploads: Map<string, AbortController> = new Map()
  private chunkedSessions: Map<string, ChunkedUploadSession> = new Map()

  // Default configurations
  private defaultOptions: UploadOptions = {
    fieldName: 'file',
    allowedTypes: [],
    maxSize: 100 * 1024 * 1024, // 100MB
    chunkSize: 5 * 1024 * 1024, // 5MB chunks for large files
  }

  /**
   * Upload a single file
   */
  async uploadFile(
    endpoint: string,
    file: File,
    options: UploadOptions = {}
  ): Promise<UploadResult> {
    const mergedOptions = { ...this.defaultOptions, ...options }
    
    // Validate file
    this.validateFile(file, mergedOptions)
    
    const uploadId = this.generateUploadId()
    const controller = new AbortController()
    this.activeUploads.set(uploadId, controller)

    try {
      // Use chunked upload for large files
      if (file.size > mergedOptions.chunkSize! * 2) {
        return await this.chunkedUpload(endpoint, file, uploadId, mergedOptions)
      } else {
        return await this.simpleUpload(endpoint, file, uploadId, mergedOptions)
      }
    } finally {
      this.activeUploads.delete(uploadId)
    }
  }

  /**
   * Upload multiple files
   */
  async uploadFiles(
    endpoint: string,
    files: File[],
    options: UploadOptions = {}
  ): Promise<UploadResult[]> {
    const uploadPromises = files.map(file => 
      this.uploadFile(endpoint, file, options)
    )
    
    return await Promise.all(uploadPromises)
  }

  /**
   * Upload with progress tracking
   */
  async uploadWithProgress(
    endpoint: string,
    file: File,
    onProgress: (progress: UploadProgress) => void,
    options: UploadOptions = {}
  ): Promise<UploadResult> {
    return this.uploadFile(endpoint, file, {
      ...options,
      onProgress
    })
  }

  /**
   * Simple file upload for smaller files
   */
  private async simpleUpload(
    endpoint: string,
    file: File,
    uploadId: string,
    options: UploadOptions
  ): Promise<UploadResult> {
    const formData = new FormData()
    formData.append(options.fieldName!, file)
    
    // Add metadata
    if (options.metadata) {
      Object.entries(options.metadata).forEach(([key, value]) => {
        formData.append(key, typeof value === 'string' ? value : JSON.stringify(value))
      })
    }

    const controller = this.activeUploads.get(uploadId)
    let lastLoaded = 0
    let startTime = Date.now()

    const result = await apiService.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      signal: controller?.signal,
      onUploadProgress: (progressEvent) => {
        const { loaded, total } = progressEvent
        const now = Date.now()
        const timeDiff = (now - startTime) / 1000
        const speed = loaded / timeDiff
        const estimatedTime = (total - loaded) / speed

        const progress: UploadProgress = {
          loaded,
          total,
          percentage: Math.round((loaded / total) * 100),
          speed,
          estimatedTime
        }

        options.onProgress?.(progress)
        lastLoaded = loaded
      }
    })

    const uploadResult: UploadResult = {
      id: result.id || this.generateUploadId(),
      url: result.url || result.file_url,
      filename: file.name,
      size: file.size,
      mimeType: file.type,
      metadata: result.metadata
    }

    options.onSuccess?.(uploadResult)
    return uploadResult
  }

  /**
   * Chunked upload for large files
   */
  private async chunkedUpload(
    endpoint: string,
    file: File,
    uploadId: string,
    options: UploadOptions
  ): Promise<UploadResult> {
    const chunkSize = options.chunkSize!
    const totalChunks = Math.ceil(file.size / chunkSize)
    
    // Initialize upload session
    const session = await this.initializeChunkedUpload(endpoint, file, totalChunks, options)
    this.chunkedSessions.set(uploadId, session)

    try {
      // Upload chunks
      for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
        if (!session.uploadedChunks.has(chunkIndex)) {
          await this.uploadChunk(session, chunkIndex, options)
        }
      }

      // Finalize upload
      return await this.finalizeChunkedUpload(session, options)
    } finally {
      this.chunkedSessions.delete(uploadId)
    }
  }

  /**
   * Initialize chunked upload session
   */
  private async initializeChunkedUpload(
    endpoint: string,
    file: File,
    totalChunks: number,
    options: UploadOptions
  ): Promise<ChunkedUploadSession> {
    const response = await apiService.post(`${endpoint}/chunked/init`, {
      filename: file.name,
      size: file.size,
      mimeType: file.type,
      totalChunks,
      chunkSize: options.chunkSize,
      metadata: options.metadata
    })

    return {
      id: response.upload_id,
      url: response.upload_url,
      chunkSize: options.chunkSize!,
      totalChunks,
      uploadedChunks: new Set(response.uploaded_chunks || []),
      file,
      options
    }
  }

  /**
   * Upload a single chunk
   */
  private async uploadChunk(
    session: ChunkedUploadSession,
    chunkIndex: number,
    options: UploadOptions
  ): Promise<void> {
    const start = chunkIndex * session.chunkSize
    const end = Math.min(start + session.chunkSize, session.file.size)
    const chunk = session.file.slice(start, end)

    const formData = new FormData()
    formData.append('chunk', chunk)
    formData.append('chunk_index', chunkIndex.toString())
    formData.append('upload_id', session.id)

    await apiService.post(`${session.url}/chunk`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: (progressEvent) => {
        const chunkProgress = (progressEvent.loaded / progressEvent.total) * 100
        const totalProgress = ((session.uploadedChunks.size * 100) + chunkProgress) / session.totalChunks

        options.onProgress?.({
          loaded: Math.round((totalProgress / 100) * session.file.size),
          total: session.file.size,
          percentage: Math.round(totalProgress)
        })
      }
    })

    session.uploadedChunks.add(chunkIndex)
  }

  /**
   * Finalize chunked upload
   */
  private async finalizeChunkedUpload(
    session: ChunkedUploadSession,
    options: UploadOptions
  ): Promise<UploadResult> {
    const response = await apiService.post(`${session.url}/finalize`, {
      upload_id: session.id
    })

    const result: UploadResult = {
      id: response.id,
      url: response.url,
      filename: session.file.name,
      size: session.file.size,
      mimeType: session.file.type,
      metadata: response.metadata
    }

    options.onSuccess?.(result)
    return result
  }

  /**
   * Resume chunked upload
   */
  async resumeUpload(uploadId: string): Promise<UploadResult> {
    const session = this.chunkedSessions.get(uploadId)
    if (!session) {
      throw new Error('Upload session not found')
    }

    // Check upload status
    const status = await apiService.get(`${session.url}/status`)
    session.uploadedChunks = new Set(status.uploaded_chunks)

    // Continue upload
    for (let chunkIndex = 0; chunkIndex < session.totalChunks; chunkIndex++) {
      if (!session.uploadedChunks.has(chunkIndex)) {
        await this.uploadChunk(session, chunkIndex, session.options)
      }
    }

    return await this.finalizeChunkedUpload(session, session.options)
  }

  /**
   * Cancel upload
   */
  cancelUpload(uploadId: string): void {
    // Cancel HTTP request
    const controller = this.activeUploads.get(uploadId)
    if (controller) {
      controller.abort()
      this.activeUploads.delete(uploadId)
    }

    // Clean up chunked session
    const session = this.chunkedSessions.get(uploadId)
    if (session) {
      // Notify server to cleanup
      apiService.delete(`${session.url}/cancel`).catch(() => {
        // Ignore cleanup errors
      })
      this.chunkedSessions.delete(uploadId)
    }
  }

  /**
   * Get upload progress
   */
  getUploadProgress(uploadId: string): UploadProgress | null {
    const session = this.chunkedSessions.get(uploadId)
    if (!session) {
      return null
    }

    const uploaded = session.uploadedChunks.size * session.chunkSize
    return {
      loaded: Math.min(uploaded, session.file.size),
      total: session.file.size,
      percentage: Math.round((uploaded / session.file.size) * 100)
    }
  }

  /**
   * Validate file before upload
   */
  private validateFile(file: File, options: UploadOptions): void {
    // Check file size
    if (options.maxSize && file.size > options.maxSize) {
      const maxSizeMB = Math.round(options.maxSize / (1024 * 1024))
      throw new UploadError('FILE_TOO_LARGE', `File size exceeds ${maxSizeMB}MB limit`)
    }

    // Check file type
    if (options.allowedTypes?.length && !options.allowedTypes.includes(file.type)) {
      throw new UploadError('INVALID_FILE_TYPE', `File type ${file.type} is not allowed`)
    }

    // Check if file is empty
    if (file.size === 0) {
      throw new UploadError('EMPTY_FILE', 'File is empty')
    }
  }

  /**
   * Generate unique upload ID
   */
  private generateUploadId(): string {
    return `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  /**
   * Get file preview
   */
  async getFilePreview(file: File): Promise<string | null> {
    if (!file.type.startsWith('image/')) {
      return null
    }

    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target?.result as string)
      reader.onerror = () => resolve(null)
      reader.readAsDataURL(file)
    })
  }

  /**
   * Compress image before upload
   */
  async compressImage(
    file: File,
    options: {
      maxWidth?: number
      maxHeight?: number
      quality?: number
      format?: 'jpeg' | 'png' | 'webp'
    } = {}
  ): Promise<File> {
    if (!file.type.startsWith('image/')) {
      return file
    }

    const {
      maxWidth = 1920,
      maxHeight = 1080,
      quality = 0.8,
      format = 'jpeg'
    } = options

    return new Promise((resolve) => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')!
      const img = new Image()

      img.onload = () => {
        // Calculate new dimensions
        let { width, height } = img
        if (width > height) {
          if (width > maxWidth) {
            height = (height * maxWidth) / width
            width = maxWidth
          }
        } else {
          if (height > maxHeight) {
            width = (width * maxHeight) / height
            height = maxHeight
          }
        }

        canvas.width = width
        canvas.height = height

        // Draw and compress
        ctx.drawImage(img, 0, 0, width, height)
        
        canvas.toBlob(
          (blob) => {
            const compressedFile = new File([blob!], file.name, {
              type: `image/${format}`,
              lastModified: Date.now()
            })
            resolve(compressedFile)
          },
          `image/${format}`,
          quality
        )
      }

      img.src = URL.createObjectURL(file)
    })
  }

  /**
   * Bulk upload with concurrency control
   */
  async bulkUpload(
    endpoint: string,
    files: File[],
    options: UploadOptions & {
      concurrency?: number
    } = {}
  ): Promise<UploadResult[]> {
    const concurrency = options.concurrency || 3
    const results: UploadResult[] = []
    
    for (let i = 0; i < files.length; i += concurrency) {
      const batch = files.slice(i, i + concurrency)
      const batchResults = await Promise.all(
        batch.map(file => this.uploadFile(endpoint, file, options))
      )
      results.push(...batchResults)
    }

    return results
  }
}

// Custom error class
class UploadError extends Error {
  constructor(public code: string, message: string, public details?: any) {
    super(message)
    this.name = 'UploadError'
  }
}

// Create singleton instance
export const uploadService = new UploadService()

// Export convenience methods
export const upload = {
  file: uploadService.uploadFile.bind(uploadService),
  files: uploadService.uploadFiles.bind(uploadService),
  withProgress: uploadService.uploadWithProgress.bind(uploadService),
  resume: uploadService.resumeUpload.bind(uploadService),
  cancel: uploadService.cancelUpload.bind(uploadService),
  getProgress: uploadService.getUploadProgress.bind(uploadService),
  getPreview: uploadService.getFilePreview.bind(uploadService),
  compressImage: uploadService.compressImage.bind(uploadService),
  bulk: uploadService.bulkUpload.bind(uploadService)
}

export { UploadError }
export default uploadService