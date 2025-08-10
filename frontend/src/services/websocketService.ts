/**
 * WebSocket Service - Real-time communication
 * Owner: Frontend Team Lead
 */

import React, { useState, useEffect } from 'react'
import { env } from '@/config/env'
import { useAuthStore } from '@/stores/authStore'

export interface WebSocketMessage {
  type: string
  data: any
  id?: string
  timestamp?: string
}

export interface WebSocketOptions {
  autoReconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
  pingInterval?: number
  pongTimeout?: number
}

export type WebSocketEventHandler = (data: any) => void
export type WebSocketStatusHandler = (status: WebSocketStatus) => void

export enum WebSocketStatus {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTING = 'disconnecting',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
  RECONNECTING = 'reconnecting'
}

class WebSocketService {
  private ws: WebSocket | null = null
  private eventHandlers: Map<string, Set<WebSocketEventHandler>> = new Map()
  private statusHandlers: Set<WebSocketStatusHandler> = new Set()
  
  private status: WebSocketStatus = WebSocketStatus.DISCONNECTED
  private reconnectAttempts = 0
  private reconnectTimer: NodeJS.Timeout | null = null
  private pingTimer: NodeJS.Timeout | null = null
  private pongTimer: NodeJS.Timeout | null = null
  private messageQueue: WebSocketMessage[] = []

  private options: Required<WebSocketOptions> = {
    autoReconnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 5,
    pingInterval: 30000,
    pongTimeout: 10000
  }

  constructor(options: WebSocketOptions = {}) {
    this.options = { ...this.options, ...options }
  }

  /**
   * Connect to WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve()
        return
      }

      this.setStatus(WebSocketStatus.CONNECTING)

      try {
        const token = useAuthStore.getState().accessToken
        const wsUrl = `${env.WS_URL}?token=${token}`
        
        this.ws = new WebSocket(wsUrl)
        
        this.ws.onopen = () => {
          this.setStatus(WebSocketStatus.CONNECTED)
          this.reconnectAttempts = 0
          this.startPing()
          this.flushMessageQueue()
          resolve()
        }

        this.ws.onmessage = (event) => {
          this.handleMessage(event)
        }

        this.ws.onclose = (event) => {
          this.handleClose(event)
        }

        this.ws.onerror = (error) => {
          this.handleError(error)
          reject(error)
        }

        // Connection timeout
        setTimeout(() => {
          if (this.ws?.readyState === WebSocket.CONNECTING) {
            this.ws.close()
            reject(new Error('WebSocket connection timeout'))
          }
        }, 10000)

      } catch (error) {
        this.setStatus(WebSocketStatus.ERROR)
        reject(error)
      }
    })
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.options.autoReconnect = false
    this.clearReconnectTimer()
    this.stopPing()
    
    if (this.ws) {
      this.setStatus(WebSocketStatus.DISCONNECTING)
      this.ws.close(1000, 'Client disconnect')
    }
  }

  /**
   * Send message to server
   */
  send(message: Omit<WebSocketMessage, 'id' | 'timestamp'>): void {
    const fullMessage: WebSocketMessage = {
      ...message,
      id: this.generateMessageId(),
      timestamp: new Date().toISOString()
    }

    if (this.isConnected()) {
      this.ws!.send(JSON.stringify(fullMessage))
    } else {
      // Queue message for when connection is restored
      this.messageQueue.push(fullMessage)
      
      // Auto-connect if not connected
      if (this.status === WebSocketStatus.DISCONNECTED) {
        this.connect().catch(() => {
          // Connection failed, message remains in queue
        })
      }
    }
  }

  /**
   * Subscribe to event
   */
  on(eventType: string, handler: WebSocketEventHandler): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set())
    }
    
    this.eventHandlers.get(eventType)!.add(handler)
    
    // Return unsubscribe function
    return () => {
      this.off(eventType, handler)
    }
  }

  /**
   * Unsubscribe from event
   */
  off(eventType: string, handler?: WebSocketEventHandler): void {
    if (!handler) {
      // Remove all handlers for event type
      this.eventHandlers.delete(eventType)
    } else {
      // Remove specific handler
      const handlers = this.eventHandlers.get(eventType)
      if (handlers) {
        handlers.delete(handler)
        if (handlers.size === 0) {
          this.eventHandlers.delete(eventType)
        }
      }
    }
  }

  /**
   * Subscribe to connection status changes
   */
  onStatusChange(handler: WebSocketStatusHandler): () => void {
    this.statusHandlers.add(handler)
    
    // Return unsubscribe function
    return () => {
      this.statusHandlers.delete(handler)
    }
  }

  /**
   * Get current connection status
   */
  getStatus(): WebSocketStatus {
    return this.status
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  /**
   * Handle incoming message
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      
      // Handle pong responses
      if (message.type === 'pong') {
        this.handlePong()
        return
      }

      // Emit event to subscribers
      const handlers = this.eventHandlers.get(message.type)
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(message.data)
          } catch (error) {
            console.error('WebSocket event handler error:', error)
          }
        })
      }

      // Log in development
      if (env.IS_DEVELOPMENT && env.ENABLE_DEBUG) {
        console.log('📥 WebSocket Message:', message)
      }

    } catch (error) {
      console.error('Failed to parse WebSocket message:', error)
    }
  }

  /**
   * Handle connection close
   */
  private handleClose(event: CloseEvent): void {
    this.setStatus(WebSocketStatus.DISCONNECTED)
    this.stopPing()

    if (env.IS_DEVELOPMENT) {
      console.log('WebSocket closed:', event.code, event.reason)
    }

    // Auto-reconnect if enabled
    if (this.options.autoReconnect && this.reconnectAttempts < this.options.maxReconnectAttempts) {
      this.scheduleReconnect()
    }
  }

  /**
   * Handle connection error
   */
  private handleError(error: Event): void {
    console.error('WebSocket error:', error)
    this.setStatus(WebSocketStatus.ERROR)
  }

  /**
   * Set connection status
   */
  private setStatus(status: WebSocketStatus): void {
    if (this.status !== status) {
      this.status = status
      this.statusHandlers.forEach(handler => {
        try {
          handler(status)
        } catch (error) {
          console.error('WebSocket status handler error:', error)
        }
      })
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    this.clearReconnectTimer()
    this.setStatus(WebSocketStatus.RECONNECTING)
    
    const delay = this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts)
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++
      this.connect().catch(() => {
        // Reconnection failed, will try again if attempts remain
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
          this.setStatus(WebSocketStatus.ERROR)
        }
      })
    }, delay)
  }

  /**
   * Clear reconnection timer
   */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  /**
   * Start ping mechanism
   */
  private startPing(): void {
    this.stopPing()
    
    this.pingTimer = setInterval(() => {
      if (this.isConnected()) {
        this.send({ type: 'ping', data: {} })
        
        // Set pong timeout
        this.pongTimer = setTimeout(() => {
          // No pong received, close connection
          this.ws?.close(1000, 'Ping timeout')
        }, this.options.pongTimeout)
      }
    }, this.options.pingInterval)
  }

  /**
   * Stop ping mechanism
   */
  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer)
      this.pingTimer = null
    }
    
    if (this.pongTimer) {
      clearTimeout(this.pongTimer)
      this.pongTimer = null
    }
  }

  /**
   * Handle pong response
   */
  private handlePong(): void {
    if (this.pongTimer) {
      clearTimeout(this.pongTimer)
      this.pongTimer = null
    }
  }

  /**
   * Flush queued messages
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected()) {
      const message = this.messageQueue.shift()!
      this.ws!.send(JSON.stringify(message))
    }
  }

  /**
   * Generate unique message ID
   */
  private generateMessageId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }

  /**
   * Get connection statistics
   */
  getStats(): {
    status: WebSocketStatus
    reconnectAttempts: number
    queuedMessages: number
    eventSubscriptions: number
    uptime: number
  } {
    return {
      status: this.status,
      reconnectAttempts: this.reconnectAttempts,
      queuedMessages: this.messageQueue.length,
      eventSubscriptions: Array.from(this.eventHandlers.values()).reduce((sum, handlers) => sum + handlers.size, 0),
      uptime: this.isConnected() ? Date.now() - (this.ws as any)._connectTime || 0 : 0
    }
  }
}

// Singleton instance for application-wide use
export const websocketService = new WebSocketService({
  autoReconnect: true,
  reconnectInterval: 3000,
  maxReconnectAttempts: 10,
  pingInterval: 30000,
  pongTimeout: 10000
})

// React hook for WebSocket integration
export const useWebSocket = () => {
  const [status, setStatus] = useState(websocketService.getStatus())
  
  useEffect(() => {
    const unsubscribe = websocketService.onStatusChange(setStatus)
    return unsubscribe
  }, [])

  return {
    status,
    isConnected: websocketService.isConnected(),
    send: websocketService.send.bind(websocketService),
    on: websocketService.on.bind(websocketService),
    off: websocketService.off.bind(websocketService),
    connect: websocketService.connect.bind(websocketService),
    disconnect: websocketService.disconnect.bind(websocketService),
    stats: websocketService.getStats()
  }
}

// Convenience methods for common events
export const websocket = {
  // Video generation events
  onVideoStatusUpdate: (handler: (data: { video_id: string; status: string; progress?: number }) => void) =>
    websocketService.on('video_status_update', handler),
    
  onQueueUpdate: (handler: (data: { queue_id: string; updates: any }) => void) =>
    websocketService.on('queue_update', handler),

  // Channel events  
  onChannelUpdate: (handler: (data: { channel_id: string; updates: any }) => void) =>
    websocketService.on('channel_update', handler),

  onChannelAnalytics: (handler: (data: { channel_id: string; analytics: any }) => void) =>
    websocketService.on('channel_analytics', handler),

  // System events
  onSystemNotification: (handler: (data: { type: string; message: string; level: string }) => void) =>
    websocketService.on('system_notification', handler),

  onSystemStatus: (handler: (data: { status: string; details?: any }) => void) =>
    websocketService.on('system_status', handler),

  // User events
  onUserUpdate: (handler: (data: { user_id: string; updates: any }) => void) =>
    websocketService.on('user_update', handler),

  // Real-time metrics
  onMetricsUpdate: (handler: (data: any) => void) =>
    websocketService.on('metrics_update', handler),

  // Utility methods
  connect: websocketService.connect.bind(websocketService),
  disconnect: websocketService.disconnect.bind(websocketService),
  send: websocketService.send.bind(websocketService),
  getStatus: websocketService.getStatus.bind(websocketService),
  isConnected: websocketService.isConnected.bind(websocketService),
  getStats: websocketService.getStats.bind(websocketService)
}

export default websocketService