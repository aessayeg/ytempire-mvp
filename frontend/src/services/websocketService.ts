/**
 * WebSocket Service for Real-time Updates
 * Handles WebSocket connections with automatic reconnection and message handling
 */

import { EventEmitter } from 'events';

export interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  enableHeartbeat?: boolean;
  debug?: boolean;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
  id?: string;
}

export enum WebSocketStatus {
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  DISCONNECTED = 'DISCONNECTED',
  RECONNECTING = 'RECONNECTING',
  ERROR = 'ERROR',
}

export class WebSocketService extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private status: WebSocketStatus = WebSocketStatus.DISCONNECTED;
  private lastPing: number = 0;
  private latency: number = 0;

  constructor(config: WebSocketConfig) {
    super();
    this.config = {
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      enableHeartbeat: true,
      debug: false,
      ...config,
    };
  }

  /**
   * Connect to WebSocket server
   */
  public connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.log('Already connected');
      return;
    }

    this.status = WebSocketStatus.CONNECTING;
    this.emit('status', this.status);

    try {
      // Add authentication token if available
      const token = localStorage.getItem('token');
      const url = token 
        ? `${this.config.url}?token=${token}`
        : this.config.url;

      this.ws = new WebSocket(url);
      this.setupEventHandlers();
    } catch (error) {
      this.handleError(error);
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    this.clearTimers();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.status = WebSocketStatus.DISCONNECTED;
    this.emit('status', this.status);
  }

  /**
   * Send message through WebSocket
   */
  public send(type: string, data: any): void {
    const message: WebSocketMessage = {
      type,
      data,
      timestamp: Date.now(),
      id: this.generateMessageId(),
    };

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
      this.emit('sent', message);
    } else {
      // Queue message for later sending
      this.messageQueue.push(message);
      this.log('Message queued (not connected):', message);
    }
  }

  /**
   * Subscribe to specific message types
   */
  public subscribe(type: string, callback: (data: any) => void): () => void {
    const handler = (message: WebSocketMessage) => {
      if (message.type === type) {
        callback(message.data);
      }
    };

    this.on('message', handler);

    // Return unsubscribe function
    return () => {
      this.off('message', handler);
    };
  }

  /**
   * Get current connection status
   */
  public getStatus(): WebSocketStatus {
    return this.status;
  }

  /**
   * Get connection latency
   */
  public getLatency(): number {
    return this.latency;
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = this.handleOpen.bind(this);
    this.ws.onclose = this.handleClose.bind(this);
    this.ws.onerror = this.handleError.bind(this);
    this.ws.onmessage = this.handleMessage.bind(this);
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen(): void {
    this.log('WebSocket connected');
    
    this.status = WebSocketStatus.CONNECTED;
    this.reconnectAttempts = 0;
    this.emit('status', this.status);
    this.emit('connected');

    // Start heartbeat
    if (this.config.enableHeartbeat) {
      this.startHeartbeat();
    }

    // Send queued messages
    this.flushMessageQueue();
  }

  /**
   * Handle WebSocket close event
   */
  private handleClose(event: CloseEvent): void {
    this.log('WebSocket closed:', event.code, event.reason);
    
    this.ws = null;
    this.clearTimers();

    if (event.code === 1000) {
      // Normal closure
      this.status = WebSocketStatus.DISCONNECTED;
      this.emit('status', this.status);
      this.emit('disconnected');
    } else {
      // Unexpected closure - attempt reconnect
      this.attemptReconnect();
    }
  }

  /**
   * Handle WebSocket error event
   */
  private handleError(error: any): void {
    this.log('WebSocket error:', error);
    
    this.status = WebSocketStatus.ERROR;
    this.emit('status', this.status);
    this.emit('error', error);
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as WebSocketMessage;
      
      // Handle different message types
      switch (message.type) {
        case 'pong':
          this.handlePong(message);
          break;
        
        case 'error':
          this.emit('error', message.data);
          break;
        
        default:
          this.emit('message', message);
          this.emit(`message:${message.type}`, message.data);
          break;
      }

      this.log('Received message:', message);
    } catch (error) {
      this.log('Failed to parse message:', event.data, error);
    }
  }

  /**
   * Attempt to reconnect
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.log('Max reconnection attempts reached');
      this.status = WebSocketStatus.DISCONNECTED;
      this.emit('status', this.status);
      this.emit('reconnectFailed');
      return;
    }

    this.status = WebSocketStatus.RECONNECTING;
    this.emit('status', this.status);
    this.reconnectAttempts++;

    const delay = this.getReconnectDelay();
    this.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Get reconnection delay with exponential backoff
   */
  private getReconnectDelay(): number {
    const baseDelay = this.config.reconnectInterval;
    const factor = Math.min(this.reconnectAttempts, 5);
    return baseDelay * Math.pow(1.5, factor);
  }

  /**
   * Start heartbeat mechanism
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.lastPing = Date.now();
        this.send('ping', { timestamp: this.lastPing });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Handle pong response
   */
  private handlePong(message: WebSocketMessage): void {
    if (this.lastPing) {
      this.latency = Date.now() - this.lastPing;
      this.emit('latency', this.latency);
    }
  }

  /**
   * Clear all timers
   */
  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Flush queued messages
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message.type, message.data);
      }
    }
  }

  /**
   * Generate unique message ID
   */
  private generateMessageId(): string {
    return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Log debug messages
   */
  private log(...args: any[]): void {
    if (this.config.debug) {
      console.log('[WebSocket]', ...args);
    }
  }
}

// Singleton instance
let wsInstance: WebSocketService | null = null;

/**
 * Get or create WebSocket service instance
 */
export function getWebSocketService(config?: WebSocketConfig): WebSocketService {
  if (!wsInstance && config) {
    wsInstance = new WebSocketService(config);
  }
  
  if (!wsInstance) {
    throw new Error('WebSocket service not initialized');
  }
  
  return wsInstance;
}

/**
 * Initialize WebSocket service
 */
export function initializeWebSocket(config: WebSocketConfig): WebSocketService {
  if (wsInstance) {
    wsInstance.disconnect();
  }
  
  wsInstance = new WebSocketService(config);
  return wsInstance;
}