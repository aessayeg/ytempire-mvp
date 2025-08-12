import { io, Socket } from 'socket.io-client';
import { EventEmitter } from '../utils/EventEmitter';

export enum WSEventType {
  // Video events
  VIDEO_GENERATION_STARTED = 'video.generation.started',
  VIDEO_GENERATION_PROGRESS = 'video.generation.progress',
  VIDEO_GENERATION_COMPLETED = 'video.generation.completed',
  VIDEO_GENERATION_FAILED = 'video.generation.failed',
  VIDEO_PUBLISHED = 'video.published',
  VIDEO_ANALYTICS_UPDATE = 'video.analytics.update',
  
  // Channel events
  CHANNEL_STATUS_CHANGED = 'channel.status.changed',
  CHANNEL_METRICS_UPDATE = 'channel.metrics.update',
  CHANNEL_QUOTA_WARNING = 'channel.quota.warning',
  CHANNEL_HEALTH_UPDATE = 'channel.health.update',
  
  // System events
  SYSTEM_ALERT = 'system.alert',
  SYSTEM_METRICS = 'system.metrics',
  COST_ALERT = 'cost.alert',
  PERFORMANCE_WARNING = 'performance.warning',
  
  // User events
  USER_NOTIFICATION = 'user.notification',
  USER_ACTION_REQUIRED = 'user.action.required',
  
  // AI/ML events
  MODEL_UPDATE = 'model.update',
  TREND_DETECTED = 'trend.detected',
  QUALITY_SCORE_UPDATE = 'quality.score.update',
  
  // Connection events
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  RECONNECT = 'reconnect',
  ERROR = 'error'
}

export interface WSMessage<T = any> {
  event: string;
  timestamp: string;
  data: T;
  metadata?: Record<string, any>;
}

export interface VideoGenerationUpdate {
  videoId: string;
  channelId: string;
  status: 'started' | 'processing' | 'completed' | 'failed';
  progress: number;
  currentStep?: string;
  estimatedCompletion?: string;
  error?: string;
  metadata?: Record<string, any>;
}

export interface ChannelMetricsUpdate {
  channelId: string;
  subscribers: number;
  viewsToday: number;
  revenueToday: number;
  videosPublished: number;
  healthScore: number;
  quotaUsed: number;
  quotaLimit: number;
}

export interface SystemMetrics {
  activeGenerations: number;
  queueDepth: number;
  avgGenerationTime: number;
  successRate: number;
  costToday: number;
  apiHealth: Record<string, string>;
  performanceMetrics: Record<string, number>;
}

class WebSocketClient extends EventEmitter {
  private socket: Socket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnected = false;
  private subscriptions: Map<string, Set<Function>> = new Map();
  private messageQueue: WSMessage[] = [];
  private clientId: string;
  
  constructor(url: string) {
    super();
    this.url = url;
    this.clientId = this.generateClientId();
  }
  
  connect(token?: string): void {
    if (this.socket?.connected) {
      console.log('WebSocket already connected');
      return;
    }
    
    const socketUrl = `${this.url}/ws/${this.clientId}`;
    
    this.socket = io(socketUrl, {
      transports: ['websocket'],
      auth: token ? { token } : undefined,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
      reconnectionDelayMax: 10000,
    });
    
    this.setupEventListeners();
  }
  
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      this.emit(WSEventType.DISCONNECT);
    }
  }
  
  private setupEventListeners(): void {
    if (!this.socket) return;
    
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit(WSEventType.CONNECT);
      this.flushMessageQueue();
    });
    
    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isConnected = false;
      this.emit(WSEventType.DISCONNECT, reason);
    });
    
    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.emit(WSEventType.ERROR, error);
    });
    
    this.socket.on('reconnect', (attemptNumber) => {
      console.log(`WebSocket reconnected after ${attemptNumber} attempts`);
      this.emit(WSEventType.RECONNECT, attemptNumber);
    });
    
    // Listen for all custom events
    Object.values(WSEventType).forEach(eventType => {
      if (!['connect', 'disconnect', 'error', 'reconnect'].includes(eventType)) {
        this.socket?.on(eventType, (data: any) => {
          this.handleMessage({ event: eventType, data, timestamp: new Date().toISOString() });
        });
      }
    });
  }
  
  private handleMessage(message: WSMessage): void {
    console.log('WebSocket message received:', message.event);
    
    // Emit to global listeners
    this.emit(message.event, message.data);
    
    // Emit to specific subscribers
    const subscribers = this.subscriptions.get(message.event);
    if (subscribers) {
      subscribers.forEach(callback => callback(message.data));
    }
  }
  
  subscribe(event: WSEventType, callback: Function): () => void {
    if (!this.subscriptions.has(event)) {
      this.subscriptions.set(event, new Set());
    }
    
    this.subscriptions.get(event)?.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.subscriptions.get(event)?.delete(callback);
    };
  }
  
  send(event: string, data: any): void {
    const message: WSMessage = {
      event,
      data,
      timestamp: new Date().toISOString()
    };
    
    if (this.isConnected && this.socket) {
      this.socket.emit(event, message);
    } else {
      // Queue message for later
      this.messageQueue.push(message);
    }
  }
  
  joinRoom(roomId: string): void {
    this.send('subscribe', { roomId });
  }
  
  leaveRoom(roomId: string): void {
    this.send('unsubscribe', { roomId });
  }
  
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.socket?.emit(message.event, message);
      }
    }
  }
  
  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  getConnectionStatus(): boolean {
    return this.isConnected;
  }
  
  getSocket(): Socket | null {
    return this.socket;
  }
}

// Create singleton instance
const wsClient = new WebSocketClient(import.meta.env.VITE_API_URL || 'http://localhost:8000');

// Hook for React components
import { useEffect, useState, useCallback } from 'react';

export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const [lastMessage] = useState<WSMessage | null>(null);
  
  useEffect(() => {
    const handleConnect = () => setConnected(true);
    const handleDisconnect = () => setConnected(false);
    
    wsClient.on(WSEventType.CONNECT, handleConnect);
    wsClient.on(WSEventType.DISCONNECT, handleDisconnect);
    
    // Set initial state
    setConnected(wsClient.getConnectionStatus());
    
    return () => {
      wsClient.off(WSEventType.CONNECT, handleConnect);
      wsClient.off(WSEventType.DISCONNECT, handleDisconnect);
    };
  }, []);
  
  const subscribe = useCallback((event: WSEventType, callback: Function) => {
    return wsClient.subscribe(event, callback);
  }, []);
  
  const send = useCallback((event: string, data: any) => {
    wsClient.send(event, data);
  }, []);
  
  const connect = useCallback((token?: string) => {
    wsClient.connect(token);
  }, []);
  
  const disconnect = useCallback(() => {
    wsClient.disconnect();
  }, []);
  
  const joinRoom = useCallback((roomId: string) => {
    wsClient.joinRoom(roomId);
  }, []);
  
  const leaveRoom = useCallback((roomId: string) => {
    wsClient.leaveRoom(roomId);
  }, []);
  
  return {
    connected,
    lastMessage,
    subscribe,
    send,
    connect,
    disconnect,
    joinRoom,
    leaveRoom
  };
}

// Hook for video generation updates
export function useVideoGenerationUpdates(videoId?: string) {
  const [status, setStatus] = useState<VideoGenerationUpdate | null>(null);
  const { subscribe } = useWebSocket();
  
  useEffect(() => {
    if (!videoId) return;
    
    const unsubscribes = [
      subscribe(WSEventType.VIDEO_GENERATION_STARTED, (data: VideoGenerationUpdate) => {
        if (data.videoId === videoId) setStatus(data);
      }),
      subscribe(WSEventType.VIDEO_GENERATION_PROGRESS, (data: VideoGenerationUpdate) => {
        if (data.videoId === videoId) setStatus(data);
      }),
      subscribe(WSEventType.VIDEO_GENERATION_COMPLETED, (data: VideoGenerationUpdate) => {
        if (data.videoId === videoId) setStatus(data);
      }),
      subscribe(WSEventType.VIDEO_GENERATION_FAILED, (data: VideoGenerationUpdate) => {
        if (data.videoId === videoId) setStatus(data);
      })
    ];
    
    return () => {
      unsubscribes.forEach(unsubscribe => unsubscribe());
    };
  }, [videoId, subscribe]);
  
  return status;
}

// Hook for channel metrics
export function useChannelMetrics(channelId?: string) {
  const [metrics, setMetrics] = useState<ChannelMetricsUpdate | null>(null);
  const { subscribe, joinRoom, leaveRoom } = useWebSocket();
  
  useEffect(() => {
    if (!channelId) return;
    
    // Join channel room
    joinRoom(`channel:${channelId}`);
    
    const unsubscribe = subscribe(WSEventType.CHANNEL_METRICS_UPDATE, (data: ChannelMetricsUpdate) => {
      if (data.channelId === channelId) {
        setMetrics(data);
      }
    });
    
    return () => {
      leaveRoom(`channel:${channelId}`);
      unsubscribe();
    };
  }, [channelId, subscribe, joinRoom, leaveRoom]);
  
  return metrics;
}

// Hook for system metrics
export function useSystemMetrics() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const { subscribe } = useWebSocket();
  
  useEffect(() => {
    const unsubscribe = subscribe(WSEventType.SYSTEM_METRICS, (data: SystemMetrics) => {
      setMetrics(data);
    });
    
    return unsubscribe;
  }, [subscribe]);
  
  return metrics;
}

// Hook for notifications
export function useNotifications() {
  const [notifications, setNotifications] = useState<any[]>([]);
  const { subscribe } = useWebSocket();
  
  useEffect(() => {
    const unsubscribe = subscribe(WSEventType.USER_NOTIFICATION, (data: any) => {
      setNotifications(prev => [...prev, { ...data, id: Date.now(), timestamp: new Date() }]);
    });
    
    return unsubscribe;
  }, [subscribe]);
  
  const clearNotification = useCallback((id: number) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);
  
  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);
  
  return {
    notifications,
    clearNotification,
    clearAll
  };
}

export default wsClient;