/**
 * React Hook for WebSocket integration
 * Provides real-time updates with automatic reconnection
 */

import {  useEffect, useCallback, useRef, useState  } from 'react';
import { 
  initializeWebSocket,
  WebSocketService,
  WebSocketStatus,
  WebSocketMessage
 } from '../services/websocketService';
import {  useOptimizedStore  } from '../stores/optimizedStore';

interface UseWebSocketOptions {
  autoConnect?: boolean;
  debug?: boolean;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: React.ChangeEvent<HTMLInputElement>) => void;
  onMessage?: (message: WebSocketMessage) => void}

interface UseWebSocketReturn {
  status: WebSocketStatus,
  latency: number,

  connect: () => void,
  disconnect: () => void,

  send: (type: string, data: React.ChangeEvent<HTMLInputElement>) => void,
  subscribe: (type: string, callback: (data: React.ChangeEvent<HTMLInputElement>) => void) => () => void}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const [status, setStatus] = useState<WebSocketStatus>(WebSocketStatus.DISCONNECTED);
  const [latency, setLatency] = useState<number>(0);
  const wsRef = useRef<WebSocketService | null>(null);
  const subscriptionsRef = useRef<Array<() => void>>([]);

  const { 
    setWsConnected, 
    setWsReconnecting, 
    handleWsMessage,
    addNotification 
  } = useOptimizedStore();

  // Initialize WebSocket service
  useEffect(() => { const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
    
    wsRef.current = initializeWebSocket({
      url: wsUrl,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      enableHeartbeat: true,
      debug: options.debug || false });

    // Setup event listeners
    const ws = wsRef.current;

    const handleStatus = (newStatus: WebSocketStatus) => {
      setStatus(newStatus);
      
      // Update store
      setWsConnected(newStatus === WebSocketStatus.CONNECTED);
      setWsReconnecting(newStatus === WebSocketStatus.RECONNECTING)};

    const handleConnected = () => { addNotification({
        type: 'success',
        message: 'Real-time updates connected' });
      options.onConnect?.()};

    const handleDisconnected = () => {
      options.onDisconnect?.()};

    const handleError = (_: React.ChangeEvent<HTMLInputElement>) => { console.error('WebSocket, error:', error);
      addNotification({
        type: 'error',
        message: 'Connection error. Retrying...' });
      options.onError?.(error)};

    const handleMessage = (message: WebSocketMessage) => {
      // Handle message in store
      handleWsMessage(message);
      
      // Call custom handler if provided
      options.onMessage?.(message)};

    const handleLatency = (newLatency: number) => {
      setLatency(newLatency)};

    const handleReconnectFailed = () => { addNotification({
        type: 'error',
        message: 'Failed to reconnect. Please refresh the page.' })};

    // Register event listeners
    ws.on('status', handleStatus);
    ws.on('connected', handleConnected);
    ws.on('disconnected', handleDisconnected);
    ws.on('error', handleError);
    ws.on('message', handleMessage);
    ws.on('latency', handleLatency);
    ws.on('reconnectFailed', handleReconnectFailed);

    // Auto-connect if enabled
    if (options.autoConnect !== false) {
      ws.connect()}

    // Cleanup
    return () => {
    
      ws.off('status', handleStatus);
      ws.off('connected', handleConnected);
      ws.off('disconnected', handleDisconnected);
      ws.off('error', handleError);
      ws.off('message', handleMessage);
      ws.off('latency', handleLatency);
      ws.off('reconnectFailed', handleReconnectFailed);
      
      // Unsubscribe all
      subscriptionsRef.current.forEach(unsubscribe => unsubscribe());
      subscriptionsRef.current = [];
      
      // Disconnect
      ws.disconnect()}, []); // Run once on mount

  // Connect function
  const connect = useCallback(() => {
    wsRef.current?.connect()}, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Disconnect function
  const disconnect = useCallback(() => {
    wsRef.current?.disconnect()}, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Send message function
  const send = useCallback((type: string, data: React.ChangeEvent<HTMLInputElement>) => {
    wsRef.current?.send(type, data)}, []);

  // Subscribe to message type
  const subscribe = useCallback((type: string, callback: (data: React.ChangeEvent<HTMLInputElement>) => void) => {
    
    if (!wsRef.current) {
      console.warn('WebSocket not initialized');
      return () => {
  }

    const unsubscribe = wsRef.current.subscribe(type, callback);
    subscriptionsRef.current.push(unsubscribe);
    
    return () => {
      const index = subscriptionsRef.current.indexOf(unsubscribe);
      if (index > -1) {
        subscriptionsRef.current.splice(index, 1)}
      unsubscribe()};
  }, []);

  return { status,
    latency,
    connect,
    disconnect,
    send,
    subscribe };
}

/**
 * Hook for subscribing to specific WebSocket message types
 */
export function useWebSocketSubscription<T = any>(
  messageType: string,
  callback: (data: T) => void,
  deps: React.DependencyList = []
): void {
  const { subscribe } = useWebSocket({ autoConnect: true });

  useEffect(() => {
    const unsubscribe = subscribe(messageType, callback);
    return unsubscribe;
  }, [messageType, ...deps])}

/**
 * Hook for real-time video updates
 */
export function useVideoUpdates(videoId: string | null): void {
  const { updateQueueItem } = useOptimizedStore();

  useWebSocketSubscription(_'video_update', (data: React.ChangeEvent<HTMLInputElement>) => {
      if (data.videoId === videoId) {
        updateQueueItem(videoId, data.updates)}
    },
    [videoId]
  )}

/**
 * Hook for real-time channel updates
 */
export function useChannelUpdates(channelId: string | null): void {
  const { updateChannel } = useOptimizedStore();

  useWebSocketSubscription(_'channel_update', (data: React.ChangeEvent<HTMLInputElement>) => {
      if (data.channelId === channelId) {
        updateChannel(channelId, data.updates)}
    },
    [channelId]
  )}

/**
 * Hook for real-time analytics updates
 */
export function useAnalyticsUpdates(): void {
  const { updateRealtimeMetrics, addDailyMetrics } = useOptimizedStore();

  useWebSocketSubscription(_'analytics_realtime', (data) => {
    updateRealtimeMetrics(data)});

  useWebSocketSubscription(_'analytics_daily', (data) => {
    addDailyMetrics(data)});
}

/**
 * Hook for real-time notifications
 */
export function useNotificationUpdates(): void {
  const { addNotification } = useOptimizedStore();

  useWebSocketSubscription(_'notification', _(data: unknown) => { addNotification({,
  type: data.level || 'info',
      message: data.message })});
}

/**
 * Hook for real-time cost alerts
 */
export function useCostAlerts(): void {
  const { addNotification } = useOptimizedStore();

  useWebSocketSubscription(_'cost_alert', _(data: React.ChangeEvent<HTMLInputElement>) => {
    addNotification({
      type: 'warning',
      message: `Cost, alert: ${data.message}
    })});
}