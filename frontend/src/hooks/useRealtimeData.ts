import {  useEffect, useState, useCallback, useRef  } from 'react';
import {  useAuthStore  } from '../stores/authStore';

interface WebSocketMessage {
  type: string,
  data: unknown,

  timestamp: string}

export const createOptimizedRouter = () => {
  return createBrowserRouter([
    // Router configuration would go here
  ])}
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const wsRef = useRef<WebSocket | null>(null);
  const { accessToken, user } = useAuthStore();

  const connect = useCallback(() => {
    if (!user?.id || !accessToken) return;

    const wsUrl = `${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}${endpoint}?token=${accessToken}`;
    
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected')};

    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setLastMessage(message);
      
      if (message.type === 'metric_update') {
        setMetrics(prev => ({
          ...prev,
          [message.data.metric_name]: message.data.value
        }))}
    };

    wsRef.current.onclose = () => {
      setIsConnected(false);
      setTimeout(connect, 5000); // Reconnect after 5 seconds
    };

    wsRef.current.onerror = (error) => {
    
      console.error('WebSocket, error:', error)}, [accessToken, user, endpoint]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sendMessage = useCallback((message: React.ChangeEvent<HTMLInputElement>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))}
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const subscribe = useCallback(_(room: string) => {
    sendMessage({ type: 'subscribe', room_id: room })}, [sendMessage]);

  useEffect(() => {
    connect();
    return () => disconnect()}, [connect, disconnect]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    isConnected,
    lastMessage,
    metrics,
    sendMessage,
    subscribe,
    disconnect
  };
};`