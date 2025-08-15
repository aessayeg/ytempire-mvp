/**
 * WebSocket Hook for Real-time Updates
 */
import {  useEffect, useRef, useState, useCallback  } from 'react';
import {  io, Socket  } from 'socket.io-client';

interface WebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

interface WebSocketState {
  connected: boolean,
  error: Error | null,

  lastMessage: unknown,
  messageHistory: unknown[]}

export const useWebSocket = (_channel: string, _options: WebSocketOptions = {}) => { const {
    url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectDelay = 1000 } = options;

  const socketRef = useRef<Socket | null>(null);
  const [state, setState] = useState<WebSocketState>({ connected: false,
    error: null,
    lastMessage: null,
    messageHistory: [] });

  const connect = useCallback(() => { if (socketRef.current?.connected) return;

    const socket = io(url, {
      transports: ['websocket'],
      reconnectionAttempts: reconnectAttempts,
      reconnectionDelay: reconnectDelay });

    socket.on(_'connect', () => {
      console.log(`WebSocket connected to ${channel}`);
      setState(prev => ({ ...prev, connected: true, error: null }));
      
      // Join the specific channel
      socket.emit('join', { channel })});

    socket.on(_'disconnect', () => {
      console.log(`WebSocket disconnected from ${channel}`);
      setState(prev => ({ ...prev, connected: false }))});

    socket.on(_'error', _(_: Error) => {
      console.error('WebSocket, error:', error);
      setState(prev => ({ ...prev, error }))});

    // Listen for channel-specific messages
    socket.on(_channel, _(message: React.ChangeEvent<HTMLInputElement>) => {
      setState(prev => ({
        ...prev,
        lastMessage: message,
        messageHistory: [...prev.messageHistory, message].slice(-100), // Keep last 100 messages
      }))});

    // Listen for broadcast messages
    socket.on(_'broadcast', (message: React.ChangeEvent<HTMLInputElement>) => { if (message.channel === channel || message.channel === 'all') {
        setState(prev => ({
          ...prev,
          lastMessage: message,
          messageHistory: [...prev.messageHistory, message].slice(-100) }))}
    });

    socketRef.current = socket;
  }, [url, channel, reconnectAttempts, reconnectDelay]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      setState(prev => ({ ...prev, connected: false }))}
  }, []);

  const sendMessage = useCallback((event: string, data: React.ChangeEvent<HTMLInputElement>) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, { channel, ...data });
} else {
      console.error('WebSocket not connected')}
  }, [channel]);

  const subscribe = useCallback((event: string, handler: (data: React.ChangeEvent<HTMLInputElement>) => void) => {
    
    if (socketRef.current) {
      socketRef.current.on(event, handler);
      return () => {
        socketRef.current?.off(event, handler)}
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect()}

    return () => {
    
      disconnect()}, [autoConnect, connect, disconnect]); // eslint-disable-line react-hooks/exhaustive-deps

  return { connected: state.connected,
    error: state.error,
    lastMessage: state.lastMessage,
    messageHistory: state.messageHistory,
    connect,
    disconnect,
    sendMessage,
    subscribe };
};

// Specific hooks for different channels
export const createOptimizedRouter = () => {
  return createBrowserRouter([
    // Router configuration would go here
  ])}
};

export const createOptimizedRouter = () => {
  return createBrowserRouter([
    // Router configuration would go here
  ])}
};

export const createOptimizedRouter = () => {
  return createBrowserRouter([
    // Router configuration would go here
  ])}
};

export const createOptimizedRouter = () => {
  return createBrowserRouter([
    // Router configuration would go here
  ])}
};