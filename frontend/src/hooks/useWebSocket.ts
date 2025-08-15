/**
 * WebSocket Hook for Real-time Updates
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

interface WebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

interface WebSocketState {
  connected: boolean;
  error: Error | null;
  lastMessage: unknown;
  messageHistory: unknown[];
}

export const useWebSocket = (
  channel: string,
  options: WebSocketOptions = {}
) => {
  const {
    url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectDelay = 1000,
  } = options;

  const socketRef = useRef<Socket | null>(null);
  const [state, setState] = useState<WebSocketState>({
    connected: false,
    error: null,
    lastMessage: null,
    messageHistory: [],
  });

  const connect = useCallback(() => {
    if (socketRef.current?.connected) return;

    const socket = io(url, {
      transports: ['websocket'],
      reconnectionAttempts: reconnectAttempts,
      reconnectionDelay: reconnectDelay,
    });

    socket.on('connect', () => {
      console.log(`WebSocket connected to ${channel}`);
      setState(prev => ({ ...prev, connected: true, error: null }));
      
      // Join the specific channel
      socket.emit('join', { channel });
    });

    socket.on('disconnect', () => {
      console.log(`WebSocket disconnected from ${channel}`);
      setState(prev => ({ ...prev, connected: false }));
    });

    socket.on('error', (_error: Error) => {
      console.error('WebSocket error:', error);
      setState(prev => ({ ...prev, error }));
    });

    // Listen for channel-specific messages
    socket.on(channel, (message: unknown) => {
      setState(prev => ({
        ...prev,
        lastMessage: message,
        messageHistory: [...prev.messageHistory, message].slice(-100), // Keep last 100 messages
      }));
    });

    // Listen for broadcast messages
    socket.on('broadcast', (message: unknown) => {
      if (message.channel === channel || message.channel === 'all') {
        setState(prev => ({
          ...prev,
          lastMessage: message,
          messageHistory: [...prev.messageHistory, message].slice(-100),
        }));
      }
    });

    socketRef.current = socket;
  }, [url, channel, reconnectAttempts, reconnectDelay]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      setState(prev => ({ ...prev, connected: false }));
    }
  }, []);

  const sendMessage = useCallback((event: string, data: unknown) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, { channel, ...data });
    } else {
      console.error('WebSocket not connected');
    }
  }, [channel]);

  const subscribe = useCallback((event: string, handler: (data: unknown) => void) => {
    if (socketRef.current) {
      socketRef.current.on(event, handler);
      return () => {
        socketRef.current?.off(event, handler);
      };
    }
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    connected: state.connected,
    error: state.error,
    lastMessage: state.lastMessage,
    messageHistory: state.messageHistory,
    connect,
    disconnect,
    sendMessage,
    subscribe,
  };
};

// Specific hooks for different channels
export const useDashboardWebSocket = () => {
  return useWebSocket('dashboard');
};

export const useVideoGenerationWebSocket = (videoId?: string) => {
  return useWebSocket(videoId ? `video-generation-${videoId}` : 'video-generation');
};

export const useNotificationsWebSocket = () => {
  return useWebSocket('notifications');
};

export const useMetricsWebSocket = () => {
  return useWebSocket('metrics');
};