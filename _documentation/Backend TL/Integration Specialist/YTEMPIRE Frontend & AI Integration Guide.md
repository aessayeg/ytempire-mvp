# YTEMPIRE Frontend & AI Integration Guide

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Backend Team Lead  
**Audience**: Integration Specialist  
**Scope**: Frontend Specifications, AI Team Integration, and Communication Patterns

---

## Table of Contents
1. [Frontend Architecture Specifications](#1-frontend-architecture-specifications)
2. [AI Team Integration Contracts](#2-ai-team-integration-contracts)
3. [API Communication Patterns](#3-api-communication-patterns)
4. [Dashboard UI Components](#4-dashboard-ui-components)
5. [State Management Architecture](#5-state-management-architecture)

---

## 1. Frontend Architecture Specifications

### 1.1 React Application Structure

```javascript
// Frontend Application Structure
ytempire-frontend/
├── src/
│   ├── components/           // Reusable UI components
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Table.tsx
│   │   │   └── Form.tsx
│   │   ├── dashboard/
│   │   │   ├── MetricsCard.tsx
│   │   │   ├── RevenueChart.tsx
│   │   │   ├── VideoQueue.tsx
│   │   │   └── ChannelStatus.tsx
│   │   ├── channels/
│   │   │   ├── ChannelList.tsx
│   │   │   ├── ChannelCard.tsx
│   │   │   ├── ChannelSettings.tsx
│   │   │   └── ChannelAnalytics.tsx
│   │   └── videos/
│   │       ├── VideoList.tsx
│   │       ├── VideoPlayer.tsx
│   │       ├── VideoEditor.tsx
│   │       └── VideoMetrics.tsx
│   ├── pages/                // Page components
│   │   ├── Dashboard.tsx
│   │   ├── Channels.tsx
│   │   ├── Videos.tsx
│   │   ├── Analytics.tsx
│   │   ├── Settings.tsx
│   │   └── Billing.tsx
│   ├── hooks/                // Custom React hooks
│   │   ├── useAuth.ts
│   │   ├── useWebSocket.ts
│   │   ├── useApi.ts
│   │   └── useNotifications.ts
│   ├── services/            // API service layer
│   │   ├── api.ts
│   │   ├── auth.service.ts
│   │   ├── channel.service.ts
│   │   ├── video.service.ts
│   │   └── analytics.service.ts
│   ├── store/               // State management
│   │   ├── index.ts
│   │   ├── auth.slice.ts
│   │   ├── channel.slice.ts
│   │   ├── video.slice.ts
│   │   └── ui.slice.ts
│   ├── types/               // TypeScript types
│   │   ├── user.types.ts
│   │   ├── channel.types.ts
│   │   ├── video.types.ts
│   │   └── api.types.ts
│   └── utils/               // Utility functions
│       ├── constants.ts
│       ├── helpers.ts
│       └── validators.ts
```

### 1.2 Technology Stack Implementation

```json
{
  "name": "ytempire-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@reduxjs/toolkit": "^1.9.7",
    "react-redux": "^8.1.3",
    "@mui/material": "^5.14.20",
    "@mui/icons-material": "^5.14.19",
    "@emotion/react": "^11.11.1",
    "@emotion/styled": "^11.11.0",
    "axios": "^1.6.2",
    "socket.io-client": "^4.5.4",
    "recharts": "^2.10.3",
    "react-hook-form": "^7.48.2",
    "yup": "^1.3.3",
    "@tanstack/react-query": "^5.12.2",
    "date-fns": "^2.30.0",
    "react-player": "^2.13.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0",
    "vite": "^5.0.8",
    "vitest": "^1.0.4",
    "@testing-library/react": "^14.1.2",
    "@testing-library/jest-dom": "^6.1.5"
  }
}
```

### 1.3 Component Architecture

```typescript
// components/dashboard/DashboardMetrics.tsx
import React from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import { TrendingUp, VideoLibrary, AttachMoney, Speed } from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';

interface MetricCardProps {
  title: string;
  value: string | number;
  change: number;
  icon: React.ReactNode;
  color: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, icon, color }) => {
  return (
    <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Box>
          <Typography color="textSecondary" gutterBottom variant="h6">
            {title}
          </Typography>
          <Typography variant="h4" component="h2">
            {value}
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: change >= 0 ? 'success.main' : 'error.main',
              mt: 1
            }}
          >
            {change >= 0 ? '+' : ''}{change}% from last month
          </Typography>
        </Box>
        <Box
          sx={{
            backgroundColor: color,
            borderRadius: 2,
            p: 1.5,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {icon}
        </Box>
      </Box>
    </Paper>
  );
};

export const DashboardMetrics: React.FC = () => {
  const metrics = useSelector((state: RootState) => state.analytics.metrics);
  
  const metricCards = [
    {
      title: 'Total Revenue',
      value: `$${metrics.totalRevenue.toLocaleString()}`,
      change: metrics.revenueChange,
      icon: <AttachMoney sx={{ color: 'white', fontSize: 30 }} />,
      color: '#4caf50'
    },
    {
      title: 'Videos Generated',
      value: metrics.videosGenerated,
      change: metrics.videosChange,
      icon: <VideoLibrary sx={{ color: 'white', fontSize: 30 }} />,
      color: '#2196f3'
    },
    {
      title: 'Active Channels',
      value: metrics.activeChannels,
      change: metrics.channelsChange,
      icon: <Speed sx={{ color: 'white', fontSize: 30 }} />,
      color: '#ff9800'
    },
    {
      title: 'Avg Cost/Video',
      value: `$${metrics.avgCostPerVideo.toFixed(2)}`,
      change: -metrics.costChange, // Negative is good for costs
      icon: <TrendingUp sx={{ color: 'white', fontSize: 30 }} />,
      color: '#9c27b0'
    }
  ];
  
  return (
    <Grid container spacing={3}>
      {metricCards.map((card, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <MetricCard {...card} />
        </Grid>
      ))}
    </Grid>
  );
};
```

### 1.4 Dashboard Layout

```typescript
// pages/Dashboard.tsx
import React, { useEffect } from 'react';
import { Container, Grid, Paper, Box, Typography } from '@mui/material';
import { DashboardMetrics } from '../components/dashboard/DashboardMetrics';
import { VideoQueue } from '../components/dashboard/VideoQueue';
import { RevenueChart } from '../components/dashboard/RevenueChart';
import { ChannelPerformance } from '../components/dashboard/ChannelPerformance';
import { RecentActivity } from '../components/dashboard/RecentActivity';
import { useDispatch } from 'react-redux';
import { fetchDashboardData } from '../store/dashboard.slice';

export const Dashboard: React.FC = () => {
  const dispatch = useDispatch();
  
  useEffect(() => {
    dispatch(fetchDashboardData());
    
    // Set up real-time updates
    const interval = setInterval(() => {
      dispatch(fetchDashboardData());
    }, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, [dispatch]);
  
  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      {/* Key Metrics */}
      <Box mb={3}>
        <DashboardMetrics />
      </Box>
      
      <Grid container spacing={3}>
        {/* Revenue Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 360 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Revenue Trend
            </Typography>
            <RevenueChart />
          </Paper>
        </Grid>
        
        {/* Video Queue Status */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 360 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Video Queue
            </Typography>
            <VideoQueue />
          </Paper>
        </Grid>
        
        {/* Channel Performance */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Channel Performance
            </Typography>
            <ChannelPerformance />
          </Paper>
        </Grid>
        
        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Recent Activity
            </Typography>
            <RecentActivity />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};
```

---

## 2. AI Team Integration Contracts

### 2.1 AI Service Endpoints

```typescript
// AI Team API Contract
interface AIServiceEndpoints {
  // Script Generation
  '/api/v1/ai/generate-script': {
    method: 'POST';
    request: {
      topic: string;
      niche: string;
      style: 'educational' | 'entertaining' | 'news' | 'tutorial';
      length: number; // Target video length in seconds
      optimization_level: 'economy' | 'standard' | 'premium';
      keywords?: string[];
      tone?: 'professional' | 'casual' | 'humorous' | 'serious';
    };
    response: {
      script: string;
      title: string;
      description: string;
      tags: string[];
      thumbnail_suggestions: string[];
      hooks: {
        intro: string;
        outro: string;
      };
      quality_score: number; // 0-1
      estimated_duration: number;
      cost: number;
      model_used: string;
    };
  };
  
  // Topic Generation
  '/api/v1/ai/generate-topics': {
    method: 'POST';
    request: {
      niche: string;
      count: number;
      trending: boolean;
      competitor_analysis?: boolean;
      target_audience?: {
        age_range: string;
        interests: string[];
        location?: string;
      };
    };
    response: {
      topics: Array<{
        title: string;
        score: number; // Viability score 0-1
        competition: 'low' | 'medium' | 'high';
        search_volume: number;
        trend_direction: 'rising' | 'stable' | 'declining';
        keywords: string[];
      }>;
    };
  };
  
  // Content Quality Scoring
  '/api/v1/ai/score-content': {
    method: 'POST';
    request: {
      content_type: 'script' | 'title' | 'description' | 'thumbnail';
      content: string | Buffer;
      niche: string;
      target_metrics?: {
        engagement_rate?: number;
        ctr?: number;
        retention?: number;
      };
    };
    response: {
      quality_score: number; // 0-1
      improvements: string[];
      predicted_performance: {
        views: number;
        engagement_rate: number;
        ctr: number;
      };
      compliance: {
        youtube_guidelines: boolean;
        copyright_risk: 'low' | 'medium' | 'high';
        monetization_safe: boolean;
      };
    };
  };
  
  // Niche Analysis
  '/api/v1/ai/analyze-niche': {
    method: 'POST';
    request: {
      niche: string;
      depth: 'basic' | 'detailed' | 'comprehensive';
    };
    response: {
      viability_score: number;
      competition_level: number;
      monetization_potential: number;
      growth_rate: number;
      sub_niches: string[];
      content_gaps: string[];
      recommended_strategy: {
        content_types: string[];
        posting_frequency: number;
        best_times: string[];
      };
    };
  };
}
```

### 2.2 AI Model Integration Implementation

```python
# ai_integration.py - Backend Integration with AI Team Services
from typing import Dict, Optional, List
import aiohttp
import asyncio
from datetime import datetime

class AIServiceIntegration:
    """
    Integration layer for AI team services
    """
    
    def __init__(self):
        self.ai_base_url = "http://localhost:8001"  # AI service endpoint
        self.timeout = 30  # seconds
        self.retry_attempts = 3
        
        # Model configurations
        self.model_configs = {
            'economy': {
                'model': 'gpt-3.5-turbo',
                'max_tokens': 1500,
                'temperature': 0.7,
                'quality_threshold': 0.6
            },
            'standard': {
                'model': 'gpt-3.5-turbo',
                'max_tokens': 2000,
                'temperature': 0.8,
                'quality_threshold': 0.75
            },
            'premium': {
                'model': 'gpt-4',
                'max_tokens': 3000,
                'temperature': 0.7,
                'quality_threshold': 0.85
            }
        }
    
    async def generate_script(
        self,
        topic: str,
        niche: str,
        style: str = 'educational',
        optimization_level: str = 'standard'
    ) -> Dict:
        """
        Generate video script using AI service
        """
        
        config = self.model_configs[optimization_level]
        
        payload = {
            'topic': topic,
            'niche': niche,
            'style': style,
            'optimization_level': optimization_level,
            'length': 480,  # 8 minutes default
            'model_config': config
        }
        
        # Call AI service with retry logic
        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.ai_base_url}/api/v1/ai/generate-script",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Validate quality score
                            if result['quality_score'] < config['quality_threshold']:
                                # Regenerate with higher quality settings
                                if optimization_level != 'premium':
                                    return await self.generate_script(
                                        topic, niche, style, 'premium'
                                    )
                            
                            return result
                        
                        elif response.status == 503:  # Service unavailable
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        
                        else:
                            error = await response.text()
                            raise Exception(f"AI service error: {error}")
                            
            except asyncio.TimeoutError:
                if attempt == self.retry_attempts - 1:
                    # Use fallback template generation
                    return await self._generate_template_script(topic, niche, style)
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("AI service unavailable after retries")
    
    async def score_content(
        self,
        content: str,
        content_type: str = 'script'
    ) -> Dict:
        """
        Score content quality using AI
        """
        
        payload = {
            'content_type': content_type,
            'content': content,
            'niche': 'general'  # Will be extracted from content
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ai_base_url}/api/v1/ai/score-content",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                
                # Return default scores if service unavailable
                return {
                    'quality_score': 0.7,
                    'improvements': [],
                    'predicted_performance': {
                        'views': 1000,
                        'engagement_rate': 0.05,
                        'ctr': 0.03
                    },
                    'compliance': {
                        'youtube_guidelines': True,
                        'copyright_risk': 'low',
                        'monetization_safe': True
                    }
                }
    
    async def _generate_template_script(
        self,
        topic: str,
        niche: str,
        style: str
    ) -> Dict:
        """
        Fallback template-based script generation
        """
        
        templates = {
            'educational': """
                Introduction: Welcome back to our channel! Today we're exploring {topic}.
                
                Section 1: Let's start with the basics of {topic}.
                [Main content about {topic} in {niche} context]
                
                Section 2: Here are the key points to remember:
                [Key points about {topic}]
                
                Section 3: Practical applications:
                [How to apply {topic} in real life]
                
                Conclusion: Thanks for watching! Don't forget to like and subscribe.
            """,
            'entertaining': """
                Hook: You won't believe what we discovered about {topic}!
                
                Story: Let me tell you about {topic} in the world of {niche}.
                [Engaging story about {topic}]
                
                Fun Facts: Did you know these amazing facts about {topic}?
                [Interesting facts]
                
                Wrap-up: That's all for today! Hit that subscribe button for more!
            """
        }
        
        script = templates.get(style, templates['educational']).format(
            topic=topic,
            niche=niche
        )
        
        return {
            'script': script,
            'title': f"Everything You Need to Know About {topic}",
            'description': f"In this video, we explore {topic} in detail.",
            'tags': [topic.lower(), niche.lower(), style],
            'thumbnail_suggestions': [f"{topic} thumbnail"],
            'hooks': {
                'intro': f"Welcome! Today's topic: {topic}",
                'outro': "Thanks for watching!"
            },
            'quality_score': 0.6,  # Lower score for template
            'estimated_duration': 480,
            'cost': 0.05,  # Minimal cost for template
            'model_used': 'template_fallback'
        }
```

### 2.3 AI Data Pipeline

```python
class AIDataPipeline:
    """
    Data pipeline for AI model training and improvement
    """
    
    def __init__(self):
        self.data_collection_enabled = True
        self.anonymization_enabled = True
        
    async def collect_performance_data(self, video_id: str) -> Dict:
        """
        Collect video performance data for AI training
        """
        
        # Get video performance metrics
        performance = await self.db.fetch_one("""
            SELECT 
                v.title,
                v.script,
                v.tags,
                pm.views,
                pm.likes,
                pm.average_view_duration_seconds,
                pm.click_through_rate
            FROM videos.video_records v
            JOIN videos.performance_metrics pm ON v.id = pm.video_id
            WHERE v.id = $1
        """, video_id)
        
        if not performance:
            return {}
        
        # Anonymize data
        if self.anonymization_enabled:
            performance = self._anonymize_data(performance)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            views=performance['views'],
            likes=performance['likes'],
            retention=performance['average_view_duration_seconds'] / 480  # Assume 8 min videos
        )
        
        # Send to AI team for model improvement
        await self._send_to_ai_training({
            'script_features': self._extract_script_features(performance['script']),
            'title_features': self._extract_title_features(performance['title']),
            'performance_score': performance_score,
            'metrics': {
                'views': performance['views'],
                'engagement': performance['likes'] / max(performance['views'], 1),
                'retention': performance['average_view_duration_seconds'],
                'ctr': performance['click_through_rate']
            }
        })
        
        return {
            'video_id': video_id,
            'performance_score': performance_score,
            'data_collected': True
        }
```

---

## 3. API Communication Patterns

### 3.1 API Client Service

```typescript
// services/api.service.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { store } from '../store';
import { logout } from '../store/auth.slice';

class ApiService {
  private api: AxiosInstance;
  private refreshingToken: Promise<string> | null = null;
  
  constructor() {
    this.api = axios.create({
      baseURL: process.env.VITE_API_URL || 'http://localhost:8000/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        const token = store.getState().auth.accessToken;
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Response interceptor
    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;
        
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          
          try {
            const newToken = await this.refreshToken();
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            return this.api(originalRequest);
          } catch (refreshError) {
            store.dispatch(logout());
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }
        
        return Promise.reject(error);
      }
    );
  }
  
  private async refreshToken(): Promise<string> {
    if (!this.refreshingToken) {
      this.refreshingToken = this.api
        .post('/auth/refresh', {
          refresh_token: store.getState().auth.refreshToken
        })
        .then((response) => {
          const { access_token } = response.data;
          store.dispatch(setAccessToken(access_token));
          this.refreshingToken = null;
          return access_token;
        });
    }
    
    return this.refreshingToken;
  }
  
  // Generic request methods
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.api.get<T>(url, config);
    return response.data;
  }
  
  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.api.post<T>(url, data, config);
    return response.data;
  }
  
  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.api.put<T>(url, data, config);
    return response.data;
  }
  
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.api.delete<T>(url, config);
    return response.data;
  }
  
  // File upload
  async uploadFile(url: string, file: File, onProgress?: (progress: number) => void): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.api.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      }
    });
  }
}

export const apiService = new ApiService();
```

### 3.2 WebSocket Connection

```typescript
// services/websocket.service.ts
import { io, Socket } from 'socket.io-client';
import { store } from '../store';
import { updateVideoStatus, addNotification } from '../store/app.slice';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  connect(): void {
    const token = store.getState().auth.accessToken;
    
    this.socket = io(process.env.VITE_WS_URL || 'http://localhost:8000', {
      auth: {
        token
      },
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: this.maxReconnectAttempts
    });
    
    this.setupEventListeners();
  }
  
  private setupEventListeners(): void {
    if (!this.socket) return;
    
    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.subscribeToUpdates();
    });
    
    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
    });
    
    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        store.dispatch(addNotification({
          type: 'error',
          message: 'Lost connection to server. Please refresh the page.'
        }));
      }
    });
    
    // Application events
    this.socket.on('video.status.update', (data) => {
      store.dispatch(updateVideoStatus(data));
    });
    
    this.socket.on('video.generation.complete', (data) => {
      store.dispatch(addNotification({
        type: 'success',
        message: `Video "${data.title}" has been generated successfully!`
      }));
    });
    
    this.socket.on('video.generation.failed', (data) => {
      store.dispatch(addNotification({
        type: 'error',
        message: `Video generation failed: ${data.error}`
      }));
    });
    
    this.socket.on('channel.quota.warning', (data) => {
      store.dispatch(addNotification({
        type: 'warning',
        message: `Channel ${data.channel_name} is approaching quota limit (${data.usage}%)`
      }));
    });
    
    this.socket.on('cost.threshold.exceeded', (data) => {
      store.dispatch(addNotification({
        type: 'warning',
        message: `Video cost ($${data.cost}) exceeds threshold`
      }));
    });
  }
  
  private subscribeToUpdates(): void {
    if (!this.socket) return;
    
    // Subscribe to user-specific updates
    const userId = store.getState().auth.user?.id;
    if (userId) {
      this.socket.emit('subscribe', {
        channels: [`user.${userId}`, 'system.alerts']
      });
    }
  }
  
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
  
  emit(event: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    }
  }
}

export const websocketService = new WebSocketService();
```

---

## 4. Dashboard UI Components

### 4.1 Video Queue Component

```typescript
// components/dashboard/VideoQueue.tsx
import React, { useEffect, useState } from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Typography,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Cancel,
  CheckCircle,
  Error,
  HourglassEmpty
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { videoService } from '../../services/video.service';

interface QueueItem {
  id: string;
  title: string;
  channel: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  estimatedTime?: number;
  error?: string;
}

export const VideoQueue: React.FC = () => {
  const dispatch = useDispatch();
  const queue = useSelector((state: RootState) => state.videos.queue);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'queued':
        return <HourglassEmpty color="action" />;
      case 'processing':
        return <PlayArrow color="primary" />;
      case 'completed':
        return <CheckCircle color="success" />;
      case 'failed':
        return <Error color="error" />;
      default:
        return <HourglassEmpty />;
    }
  };
  
  const getStatusColor = (status: string): any => {
    switch (status) {
      case 'queued':
        return 'default';
      case 'processing':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };
  
  const handleCancelVideo = async (videoId: string) => {
    try {
      await videoService.cancelVideo(videoId);
      dispatch(removeFromQueue(videoId));
    } catch (error) {
      console.error('Failed to cancel video:', error);
    }
  };
  
  const formatEstimatedTime = (seconds?: number) => {
    if (!seconds) return 'Unknown';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };
  
  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <List>
        {queue.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="textSecondary">
              No videos in queue
            </Typography>
          </Box>
        ) : (
          queue.map((item) => (
            <ListItem
              key={item.id}
              selected={selectedItem === item.id}
              onClick={() => setSelectedItem(item.id)}
              secondaryAction={
                item.status === 'queued' || item.status === 'processing' ? (
                  <Tooltip title="Cancel">
                    <IconButton
                      edge="end"
                      aria-label="cancel"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCancelVideo(item.id);
                      }}
                    >
                      <Cancel />
                    </IconButton>
                  </Tooltip>
                ) : null
              }
            >
              <ListItemAvatar>
                <Avatar>{getStatusIcon(item.status)}</Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                      {item.title}
                    </Typography>
                    <Chip
                      label={item.status}
                      size="small"
                      color={getStatusColor(item.status)}
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="caption" color="textSecondary">
                      {item.channel} • ETA: {formatEstimatedTime(item.estimatedTime)}
                    </Typography>
                    {item.status === 'processing' && (
                      <LinearProgress
                        variant="determinate"
                        value={item.progress}
                        sx={{ mt: 1 }}
                      />
                    )}
                    {item.error && (
                      <Typography variant="caption" color="error">
                        {item.error}
                      </Typography>
                    )}
                  </Box>
                }
              />
            </ListItem>
          ))
        )}
      </List>
    </Box>
  );
};
```

### 4.2 Channel Management Component

```typescript
// components/channels/ChannelCard.tsx
import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  Menu,
  MenuItem
} from '@mui/material';
import {
  MoreVert,
  PlayArrow,
  Pause,
  Settings,
  Analytics,
  YouTube
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface ChannelCardProps {
  channel: {
    id: string;
    name: string;
    niche: string;
    status: 'active' | 'paused' | 'suspended';
    subscriberCount: number;
    videoCount: number;
    monthlyRevenue: number;
    healthScore: number;
    automationEnabled: boolean;
  };
  onToggleAutomation: (channelId: string) => void;
}

export const ChannelCard: React.FC<ChannelCardProps> = ({ channel, onToggleAutomation }) => {
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const getStatusColor = () => {
    switch (channel.status) {
      case 'active':
        return 'success';
      case 'paused':
        return 'warning';
      case 'suspended':
        return 'error';
      default:
        return 'default';
    }
  };
  
  const getHealthColor = (score: number) => {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.6) return '#ff9800';
    return '#f44336';
  };
  
  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flex: 1 }}>
        <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <YouTube color="error" />
            <Typography variant="h6" component="div">
              {channel.name}
            </Typography>
          </Box>
          <IconButton size="small" onClick={handleMenuOpen}>
            <MoreVert />
          </IconButton>
        </Box>
        
        <Box display="flex" gap={1} mb={2}>
          <Chip label={channel.niche} size="small" variant="outlined" />
          <Chip label={channel.status} size="small" color={getStatusColor()} />
        </Box>
        
        <Box mb={2}>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Health Score
          </Typography>
          <Box display="flex" alignItems="center" gap={1}>
            <LinearProgress
              variant="determinate"
              value={channel.healthScore * 100}
              sx={{
                flex: 1,
                height: 8,
                borderRadius: 4,
                '& .MuiLinearProgress-bar': {
                  backgroundColor: getHealthColor(channel.healthScore)
                }
              }}
            />
            <Typography variant="body2">
              {Math.round(channel.healthScore * 100)}%
            </Typography>
          </Box>
        </Box>
        
        <Box display="flex" justifyContent="space-between" mb={1}>
          <Typography variant="body2" color="textSecondary">
            Subscribers
          </Typography>
          <Typography variant="body2">
            {channel.subscriberCount.toLocaleString()}
          </Typography>
        </Box>
        
        <Box display="flex" justifyContent="space-between" mb={1}>
          <Typography variant="body2" color="textSecondary">
            Videos
          </Typography>
          <Typography variant="body2">
            {channel.videoCount}
          </Typography>
        </Box>
        
        <Box display="flex" justifyContent="space-between">
          <Typography variant="body2" color="textSecondary">
            Monthly Revenue
          </Typography>
          <Typography variant="body2" fontWeight="bold">
            ${channel.monthlyRevenue.toFixed(2)}
          </Typography>
        </Box>
      </CardContent>
      
      <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
        <Button
          startIcon={channel.automationEnabled ? <Pause /> : <PlayArrow />}
          onClick={() => onToggleAutomation(channel.id)}
          color={channel.automationEnabled ? 'warning' : 'primary'}
        >
          {channel.automationEnabled ? 'Pause' : 'Start'}
        </Button>
        
        <Button
          startIcon={<Analytics />}
          onClick={() => navigate(`/channels/${channel.id}/analytics`)}
        >
          Analytics
        </Button>
      </CardActions>
      
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => {
          handleMenuClose();
          navigate(`/channels/${channel.id}/settings`);
        }}>
          <Settings fontSize="small" sx={{ mr: 1 }} />
          Settings
        </MenuItem>
        <MenuItem onClick={() => {
          handleMenuClose();
          navigate(`/channels/${channel.id}/videos`);
        }}>
          View Videos
        </MenuItem>
        <MenuItem onClick={() => {
          handleMenuClose();
          // Handle refresh OAuth
        }}>
          Refresh OAuth
        </MenuItem>
      </Menu>
    </Card>
  );
};
```

---

## 5. State Management Architecture

### 5.1 Redux Store Configuration

```typescript
// store/index.ts
import { configureStore } from '@reduxjs/toolkit';
import authReducer from './auth.slice';
import channelsReducer from './channels.slice';
import videosReducer from './videos.slice';
import analyticsReducer from './analytics.slice';
import uiReducer from './ui.slice';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    channels: channelsReducer,
    videos: videosReducer,
    analytics: analyticsReducer,
    ui: uiReducer
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['auth/setTokens'],
        ignoredPaths: ['auth.expiresAt']
      }
    })
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

### 5.2 Channel State Slice

```typescript
// store/channels.slice.ts
import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { channelService } from '../services/channel.service';

interface Channel {
  id: string;
  name: string;
  niche: string;
  status: 'active' | 'paused' | 'suspended';
  subscriberCount: number;
  videoCount: number;
  monthlyRevenue: number;
  healthScore: number;
  automationEnabled: boolean;
}

interface ChannelsState {
  channels: Channel[];
  selectedChannel: Channel | null;
  loading: boolean;
  error: string | null;
}

const initialState: ChannelsState = {
  channels: [],
  selectedChannel: null,
  loading: false,
  error: null
};

// Async thunks
export const fetchChannels = createAsyncThunk(
  'channels/fetchAll',
  async () => {
    const response = await channelService.getChannels();
    return response;
  }
);

export const createChannel = createAsyncThunk(
  'channels/create',
  async (channelData: Partial<Channel>) => {
    const response = await channelService.createChannel(channelData);
    return response;
  }
);

export const updateChannel = createAsyncThunk(
  'channels/update',
  async ({ id, data }: { id: string; data: Partial<Channel> }) => {
    const response = await channelService.updateChannel(id, data);
    return response;
  }
);

export const toggleAutomation = createAsyncThunk(
  'channels/toggleAutomation',
  async (channelId: string) => {
    const response = await channelService.toggleAutomation(channelId);
    return response;
  }
);

// Slice
const channelsSlice = createSlice({
  name: 'channels',
  initialState,
  reducers: {
    selectChannel: (state, action: PayloadAction<string>) => {
      state.selectedChannel = state.channels.find(c => c.id === action.payload) || null;
    },
    updateChannelMetrics: (state, action: PayloadAction<{ id: string; metrics: Partial<Channel> }>) => {
      const channel = state.channels.find(c => c.id === action.payload.id);
      if (channel) {
        Object.assign(channel, action.payload.metrics);
      }
    }
  },
  extraReducers: (builder) => {
    builder
      // Fetch channels
      .addCase(fetchChannels.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchChannels.fulfilled, (state, action) => {
        state.loading = false;
        state.channels = action.payload;
      })
      .addCase(fetchChannels.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch channels';
      })
      
      // Create channel
      .addCase(createChannel.fulfilled, (state, action) => {
        state.channels.push(action.payload);
      })
      
      // Update channel
      .addCase(updateChannel.fulfilled, (state, action) => {
        const index = state.channels.findIndex(c => c.id === action.payload.id);
        if (index !== -1) {
          state.channels[index] = action.payload;
        }
      })
      
      // Toggle automation
      .addCase(toggleAutomation.fulfilled, (state, action) => {
        const channel = state.channels.find(c => c.id === action.payload.id);
        if (channel) {
          channel.automationEnabled = action.payload.automationEnabled;
        }
      });
  }
});

export const { selectChannel, updateChannelMetrics } = channelsSlice.actions;
export default channelsSlice.reducer;
```

---

## Key Integration Points for the Integration Specialist

### Frontend-Backend Communication
1. **REST API**: All CRUD operations use the REST API with JWT authentication
2. **WebSocket**: Real-time updates for video processing, notifications, and metrics
3. **File Uploads**: Direct upload to backend with progress tracking

### AI Service Integration
1. **Script Generation**: Direct API calls to AI service with fallback to templates
2. **Quality Scoring**: All content passes through AI quality checks
3. **Performance Feedback**: Video performance data feeds back to AI for improvement

### State Management
1. **Redux Toolkit**: Centralized state management with async thunks
2. **React Query**: For server state and caching (optional addition)
3. **Local Storage**: For user preferences and draft content

### Component Architecture
1. **Material-UI**: Consistent design system
2. **Responsive Design**: Mobile-first approach
3. **Code Splitting**: Lazy loading for performance

### Testing Strategy
1. **Unit Tests**: For utility functions and reducers
2. **Integration Tests**: For API communication
3. **E2E Tests**: For critical user flows

---

**Document Status**: Complete  
**Next Steps**: Implement API endpoints first, then build frontend components incrementally  
**Priority**: Focus on Dashboard and Video Queue for MVP demonstration