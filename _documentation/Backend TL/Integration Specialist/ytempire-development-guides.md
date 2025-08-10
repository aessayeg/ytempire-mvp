# YTEMPIRE Development Guides

## 7.1 Backend Development

### Development Environment Setup

```bash
# Backend development setup
cd ytempire-backend

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Environment variables
cp .env.example .env
# Edit .env with your credentials

# Database setup
docker-compose up -d postgres redis
alembic upgrade head

# Run development server
uvicorn app.main:app --reload --port 8000
```

### FastAPI Application Structure

```python
# app/main.py - Main application entry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, channels, videos, analytics
from app.core.config import settings

app = FastAPI(
    title="YTEMPIRE API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await database.connect()
    await redis.connect()
    
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await database.disconnect()
    await redis.close()
```

### API Development Standards

```python
# app/routers/channels.py - Example router
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.schemas import Channel, ChannelCreate, ChannelUpdate
from app.services import channel_service
from app.auth import get_current_user

router = APIRouter()

@router.get("/", response_model=List[Channel])
async def list_channels(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """
    List all channels for current user.
    
    - **skip**: Number of records to skip
    - **limit**: Maximum records to return
    """
    channels = await channel_service.get_user_channels(
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    return channels

@router.post("/", response_model=Channel, status_code=status.HTTP_201_CREATED)
async def create_channel(
    channel: ChannelCreate,
    current_user = Depends(get_current_user)
):
    """Create a new YouTube channel connection."""
    
    # Check channel limit
    if await channel_service.count_user_channels(current_user.id) >= current_user.channel_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Channel limit reached for your subscription"
        )
    
    return await channel_service.create_channel(
        user_id=current_user.id,
        channel_data=channel
    )
```

### Database Operations

```python
# app/repositories/video_repository.py
from typing import List, Optional
from sqlalchemy import select, update
from app.database import database
from app.models import videos_table

class VideoRepository:
    """Database operations for videos"""
    
    async def create(self, video_data: dict) -> dict:
        """Create new video record"""
        query = videos_table.insert().values(**video_data)
        result = await database.execute(query)
        return {**video_data, "id": result}
    
    async def get_by_id(self, video_id: str) -> Optional[dict]:
        """Get video by ID"""
        query = select(videos_table).where(videos_table.c.id == video_id)
        return await database.fetch_one(query)
    
    async def update_status(self, video_id: str, status: str) -> bool:
        """Update video processing status"""
        query = (
            update(videos_table)
            .where(videos_table.c.id == video_id)
            .values(status=status, updated_at=datetime.utcnow())
        )
        result = await database.execute(query)
        return result > 0
    
    async def get_processing_queue(self, limit: int = 10) -> List[dict]:
        """Get videos queued for processing"""
        query = (
            select(videos_table)
            .where(videos_table.c.status == "queued")
            .order_by(videos_table.c.created_at)
            .limit(limit)
        )
        return await database.fetch_all(query)
```

### Testing Guidelines

```python
# tests/test_api/test_channels.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_channel():
    """Test channel creation"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Login first
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "testpass"}
        )
        token = login_response.json()["access_token"]
        
        # Create channel
        response = await client.post(
            "/api/v1/channels",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "youtube_channel_id": "UC123456",
                "channel_title": "Test Channel",
                "niche": "technology"
            }
        )
        
        assert response.status_code == 201
        assert response.json()["channel_title"] == "Test Channel"

@pytest.mark.asyncio
async def test_channel_limit():
    """Test channel limit enforcement"""
    # Test user with 5 channel limit
    # Try to create 6th channel
    # Should return 403 Forbidden
    pass
```

## 7.2 Frontend Development

### Development Environment Setup

```bash
# Frontend development setup
cd ytempire-frontend

# Install dependencies
npm install

# Environment variables
cp .env.example .env.local
# Edit .env.local with API URL

# Run development server
npm run dev
# Open http://localhost:3000
```

### React Component Standards

```typescript
// components/channels/ChannelCard.tsx
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { YouTube } from '@mui/icons-material';
import { Channel } from '../../types/channel';

interface ChannelCardProps {
  channel: Channel;
  onSelect: (channel: Channel) => void;
}

export const ChannelCard: React.FC<ChannelCardProps> = ({ 
  channel, 
  onSelect 
}) => {
  const getHealthColor = (score: number): string => {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.6) return '#ff9800';
    return '#f44336';
  };

  return (
    <Card 
      sx={{ cursor: 'pointer' }}
      onClick={() => onSelect(channel)}
    >
      <CardContent>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <YouTube color="error" />
          <Typography variant="h6">
            {channel.title}
          </Typography>
        </Box>
        
        <Box display="flex" gap={1} mb={2}>
          <Chip 
            label={channel.niche} 
            size="small" 
            variant="outlined" 
          />
          <Chip 
            label={channel.status} 
            size="small" 
            color={channel.status === 'active' ? 'success' : 'default'}
          />
        </Box>
        
        <Box>
          <Typography variant="body2" color="textSecondary">
            Health Score
          </Typography>
          <Box 
            sx={{ 
              width: '100%', 
              height: 8, 
              bgcolor: 'grey.300',
              borderRadius: 4,
              overflow: 'hidden'
            }}
          >
            <Box
              sx={{
                width: `${channel.healthScore * 100}%`,
                height: '100%',
                bgcolor: getHealthColor(channel.healthScore)
              }}
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};
```

### State Management with Zustand

```typescript
// stores/channelStore.ts
import { create } from 'zustand';
import { Channel } from '../types/channel';
import { channelService } from '../services/channel.service';

interface ChannelStore {
  channels: Channel[];
  selectedChannel: Channel | null;
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchChannels: () => Promise<void>;
  selectChannel: (channel: Channel) => void;
  updateChannel: (id: string, data: Partial<Channel>) => Promise<void>;
  deleteChannel: (id: string) => Promise<void>;
}

export const useChannelStore = create<ChannelStore>((set, get) => ({
  channels: [],
  selectedChannel: null,
  loading: false,
  error: null,
  
  fetchChannels: async () => {
    set({ loading: true, error: null });
    try {
      const channels = await channelService.getChannels();
      set({ channels, loading: false });
    } catch (error) {
      set({ error: error.message, loading: false });
    }
  },
  
  selectChannel: (channel) => {
    set({ selectedChannel: channel });
  },
  
  updateChannel: async (id, data) => {
    try {
      const updated = await channelService.updateChannel(id, data);
      set(state => ({
        channels: state.channels.map(c => 
          c.id === id ? updated : c
        )
      }));
    } catch (error) {
      set({ error: error.message });
    }
  },
  
  deleteChannel: async (id) => {
    try {
      await channelService.deleteChannel(id);
      set(state => ({
        channels: state.channels.filter(c => c.id !== id)
      }));
    } catch (error) {
      set({ error: error.message });
    }
  }
}));
```

### API Integration

```typescript
// services/api.service.ts
import axios, { AxiosInstance } from 'axios';

class ApiService {
  private api: AxiosInstance;
  
  constructor() {
    this.api = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1',
      timeout: 30000,
    });
    
    // Request interceptor for auth
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Token expired, try refresh
          await this.refreshToken();
        }
        return Promise.reject(error);
      }
    );
  }
  
  async get<T>(url: string, params?: any): Promise<T> {
    const response = await this.api.get<T>(url, { params });
    return response.data;
  }
  
  async post<T>(url: string, data?: any): Promise<T> {
    const response = await this.api.post<T>(url, data);
    return response.data;
  }
  
  async put<T>(url: string, data?: any): Promise<T> {
    const response = await this.api.put<T>(url, data);
    return response.data;
  }
  
  async delete<T>(url: string): Promise<T> {
    const response = await this.api.delete<T>(url);
    return response.data;
  }
}

export const apiService = new ApiService();
```

## 7.3 AI/ML Development

### Model Development Workflow

```python
# ml/models/trend_predictor.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TrendPredictor(nn.Module):
    """Neural network for trend prediction"""
    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)

# Training script
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = evaluate(model, val_loader)
        
        print(f"Epoch {epoch}: Train Loss: {total_loss}, Val Loss: {val_loss}")
```

### Model Serving

```python
# ml/serving/inference.py
from typing import Dict, List
import torch
from transformers import pipeline

class InferenceService:
    """Model inference service"""
    
    def __init__(self):
        self.models = {
            'trend_predictor': self.load_trend_model(),
            'script_generator': self.load_script_model(),
            'quality_scorer': self.load_quality_model()
        }
    
    def load_trend_model(self):
        """Load trend prediction model"""
        model = TrendPredictor()
        model.load_state_dict(torch.load('models/trend_predictor.pt'))
        model.eval()
        return model
    
    async def predict_trend(self, topic: str) -> Dict:
        """Predict trend score for topic"""
        model = self.models['trend_predictor']
        
        # Preprocess input
        inputs = self.tokenizer(topic, return_tensors='pt')
        
        # Inference
        with torch.no_grad():
            score = model(**inputs).item()
        
        return {
            'topic': topic,
            'trend_score': score,
            'trending': score > 0.7,
            'confidence': abs(score - 0.5) * 2
        }
    
    async def generate_script(self, prompt: str, style: str) -> str:
        """Generate video script"""
        generator = self.models['script_generator']
        
        full_prompt = f"Style: {style}\n{prompt}"
        
        script = generator(
            full_prompt,
            max_length=1000,
            temperature=0.7,
            top_p=0.9
        )
        
        return script[0]['generated_text']
```

## 7.4 API Documentation

### OpenAPI Specification

```yaml
openapi: 3.0.0
info:
  title: YTEMPIRE API
  version: 1.0.0
  description: Automated YouTube content platform API

servers:
  - url: http://localhost:8000/api/v1
    description: Development server
  - url: https://api.ytempire.com/v1
    description: Production server

paths:
  /auth/login:
    post:
      summary: User login
      tags: [Authentication]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                email:
                  type: string
                  format: email
                password:
                  type: string
                  format: password
      responses:
        200:
          description: Successful login
          content:
            application/json:
              schema:
                type: object
                properties:
                  access_token:
                    type: string
                  refresh_token:
                    type: string
                  token_type:
                    type: string
        401:
          description: Invalid credentials

  /channels:
    get:
      summary: List user channels
      tags: [Channels]
      security:
        - bearerAuth: []
      parameters:
        - name: skip
          in: query
          schema:
            type: integer
            default: 0
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
      responses:
        200:
          description: List of channels
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Channel'

components:
  schemas:
    Channel:
      type: object
      properties:
        id:
          type: string
          format: uuid
        youtube_channel_id:
          type: string
        channel_title:
          type: string
        niche:
          type: string
        status:
          type: string
          enum: [active, paused, suspended]
        health_score:
          type: number
          minimum: 0
          maximum: 1

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

## 7.5 Testing Protocols

### Test Strategy

```yaml
Testing_Levels:
  Unit_Tests:
    coverage: 80%
    tools: [pytest, jest, vitest]
    frequency: On every commit
    
  Integration_Tests:
    coverage: 100% of API endpoints
    tools: [pytest-asyncio, supertest]
    frequency: Before merge
    
  End_to_End_Tests:
    scenarios: 20 critical user flows
    tools: [Playwright, Selenium]
    frequency: Daily
    
  Performance_Tests:
    targets:
      - API response <500ms
      - Dashboard load <2s
      - Video generation <10min
    tools: [k6, Apache Bench]
    frequency: Weekly
    
  Security_Tests:
    checks:
      - SQL injection
      - XSS vulnerabilities
      - Authentication bypass
      - API rate limiting
    tools: [OWASP ZAP, Burp Suite]
    frequency: Before release
```

### Test Examples

```python
# Backend unit test
def test_cost_calculation():
    """Test video cost calculation"""
    costs = {
        'script': 0.30,
        'voice': 0.08,
        'processing': 0.05
    }
    
    total = calculate_total_cost(costs)
    
    assert total == 0.43
    assert total < 3.00  # Under limit

# Frontend unit test
describe('ChannelCard', () => {
  it('displays channel information', () => {
    const channel = {
      id: '123',
      title: 'Test Channel',
      niche: 'technology',
      healthScore: 0.85
    };
    
    const { getByText } = render(
      <ChannelCard channel={channel} onSelect={jest.fn()} />
    );
    
    expect(getByText('Test Channel')).toBeInTheDocument();
    expect(getByText('technology')).toBeInTheDocument();
  });
});

# E2E test
async def test_video_generation_flow():
    """Test complete video generation flow"""
    # 1. Login
    await page.goto('http://localhost:3000')
    await page.fill('#email', 'test@example.com')
    await page.fill('#password', 'testpass')
    await page.click('#login-button')
    
    # 2. Select channel
    await page.click('.channel-card:first-child')
    
    # 3. Generate video
    await page.click('#generate-video')
    await page.fill('#topic', 'Test Topic')
    await page.click('#submit')
    
    # 4. Wait for completion
    await page.wait_for_selector('.video-complete', timeout=600000)
    
    # 5. Verify result
    assert await page.is_visible('.youtube-link')
```