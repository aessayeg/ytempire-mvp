# YTEMPIRE N8N Workflow Integration Guide
**Version 1.0 | January 2025**  
**For: API Development Engineer**  
**Document Type: Workflow Automation Guide**

---

## 1. N8N Integration Overview

### 1.1 N8N Architecture in YTEMPIRE
```yaml
n8n_configuration:
  deployment: Docker container (local)
  port: 5678
  authentication: Basic Auth
  database: SQLite (embedded)
  
  integration_points:
    - Video generation pipeline
    - Content scheduling
    - Analytics collection
    - Error handling and retries
    - Notification system
    
  api_communication:
    webhook_triggers: Receive events from API
    http_requests: Call internal APIs
    queue_integration: Redis for job queuing
```

### 1.2 Workflow Categories
```yaml
workflow_types:
  content_generation:
    - Script generation workflow
    - Voice synthesis workflow
    - Video assembly workflow
    - Thumbnail creation workflow
    
  publishing:
    - YouTube upload workflow
    - Scheduling workflow
    - Cross-platform distribution
    
  analytics:
    - Data collection workflow
    - Report generation workflow
    - Alert workflow
    
  maintenance:
    - Cleanup workflow
    - Backup workflow
    - Error recovery workflow
```

---

## 2. Core N8N Workflows

### 2.1 Master Video Generation Workflow
```json
{
  "name": "Master Video Generation Workflow",
  "nodes": [
    {
      "parameters": {
        "path": "/webhook/video-generation",
        "responseMode": "onReceived",
        "options": {}
      },
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/videos/{{$json.video_id}}/validate",
        "authentication": "genericCredentialType",
        "genericAuthType": "httpHeaderAuth",
        "options": {}
      },
      "name": "Validate Request",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "value1": "={{$json.valid}}",
              "value2": true
            }
          ]
        }
      },
      "name": "Check Validation",
      "type": "n8n-nodes-base.if",
      "position": [650, 300]
    },
    {
      "parameters": {
        "functionCode": "// Prepare script generation request\nconst videoData = items[0].json;\n\nreturn [{\n  json: {\n    video_id: videoData.video_id,\n    topic: videoData.topic,\n    length: videoData.video_length_seconds,\n    style: videoData.style,\n    niche: videoData.niche,\n    user_id: videoData.user_id\n  }\n}];"
      },
      "name": "Prepare Script Request",
      "type": "n8n-nodes-base.function",
      "position": [850, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/ai/generate-script",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "video_id",
              "value": "={{$json.video_id}}"
            },
            {
              "name": "topic",
              "value": "={{$json.topic}}"
            },
            {
              "name": "length_seconds",
              "value": "={{$json.length}}"
            }
          ]
        },
        "options": {
          "timeout": 30000
        }
      },
      "name": "Generate Script",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1050, 200]
    },
    {
      "parameters": {
        "functionCode": "// Check script generation cost\nconst cost = items[0].json.cost;\nconst limit = 0.50; // $0.50 limit for script\n\nif (cost > limit) {\n  throw new Error(`Script cost $${cost} exceeds limit $${limit}`);\n}\n\nreturn items;"
      },
      "name": "Check Script Cost",
      "type": "n8n-nodes-base.function",
      "position": [1250, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/voice/synthesize",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "video_id",
              "value": "={{$json.video_id}}"
            },
            {
              "name": "script",
              "value": "={{$json.script}}"
            },
            {
              "name": "voice_profile",
              "value": "={{$json.voice_profile}}"
            }
          ]
        }
      },
      "name": "Generate Voice",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1450, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/video/process",
        "method": "POST",
        "sendBody": true,
        "bodyParameters": {
          "video_id": "={{$json.video_id}}",
          "script_sections": "={{$json.script_sections}}",
          "audio_path": "={{$json.audio_path}}",
          "style_config": "={{$json.style_config}}"
        }
      },
      "name": "Process Video",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1650, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/video/thumbnail",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "video_id",
              "value": "={{$json.video_id}}"
            },
            {
              "name": "thumbnail_text",
              "value": "={{$json.thumbnail_text}}"
            }
          ]
        }
      },
      "name": "Generate Thumbnail",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1850, 200]
    },
    {
      "parameters": {
        "values": {
          "string": [
            {
              "name": "status",
              "value": "ready_to_publish"
            }
          ]
        }
      },
      "name": "Set Success Status",
      "type": "n8n-nodes-base.set",
      "position": [2050, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/videos/{{$json.video_id}}/status",
        "method": "PUT",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "status",
              "value": "={{$json.status}}"
            },
            {
              "name": "video_path",
              "value": "={{$json.video_path}}"
            },
            {
              "name": "thumbnail_path",
              "value": "={{$json.thumbnail_path}}"
            }
          ]
        }
      },
      "name": "Update Video Status",
      "type": "n8n-nodes-base.httpRequest",
      "position": [2250, 200]
    },
    {
      "parameters": {
        "functionCode": "// Error handling\nconst error = items[0].json.error || 'Unknown error';\nconst videoId = items[0].json.video_id;\n\n// Log error\nconsole.error(`Video generation failed for ${videoId}: ${error}`);\n\nreturn [{\n  json: {\n    video_id: videoId,\n    status: 'failed',\n    error: error,\n    timestamp: new Date().toISOString()\n  }\n}];"
      },
      "name": "Handle Error",
      "type": "n8n-nodes-base.function",
      "position": [850, 400]
    }
  ],
  "connections": {
    "Webhook Trigger": {
      "main": [[{"node": "Validate Request", "type": "main", "index": 0}]]
    },
    "Validate Request": {
      "main": [[{"node": "Check Validation", "type": "main", "index": 0}]]
    },
    "Check Validation": {
      "main": [
        [{"node": "Prepare Script Request", "type": "main", "index": 0}],
        [{"node": "Handle Error", "type": "main", "index": 0}]
      ]
    },
    "Prepare Script Request": {
      "main": [[{"node": "Generate Script", "type": "main", "index": 0}]]
    },
    "Generate Script": {
      "main": [[{"node": "Check Script Cost", "type": "main", "index": 0}]]
    },
    "Check Script Cost": {
      "main": [[{"node": "Generate Voice", "type": "main", "index": 0}]]
    },
    "Generate Voice": {
      "main": [[{"node": "Process Video", "type": "main", "index": 0}]]
    },
    "Process Video": {
      "main": [[{"node": "Generate Thumbnail", "type": "main", "index": 0}]]
    },
    "Generate Thumbnail": {
      "main": [[{"node": "Set Success Status", "type": "main", "index": 0}]]
    },
    "Set Success Status": {
      "main": [[{"node": "Update Video Status", "type": "main", "index": 0}]]
    }
  }
}
```

### 2.2 YouTube Publishing Workflow
```json
{
  "name": "YouTube Publishing Workflow",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "minutes",
              "minutesInterval": 30
            }
          ]
        }
      },
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/videos/ready-to-publish",
        "method": "GET",
        "authentication": "genericCredentialType",
        "genericAuthType": "httpHeaderAuth"
      },
      "name": "Get Videos to Publish",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "functionCode": "// Sort videos by channel and priority\nconst videos = items[0].json.videos || [];\n\n// Group by channel\nconst videosByChannel = videos.reduce((acc, video) => {\n  if (!acc[video.channel_id]) {\n    acc[video.channel_id] = [];\n  }\n  acc[video.channel_id].push(video);\n  return acc;\n}, {});\n\n// Create items for each video\nconst outputItems = [];\nfor (const [channelId, channelVideos] of Object.entries(videosByChannel)) {\n  // Respect rate limits - 1 video per channel per run\n  if (channelVideos.length > 0) {\n    outputItems.push({\n      json: channelVideos[0]\n    });\n  }\n}\n\nreturn outputItems;"
      },
      "name": "Process Publishing Queue",
      "type": "n8n-nodes-base.function",
      "position": [650, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/youtube/upload",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "video_id",
              "value": "={{$json.id}}"
            },
            {
              "name": "channel_id",
              "value": "={{$json.channel_id}}"
            },
            {
              "name": "video_path",
              "value": "={{$json.video_path}}"
            },
            {
              "name": "metadata",
              "value": "={{$json}}"
            }
          ]
        },
        "options": {
          "timeout": 300000
        }
      },
      "name": "Upload to YouTube",
      "type": "n8n-nodes-base.httpRequest",
      "position": [850, 300]
    },
    {
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "value1": "={{$json.success}}",
              "value2": true
            }
          ]
        }
      },
      "name": "Check Upload Success",
      "type": "n8n-nodes-base.if",
      "position": [1050, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/videos/{{$json.video_id}}/published",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "youtube_video_id",
              "value": "={{$json.youtube_video_id}}"
            },
            {
              "name": "video_url",
              "value": "={{$json.video_url}}"
            },
            {
              "name": "published_at",
              "value": "={{$now().toISO()}}"
            }
          ]
        }
      },
      "name": "Update Published Status",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1250, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/notifications/send",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "user_id",
              "value": "={{$json.user_id}}"
            },
            {
              "name": "type",
              "value": "video_published"
            },
            {
              "name": "data",
              "value": "={{$json}}"
            }
          ]
        }
      },
      "name": "Send Success Notification",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1450, 200]
    },
    {
      "parameters": {
        "functionCode": "// Handle upload failure\nconst error = items[0].json.error;\nconst videoId = items[0].json.video_id;\n\n// Determine if retry is needed\nconst retryableErrors = ['QUOTA_EXCEEDED', 'NETWORK_ERROR', 'TIMEOUT'];\nconst shouldRetry = retryableErrors.includes(error.code);\n\nreturn [{\n  json: {\n    video_id: videoId,\n    error: error,\n    should_retry: shouldRetry,\n    retry_after: shouldRetry ? 3600 : null // Retry after 1 hour\n  }\n}];"
      },
      "name": "Handle Upload Failure",
      "type": "n8n-nodes-base.function",
      "position": [1250, 400]
    }
  ]
}
```

### 2.3 Analytics Collection Workflow
```json
{
  "name": "Analytics Collection Workflow",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "hours",
              "hoursInterval": 6
            }
          ]
        }
      },
      "name": "Analytics Schedule",
      "type": "n8n-nodes-base.scheduleTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/channels/active",
        "method": "GET"
      },
      "name": "Get Active Channels",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "batchSize": 5,
        "options": {}
      },
      "name": "Split Into Batches",
      "type": "n8n-nodes-base.splitInBatches",
      "position": [650, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/youtube/analytics/channel/{{$json.channel_id}}",
        "method": "GET",
        "queryParametersUi": {
          "parameter": [
            {
              "name": "start_date",
              "value": "={{$now().minus({days: 1}).toISO()}}"
            },
            {
              "name": "end_date",
              "value": "={{$now().toISO()}}"
            }
          ]
        }
      },
      "name": "Fetch Channel Analytics",
      "type": "n8n-nodes-base.httpRequest",
      "position": [850, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/youtube/analytics/videos/{{$json.channel_id}}",
        "method": "GET",
        "queryParametersUi": {
          "parameter": [
            {
              "name": "start_date",
              "value": "={{$now().minus({days: 1}).toISO()}}"
            },
            {
              "name": "end_date",
              "value": "={{$now().toISO()}}"
            }
          ]
        }
      },
      "name": "Fetch Video Analytics",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1050, 300]
    },
    {
      "parameters": {
        "functionCode": "// Process and aggregate analytics data\nconst channelData = items[0].json.channel_analytics;\nconst videoData = items[0].json.video_analytics;\n\n// Calculate key metrics\nconst totalViews = videoData.reduce((sum, video) => sum + video.views, 0);\nconst totalRevenue = videoData.reduce((sum, video) => sum + (video.estimated_revenue || 0), 0);\nconst avgEngagement = videoData.reduce((sum, video) => sum + video.engagement_rate, 0) / videoData.length;\n\n// Identify top performing videos\nconst topVideos = videoData\n  .sort((a, b) => b.views - a.views)\n  .slice(0, 5);\n\nreturn [{\n  json: {\n    channel_id: items[0].json.channel_id,\n    date: new Date().toISOString(),\n    metrics: {\n      total_views: totalViews,\n      total_revenue: totalRevenue,\n      avg_engagement: avgEngagement,\n      subscriber_change: channelData.subscriber_change,\n      top_videos: topVideos\n    }\n  }\n}];"
      },
      "name": "Process Analytics",
      "type": "n8n-nodes-base.function",
      "position": [1250, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/analytics/store",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "analytics_data",
              "value": "={{$json}}"
            }
          ]
        }
      },
      "name": "Store Analytics",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1450, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [
            {
              "value1": "={{$json.metrics.total_revenue}}",
              "operation": "largerEqual",
              "value2": 100
            }
          ]
        }
      },
      "name": "Check Revenue Milestone",
      "type": "n8n-nodes-base.if",
      "position": [1650, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/notifications/milestone",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "type",
              "value": "revenue_milestone"
            },
            {
              "name": "channel_id",
              "value": "={{$json.channel_id}}"
            },
            {
              "name": "revenue",
              "value": "={{$json.metrics.total_revenue}}"
            }
          ]
        }
      },
      "name": "Send Milestone Alert",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1850, 200]
    }
  ]
}
```

---

## 3. API Integration with N8N

### 3.1 N8N Webhook Endpoints
```python
# app/api/v1/n8n_integration.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
import httpx
import uuid

router = APIRouter()

class N8NIntegration:
    """Handle N8N workflow triggers and callbacks"""
    
    def __init__(self):
        self.n8n_url = "http://n8n:5678"
        self.webhook_base = f"{self.n8n_url}/webhook"
        self.api_base = f"{self.n8n_url}/api/v1"
        
    async def trigger_workflow(
        self,
        workflow_name: str,
        data: Dict[str, Any]
    ) -> Dict:
        """Trigger N8N workflow via webhook"""
        
        webhook_url = f"{self.webhook_base}/{workflow_name}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    webhook_url,
                    json=data,
                    headers={
                        "X-Webhook-ID": str(uuid.uuid4()),
                        "X-Timestamp": datetime.utcnow().isoformat()
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error(f"N8N workflow trigger failed: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow trigger failed: {str(e)}"
                )
    
    async def get_workflow_status(
        self,
        execution_id: str
    ) -> Dict:
        """Get workflow execution status"""
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_base}/executions/{execution_id}",
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            return response.json()
    
    def _get_auth_headers(self) -> Dict:
        """Get N8N API authentication headers"""
        
        return {
            "X-N8N-API-KEY": settings.N8N_API_KEY,
            "Content-Type": "application/json"
        }

n8n_integration = N8NIntegration()

@router.post("/trigger/video-generation")
async def trigger_video_generation(
    video_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger video generation workflow in N8N"""
    
    # Get video details
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.channel.has(user_id=current_user.id)
    ).first()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Prepare workflow data
    workflow_data = {
        "video_id": video.id,
        "channel_id": video.channel_id,
        "topic": video.topic,
        "video_length_seconds": video.video_length_seconds,
        "style": video.style,
        "niche": video.channel.niche,
        "user_id": current_user.id,
        "callback_url": f"{settings.API_URL}/api/v1/n8n/callback/video-generation"
    }
    
    # Update video status
    video.status = "processing"
    video.processing_started_at = datetime.utcnow()
    db.commit()
    
    # Trigger workflow
    result = await n8n_integration.trigger_workflow(
        "video-generation",
        workflow_data
    )
    
    # Store execution ID
    video.generation_params["n8n_execution_id"] = result.get("executionId")
    db.commit()
    
    return {
        "message": "Video generation started",
        "execution_id": result.get("executionId"),
        "status": "processing"
    }

@router.post("/callback/video-generation")
async def video_generation_callback(
    data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Handle N8N workflow completion callback"""
    
    video_id = data.get("video_id")
    status = data.get("status")
    
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if status == "success":
        video.status = "ready"
        video.video_file_path = data.get("video_path")
        video.thumbnail_path = data.get("thumbnail_path")
        video.total_cost = data.get("total_cost", 0)
        video.processing_completed_at = datetime.utcnow()
        
        # Calculate processing time
        if video.processing_started_at:
            video.generation_time_seconds = (
                video.processing_completed_at - video.processing_started_at
            ).total_seconds()
    else:
        video.status = "failed"
        video.generation_params["error"] = data.get("error", "Unknown error")
    
    db.commit()
    
    # Send notification to user
    await send_notification(
        video.channel.user_id,
        "video_generation_complete",
        {
            "video_id": video.id,
            "status": video.status,
            "title": video.title
        }
    )
    
    return {"status": "ok"}

@router.get("/workflow/status/{execution_id}")
async def get_workflow_status(
    execution_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get N8N workflow execution status"""
    
    status = await n8n_integration.get_workflow_status(execution_id)
    
    return {
        "execution_id": execution_id,
        "status": status.get("status"),
        "started_at": status.get("startedAt"),
        "finished_at": status.get("stoppedAt"),
        "data": status.get("data", {})
    }
```

### 3.2 Queue Integration
```python
# app/services/queue_service.py
import redis
import json
from typing import Dict, Any, Optional
import asyncio

class QueueService:
    """Redis-based queue for N8N integration"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        
        self.queues = {
            "video_generation": "queue:video:generation",
            "video_publishing": "queue:video:publishing",
            "analytics_collection": "queue:analytics:collection"
        }
    
    async def enqueue(
        self,
        queue_name: str,
        data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Add job to queue"""
        
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "data": data,
            "priority": priority,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        # Add to sorted set with priority as score
        queue_key = self.queues.get(queue_name)
        if not queue_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        await self.redis.zadd(
            queue_key,
            {json.dumps(job_data): -priority}  # Negative for high priority first
        )
        
        # Store job details
        await self.redis.hset(
            f"job:{job_id}",
            mapping=job_data
        )
        
        # Set expiry
        await self.redis.expire(f"job:{job_id}", 86400)  # 24 hours
        
        return job_id
    
    async def dequeue(
        self,
        queue_name: str,
        count: int = 1
    ) -> List[Dict[str, Any]]:
        """Get jobs from queue"""
        
        queue_key = self.queues.get(queue_name)
        if not queue_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        # Get highest priority jobs
        job_data_list = await self.redis.zrange(
            queue_key,
            0,
            count - 1,
            withscores=False
        )
        
        jobs = []
        for job_data_str in job_data_list:
            job_data = json.loads(job_data_str)
            jobs.append(job_data)
            
            # Remove from queue
            await self.redis.zrem(queue_key, job_data_str)
            
            # Update status
            await self.redis.hset(
                f"job:{job_data['id']}",
                "status",
                "processing"
            )
        
        return jobs
    
    async def complete_job(
        self,
        job_id: str,
        result: Dict[str, Any]
    ):
        """Mark job as completed"""
        
        await self.redis.hset(
            f"job:{job_id}",
            mapping={
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "result": json.dumps(result)
            }
        )
    
    async def fail_job(
        self,
        job_id: str,
        error: str,
        retry: bool = True
    ):
        """Mark job as failed"""
        
        job_data = await self.redis.hgetall(f"job:{job_id}")
        
        retry_count = int(job_data.get("retry_count", 0))
        
        if retry and retry_count < 3:
            # Re-queue with lower priority
            job_data["retry_count"] = retry_count + 1
            job_data["status"] = "pending"
            
            queue_name = job_data.get("queue", "video_generation")
            await self.enqueue(
                queue_name,
                json.loads(job_data["data"]),
                priority=-retry_count  # Lower priority for retries
            )
        else:
            await self.redis.hset(
                f"job:{job_id}",
                mapping={
                    "status": "failed",
                    "failed_at": datetime.utcnow().isoformat(),
                    "error": error
                }
            )
    
    async def get_queue_stats(self, queue_name: str) -> Dict:
        """Get queue statistics"""
        
        queue_key = self.queues.get(queue_name)
        if not queue_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        # Get queue length
        queue_length = await self.redis.zcard(queue_key)
        
        # Get job stats
        pattern = "job:*"
        stats = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }
        
        for key in self.redis.scan_iter(match=pattern):
            status = await self.redis.hget(key, "status")
            if status in stats:
                stats[status] += 1
        
        return {
            "queue_name": queue_name,
            "queue_length": queue_length,
            "job_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }

# Queue worker for API
async def process_queue_worker():
    """Background worker to process queues"""
    
    queue_service = QueueService()
    n8n = N8NIntegration()
    
    while True:
        try:
            # Process video generation queue
            jobs = await queue_service.dequeue("video_generation", count=5)
            
            for job in jobs:
                try:
                    # Trigger N8N workflow
                    result = await n8n.trigger_workflow(
                        "video-generation",
                        job["data"]
                    )
                    
                    await queue_service.complete_job(
                        job["id"],
                        result
                    )
                    
                except Exception as e:
                    logger.error(f"Job processing failed: {str(e)}")
                    await queue_service.fail_job(
                        job["id"],
                        str(e)
                    )
            
            # Sleep if no jobs
            if not jobs:
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"Queue worker error: {str(e)}")
            await asyncio.sleep(30)
```

---

## 4. Advanced N8N Workflows

### 4.1 Content Optimization Workflow
```json
{
  "name": "Content Optimization Workflow",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [{"field": "days", "daysInterval": 1}]
        }
      },
      "name": "Daily Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/analytics/underperforming-videos",
        "method": "GET",
        "queryParametersUi": {
          "parameter": [
            {
              "name": "days",
              "value": "7"
            },
            {
              "name": "min_views",
              "value": "100"
            }
          ]
        }
      },
      "name": "Get Underperforming Videos",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "functionCode": "// Analyze why videos underperformed\nconst videos = items[0].json.videos || [];\nconst analysisResults = [];\n\nfor (const video of videos) {\n  const analysis = {\n    video_id: video.id,\n    title: video.title,\n    views: video.views,\n    engagement_rate: video.engagement_rate,\n    issues: []\n  };\n  \n  // Check title length\n  if (video.title.length > 60) {\n    analysis.issues.push('Title too long');\n  }\n  \n  // Check thumbnail CTR\n  if (video.click_through_rate < 2) {\n    analysis.issues.push('Low thumbnail CTR');\n  }\n  \n  // Check description\n  if (!video.description || video.description.length < 100) {\n    analysis.issues.push('Short description');\n  }\n  \n  // Check tags\n  if (!video.tags || video.tags.length < 5) {\n    analysis.issues.push('Insufficient tags');\n  }\n  \n  analysisResults.push(analysis);\n}\n\nreturn analysisResults.map(a => ({json: a}));"
      },
      "name": "Analyze Performance Issues",
      "type": "n8n-nodes-base.function",
      "position": [650, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [
            {
              "value1": "={{$json.issues.length}}",
              "operation": "largerEqual",
              "value2": 1
            }
          ]
        }
      },
      "name": "Has Issues?",
      "type": "n8n-nodes-base.if",
      "position": [850, 300]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/ai/optimize-content",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "video_id",
              "value": "={{$json.video_id}}"
            },
            {
              "name": "current_title",
              "value": "={{$json.title}}"
            },
            {
              "name": "issues",
              "value": "={{$json.issues}}"
            }
          ]
        }
      },
      "name": "Generate Optimizations",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1050, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/videos/{{$json.video_id}}/optimize",
        "method": "PUT",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "new_title",
              "value": "={{$json.optimized_title}}"
            },
            {
              "name": "new_description",
              "value": "={{$json.optimized_description}}"
            },
            {
              "name": "new_tags",
              "value": "={{$json.optimized_tags}}"
            }
          ]
        }
      },
      "name": "Apply Optimizations",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1250, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/reports/optimization",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "date",
              "value": "={{$now().toISO()}}"
            },
            {
              "name": "videos_analyzed",
              "value": "={{$items().length}}"
            },
            {
              "name": "optimizations_applied",
              "value": "={{$json}}"
            }
          ]
        }
      },
      "name": "Generate Report",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1450, 300]
    }
  ]
}
```

### 4.2 Error Recovery Workflow
```json
{
  "name": "Error Recovery Workflow",
  "nodes": [
    {
      "parameters": {
        "path": "/webhook/error-recovery",
        "responseMode": "onReceived"
      },
      "name": "Error Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300]
    },
    {
      "parameters": {
        "functionCode": "// Categorize error and determine recovery strategy\nconst error = items[0].json;\nlet strategy = 'retry';\nlet delay = 300; // 5 minutes default\n\nswitch(error.type) {\n  case 'QUOTA_EXCEEDED':\n    strategy = 'delay';\n    delay = 3600; // 1 hour\n    break;\n    \n  case 'API_ERROR':\n    if (error.service === 'openai') {\n      strategy = 'fallback';\n    } else {\n      strategy = 'retry';\n      delay = 600;\n    }\n    break;\n    \n  case 'PROCESSING_FAILED':\n    strategy = 'restart';\n    break;\n    \n  case 'UPLOAD_FAILED':\n    if (error.attempts > 3) {\n      strategy = 'manual';\n    } else {\n      strategy = 'retry';\n      delay = 1800;\n    }\n    break;\n    \n  default:\n    strategy = 'alert';\n}\n\nreturn [{\n  json: {\n    ...error,\n    recovery_strategy: strategy,\n    delay_seconds: delay\n  }\n}];"
      },
      "name": "Determine Recovery Strategy",
      "type": "n8n-nodes-base.function",
      "position": [450, 300]
    },
    {
      "parameters": {
        "mode": "multipleFields",
        "options": {}
      },
      "name": "Route by Strategy",
      "type": "n8n-nodes-base.switch",
      "position": [650, 300]
    },
    {
      "parameters": {
        "value": "={{$json.delay_seconds}}",
        "unit": "seconds"
      },
      "name": "Wait",
      "type": "n8n-nodes-base.wait",
      "position": [850, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/jobs/retry",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "job_id",
              "value": "={{$json.job_id}}"
            },
            {
              "name": "attempt",
              "value": "={{$json.attempts + 1}}"
            }
          ]
        }
      },
      "name": "Retry Job",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1050, 200]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/jobs/{{$json.job_id}}/fallback",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "original_service",
              "value": "={{$json.service}}"
            },
            {
              "name": "fallback_service",
              "value": "={{$json.service === 'elevenlabs' ? 'google_tts' : 'aws'}}"
            }
          ]
        }
      },
      "name": "Use Fallback Service",
      "type": "n8n-nodes-base.httpRequest",
      "position": [850, 400]
    },
    {
      "parameters": {
        "url": "http://api:8000/api/v1/alerts/create",
        "method": "POST",
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "type",
              "value": "manual_intervention_required"
            },
            {
              "name": "severity",
              "value": "high"
            },
            {
              "name": "data",
              "value": "={{$json}}"
            }
          ]
        }
      },
      "name": "Create Manual Alert",
      "type": "n8n-nodes-base.httpRequest",
      "position": [850, 600]
    }
  ]
}
```

---

## 5. Monitoring & Management

### 5.1 N8N Workflow Monitoring
```python
# app/services/n8n_monitoring.py
from typing import Dict, List
import httpx
from datetime import datetime, timedelta

class N8NMonitoring:
    """Monitor N8N workflow health and performance"""
    
    def __init__(self):
        self.n8n_api = f"http://n8n:5678/api/v1"
        self.metrics = {
            "workflow_executions": Counter(
                'n8n_workflow_executions_total',
                'Total workflow executions',
                ['workflow_name', 'status']
            ),
            "workflow_duration": Histogram(
                'n8n_workflow_duration_seconds',
                'Workflow execution duration',
                ['workflow_name']
            ),
            "workflow_errors": Counter(
                'n8n_workflow_errors_total',
                'Total workflow errors',
                ['workflow_name', 'error_type']
            )
        }
    
    async def get_workflow_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Get workflow execution metrics"""
        
        async with httpx.AsyncClient() as client:
            # Get executions
            response = await client.get(
                f"{self.n8n_api}/executions",
                params={
                    "filter": json.dumps({
                        "startedAfter": start_date.isoformat(),
                        "startedBefore": end_date.isoformat()
                    })
                },
                headers=self._get_auth_headers()
            )
            
            executions = response.json()["data"]
            
            # Process metrics
            metrics = {
                "total_executions": len(executions),
                "successful": 0,
                "failed": 0,
                "running": 0,
                "average_duration": 0,
                "by_workflow": {}
            }
            
            total_duration = 0
            
            for execution in executions:
                workflow_name = execution["workflowData"]["name"]
                status = execution["finished"] 
                
                if status:
                    if execution["data"]["resultData"]["error"]:
                        metrics["failed"] += 1
                        self.metrics["workflow_errors"].labels(
                            workflow_name=workflow_name,
                            error_type=execution["data"]["resultData"]["error"]["message"]
                        ).inc()
                    else:
                        metrics["successful"] += 1
                else:
                    metrics["running"] += 1
                
                # Track execution
                self.metrics["workflow_executions"].labels(
                    workflow_name=workflow_name,
                    status="success" if status and not execution["data"]["resultData"]["error"] else "failed"
                ).inc()
                
                # Calculate duration
                if execution["startedAt"] and execution["stoppedAt"]:
                    duration = (
                        datetime.fromisoformat(execution["stoppedAt"]) -
                        datetime.fromisoformat(execution["startedAt"])
                    ).total_seconds()
                    
                    total_duration += duration
                    
                    self.metrics["workflow_duration"].labels(
                        workflow_name=workflow_name
                    ).observe(duration)
                    
                    # By workflow metrics
                    if workflow_name not in metrics["by_workflow"]:
                        metrics["by_workflow"][workflow_name] = {
                            "count": 0,
                            "successful": 0,
                            "failed": 0,
                            "total_duration": 0
                        }
                    
                    workflow_metrics = metrics["by_workflow"][workflow_name]
                    workflow_metrics["count"] += 1
                    workflow_metrics["total_duration"] += duration
                    
                    if status and not execution["data"]["resultData"]["error"]:
                        workflow_metrics["successful"] += 1
                    else:
                        workflow_metrics["failed"] += 1
            
            # Calculate averages
            if metrics["total_executions"] > 0:
                metrics["average_duration"] = total_duration / metrics["total_executions"]
            
            for workflow_name, workflow_data in metrics["by_workflow"].items():
                if workflow_data["count"] > 0:
                    workflow_data["average_duration"] = (
                        workflow_data["total_duration"] / workflow_data["count"]
                    )
            
            return metrics
    
    async def get_workflow_health(self) -> Dict:
        """Get overall workflow health status"""
        
        # Check last hour metrics
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=1)
        
        metrics = await self.get_workflow_metrics(start_date, end_date)
        
        # Calculate health score
        if metrics["total_executions"] == 0:
            health_score = 100
        else:
            success_rate = (metrics["successful"] / metrics["total_executions"]) * 100
            health_score = success_rate
        
        # Determine status
        if health_score >= 95:
            status = "healthy"
        elif health_score >= 80:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "health_score": health_score,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_failed_workflows(
        self,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent failed workflow executions"""
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.n8n_api}/executions",
                params={
                    "filter": json.dumps({
                        "finished": True,
                        "success": False
                    }),
                    "limit": limit
                },
                headers=self._get_auth_headers()
            )
            
            failed_executions = response.json()["data"]
            
            # Extract relevant information
            failures = []
            for execution in failed_executions:
                failure = {
                    "execution_id": execution["id"],
                    "workflow_name": execution["workflowData"]["name"],
                    "started_at": execution["startedAt"],
                    "stopped_at": execution["stoppedAt"],
                    "error": execution["data"]["resultData"]["error"],
                    "retry_of": execution.get("retryOf"),
                    "retry_count": execution.get("retrySuccessId", 0)
                }
                failures.append(failure)
            
            return failures
    
    def _get_auth_headers(self) -> Dict:
        """Get N8N API authentication headers"""
        
        return {
            "X-N8N-API-KEY": settings.N8N_API_KEY,
            "Content-Type": "application/json"
        }

# API endpoint for workflow monitoring
@router.get("/n8n/health")
async def get_n8n_health(
    current_user: User = Depends(get_current_user)
):
    """Get N8N workflow health status"""
    
    monitoring = N8NMonitoring()
    health = await monitoring.get_workflow_health()
    
    return health

@router.get("/n8n/metrics")
async def get_n8n_metrics(
    start_date: datetime = Query(default=None),
    end_date: datetime = Query(default=None),
    current_user: User = Depends(get_current_user)
):
    """Get N8N workflow metrics"""
    
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=1)
    if not end_date:
        end_date = datetime.utcnow()
    
    monitoring = N8NMonitoring()
    metrics = await monitoring.get_workflow_metrics(start_date, end_date)
    
    return metrics
```

---

## 6. Best Practices & Optimization

### 6.1 Workflow Design Principles
```yaml
workflow_best_practices:
  design:
    - Keep workflows modular and single-purpose
    - Use sub-workflows for reusable logic
    - Implement proper error handling at each step
    - Add logging nodes for debugging
    - Use environment variables for configuration
    
  performance:
    - Batch operations when possible
    - Use parallel processing for independent tasks
    - Implement caching for expensive operations
    - Set appropriate timeouts
    - Monitor memory usage
    
  reliability:
    - Add retry logic with exponential backoff
    - Implement circuit breakers for external services
    - Use health checks before processing
    - Store state for recovery
    - Log all errors with context
    
  monitoring:
    - Track execution time for each node
    - Count successes and failures
    - Alert on repeated failures
    - Monitor resource usage
    - Create dashboards for visibility
```

### 6.2 Common Workflow Patterns
```javascript
// Retry Pattern with Exponential Backoff
const retryWithBackoff = {
  maxAttempts: 3,
  baseDelay: 1000, // 1 second
  
  async execute(fn, context) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        if (attempt < this.maxAttempts) {
          const delay = this.baseDelay * Math.pow(2, attempt - 1);
          await new Promise(resolve => setTimeout(resolve, delay));
          
          console.log(`Retry attempt ${attempt} after ${delay}ms delay`);
        }
      }
    }
    
    throw lastError;
  }
};

// Batch Processing Pattern
const batchProcessor = {
  batchSize: 10,
  
  async processBatch(items, processFunc) {
    const results = [];
    
    for (let i = 0; i < items.length; i += this.batchSize) {
      const batch = items.slice(i, i + this.batchSize);
      
      const batchResults = await Promise.all(
        batch.map(item => processFunc(item))
      );
      
      results.push(...batchResults);
      
      // Add delay between batches to avoid rate limits
      if (i + this.batchSize < items.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    return results;
  }
};

// Circuit Breaker Pattern
const circuitBreaker = {
  failureThreshold: 5,
  resetTimeout: 60000, // 1 minute
  state: 'closed',
  failures: 0,
  lastFailureTime: null,
  
  async execute(fn) {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = 'half-open';
      } else {
        throw new Error('Circuit breaker is open');
      }
    }
    
    try {
      const result = await fn();
      
      if (this.state === 'half-open') {
        this.state = 'closed';
        this.failures = 0;
      }
      
      return result;
    } catch (error) {
      this.failures++;
      this.lastFailureTime = Date.now();
      
      if (this.failures >= this.failureThreshold) {
        this.state = 'open';
      }
      
      throw error;
    }
  }
};
```

### 6.3 Workflow Testing
```python
# app/tests/test_n8n_workflows.py
import pytest
from httpx import AsyncClient
import asyncio

class TestN8NWorkflows:
    """Test N8N workflow integrations"""
    
    @pytest.mark.asyncio
    async def test_video_generation_workflow(self, client: AsyncClient, test_video):
        """Test complete video generation workflow"""
        
        # Trigger workflow
        response = await client.post(
            "/api/v1/n8n/trigger/video-generation",
            json={"video_id": test_video.id},
            headers={"Authorization": f"Bearer {test_token}"}
        )
        
        assert response.status_code == 200
        execution_id = response.json()["execution_id"]
        
        # Wait for completion (with timeout)
        max_wait = 60  # seconds
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            status_response = await client.get(
                f"/api/v1/n8n/workflow/status/{execution_id}",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            
            status = status_response.json()["status"]
            
            if status in ["success", "failed"]:
                break
                
            await asyncio.sleep(2)
        
        # Verify completion
        assert status == "success"
        
        # Check video was updated
        video_response = await client.get(
            f"/api/v1/videos/{test_video.id}",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        
        video_data = video_response.json()
        assert video_data["status"] == "ready"
        assert video_data["video_path"] is not None
        assert video_data["total_cost"] < 1.0
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, client: AsyncClient):
        """Test workflow error handling"""
        
        # Trigger with invalid data
        response = await client.post(
            "/api/v1/n8n/trigger/video-generation",
            json={"video_id": -1},  # Invalid ID
            headers={"Authorization": f"Bearer {test_token}"}
        )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_workflow_monitoring(self, client: AsyncClient):
        """Test workflow monitoring endpoints"""
        
        # Get health status
        response = await client.get(
            "/api/v1/n8n/health",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        
        assert response.status_code == 200
        health = response.json()
        
        assert "status" in health
        assert "health_score" in health
        assert health["health_score"] >= 0
        assert health["health_score"] <= 100
```

---

## Summary

This comprehensive N8N integration guide provides:

1. **Core Workflows**: Complete JSON definitions for video generation, publishing, and analytics
2. **API Integration**: Full Python implementation for triggering and monitoring workflows
3. **Queue Management**: Redis-based queue system for reliable job processing
4. **Error Handling**: Robust error recovery and retry mechanisms
5. **Monitoring**: Complete monitoring solution for workflow health
6. **Best Practices**: Design patterns and optimization strategies

The N8N integration serves as the automation backbone of YTEMPIRE, orchestrating complex multi-step processes while maintaining reliability and scalability. All workflows are designed to handle failures gracefully and provide complete visibility into the system's operation.