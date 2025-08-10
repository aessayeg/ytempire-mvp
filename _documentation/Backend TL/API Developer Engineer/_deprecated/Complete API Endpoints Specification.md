# Complete API Endpoints Specification

**Document Version**: 1.0  
**Date**: January 2025  
**For**: API Development Engineer  
**Classification**: API Reference

---

## 📑 Table of Contents

1. [Authentication Endpoints](#authentication-endpoints)
2. [Channel Management Endpoints](#channel-management-endpoints)
3. [Video Generation Endpoints](#video-generation-endpoints)
4. [Analytics Endpoints](#analytics-endpoints)
5. [Cost Management Endpoints](#cost-management-endpoints)
6. [Webhook Endpoints](#webhook-endpoints)
7. [N8N Integration Endpoints](#n8n-integration-endpoints)

---

## Authentication Endpoints

### POST /api/v1/auth/register
**Description**: Register a new user account

```python
# Request Body
{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe",
    "tier": "starter"  # Optional: free, starter, growth, scale
}

# Response (201 Created)
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "full_name": "John Doe",
    "tier": "starter",
    "channel_limit": 5,
    "created_at": "2025-01-15T10:00:00Z"
}

# Implementation
@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    # Check if email exists
    existing = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, "Email already registered")
    
    # Create user
    user = User(
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        tier=user_data.tier or UserTier.FREE
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user
```

### POST /api/v1/auth/login
**Description**: Authenticate user and receive tokens

```python
# Request Body (OAuth2 Form)
{
    "username": "user@example.com",  # Email as username
    "password": "SecurePass123!"
}

# Response (200 OK)
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 1800
}

# Implementation
@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    # Authenticate user
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    
    # Create tokens
    access_token = create_access_token({"sub": str(user.id), "role": user.role})
    refresh_token = create_refresh_token({"sub": str(user.id)})
    
    # Store refresh token
    session = Session(
        user_id=user.id,
        refresh_token_hash=get_password_hash(refresh_token),
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(session)
    await db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }
```

### POST /api/v1/auth/refresh
**Description**: Refresh access token using refresh token

```python
# Request Body
{
    "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}

# Response (200 OK)
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 1800
}
```

### POST /api/v1/auth/logout
**Description**: Invalidate refresh token

```python
# Headers
Authorization: Bearer <access_token>

# Response (200 OK)
{
    "message": "Successfully logged out"
}
```

---

## Channel Management Endpoints

### GET /api/v1/channels
**Description**: List user's channels with pagination and filtering

```python
# Query Parameters
?status=active&niche=education&page=1&limit=10&sort=created_at:desc

# Response (200 OK)
{
    "items": [
        {
            "id": "channel-uuid",
            "name": "Tech Tutorials",
            "niche": "education",
            "status": "active",
            "youtube_channel_id": "UC...",
            "automation_enabled": true,
            "daily_video_limit": 5,
            "videos_today": 2,
            "total_videos": 150,
            "created_at": "2025-01-01T00:00:00Z"
        }
    ],
    "total": 5,
    "page": 1,
    "pages": 1,
    "limit": 10
}

# Implementation
@router.get("/", response_model=PaginatedResponse[ChannelResponse])
async def list_channels(
    status: Optional[str] = None,
    niche: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    sort: str = "created_at:desc",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Build query
    query = select(Channel).where(Channel.user_id == current_user.id)
    
    if status:
        query = query.where(Channel.status == status)
    if niche:
        query = query.where(Channel.niche == niche)
    
    # Apply sorting
    sort_field, sort_order = sort.split(":")
    if sort_order == "desc":
        query = query.order_by(desc(getattr(Channel, sort_field)))
    else:
        query = query.order_by(getattr(Channel, sort_field))
    
    # Paginate
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)
    
    # Execute
    result = await db.execute(query)
    channels = result.scalars().all()
    
    # Get total count
    count_query = select(func.count()).select_from(Channel).where(Channel.user_id == current_user.id)
    if status:
        count_query = count_query.where(Channel.status == status)
    if niche:
        count_query = count_query.where(Channel.niche == niche)
    
    total = await db.scalar(count_query)
    
    return {
        "items": channels,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,
        "limit": limit
    }
```

### POST /api/v1/channels
**Description**: Create a new channel

```python
# Request Body
{
    "name": "Tech Tutorials",
    "niche": "education",
    "settings": {
        "upload_time": "14:00",
        "thumbnail_style": "modern",
        "description_template": "custom"
    }
}

# Response (201 Created)
{
    "id": "channel-uuid",
    "name": "Tech Tutorials",
    "niche": "education",
    "status": "pending_oauth",
    "youtube_channel_id": null,
    "automation_enabled": false,
    "settings": {...},
    "created_at": "2025-01-15T10:00:00Z"
}

# Implementation with quota check
@router.post("/", response_model=ChannelResponse, status_code=201)
async def create_channel(
    channel_data: ChannelCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Check channel limit
    count = await db.scalar(
        select(func.count()).select_from(Channel)
        .where(Channel.user_id == current_user.id)
    )
    
    if count >= current_user.channel_limit:
        raise HTTPException(
            403,
            f"Channel limit reached ({current_user.channel_limit}). Upgrade your plan for more channels."
        )
    
    # Create channel
    channel = Channel(
        user_id=current_user.id,
        name=channel_data.name,
        niche=channel_data.niche,
        settings=channel_data.settings or {},
        status="pending_oauth"
    )
    
    db.add(channel)
    await db.commit()
    await db.refresh(channel)
    
    # Trigger OAuth flow
    await trigger_youtube_oauth(channel.id)
    
    return channel
```

### GET /api/v1/channels/{channel_id}
**Description**: Get channel details

```python
# Response (200 OK)
{
    "id": "channel-uuid",
    "name": "Tech Tutorials",
    "niche": "education",
    "status": "active",
    "youtube_channel_id": "UC...",
    "youtube_account_id": "account-uuid",
    "automation_enabled": true,
    "daily_video_limit": 5,
    "settings": {
        "upload_time": "14:00",
        "thumbnail_style": "modern"
    },
    "statistics": {
        "total_videos": 150,
        "videos_today": 2,
        "videos_this_week": 14,
        "total_views": 50000,
        "total_revenue": 1250.50
    },
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-15T10:00:00Z"
}
```

### PATCH /api/v1/channels/{channel_id}
**Description**: Update channel settings

```python
# Request Body
{
    "name": "Updated Channel Name",
    "automation_enabled": true,
    "daily_video_limit": 3,
    "settings": {
        "upload_time": "18:00"
    }
}

# Response (200 OK)
{
    "id": "channel-uuid",
    "name": "Updated Channel Name",
    ...
}
```

### DELETE /api/v1/channels/{channel_id}
**Description**: Delete a channel (soft delete)

```python
# Response (200 OK)
{
    "message": "Channel deleted successfully",
    "channel_id": "channel-uuid"
}
```

### POST /api/v1/channels/{channel_id}/youtube-auth
**Description**: Initiate YouTube OAuth for channel

```python
# Response (200 OK)
{
    "auth_url": "https://accounts.google.com/o/oauth2/auth?...",
    "state": "state-token"
}
```

---

## Video Generation Endpoints

### POST /api/v1/videos/generate
**Description**: Queue video for generation

```python
# Request Body
{
    "channel_id": "channel-uuid",
    "topic": "10 Python Tips for Beginners",
    "style": "educational",
    "length_minutes": 8,
    "priority": 5,
    "optimization_level": "standard",  # economy, standard, premium
    "scheduled_publish_time": "2025-01-16T14:00:00Z"  # Optional
}

# Response (202 Accepted)
{
    "job_id": "job-uuid",
    "video_id": "video-uuid",
    "status": "queued",
    "queue_position": 5,
    "estimated_completion": "2025-01-15T10:15:00Z",
    "estimated_cost": 2.50,
    "webhook_url": "https://api.ytempire.com/api/v1/webhooks/video/job-uuid"
}

# Implementation with cost estimation
@router.post("/generate", response_model=VideoJobResponse, status_code=202)
async def generate_video(
    request: VideoGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    cost_tracker: CostTracker = Depends(get_cost_tracker)
):
    # Verify channel ownership
    channel = await db.get(Channel, request.channel_id)
    if not channel or channel.user_id != current_user.id:
        raise HTTPException(404, "Channel not found")
    
    # Check daily limit
    today_count = await db.scalar(
        select(func.count()).select_from(Video)
        .where(
            Video.channel_id == request.channel_id,
            Video.created_at >= datetime.utcnow().date()
        )
    )
    
    if today_count >= channel.daily_video_limit:
        raise HTTPException(
            429,
            f"Daily video limit reached ({channel.daily_video_limit})"
        )
    
    # Estimate cost
    estimated_cost = await cost_tracker.estimate_video_cost(
        request.optimization_level,
        request.length_minutes
    )
    
    # Check if within budget
    if estimated_cost > settings.COST_PER_VIDEO_HARD_LIMIT:
        raise HTTPException(
            402,
            f"Estimated cost ${estimated_cost:.2f} exceeds limit ${settings.COST_PER_VIDEO_HARD_LIMIT}"
        )
    
    # Create video record
    video = Video(
        channel_id=request.channel_id,
        title=f"Draft: {request.topic}",
        status="queued",
        metadata={
            "topic": request.topic,
            "style": request.style,
            "length_minutes": request.length_minutes,
            "optimization_level": request.optimization_level
        }
    )
    
    db.add(video)
    await db.commit()
    
    # Queue for processing
    job_id = await queue_video_generation(
        video.id,
        request.dict(),
        priority=request.priority
    )
    
    # Get queue position
    queue_position = await get_queue_position(job_id)
    
    return {
        "job_id": job_id,
        "video_id": video.id,
        "status": "queued",
        "queue_position": queue_position,
        "estimated_completion": calculate_completion_time(queue_position),
        "estimated_cost": estimated_cost,
        "webhook_url": f"{settings.API_BASE_URL}/api/v1/webhooks/video/{job_id}"
    }
```

### GET /api/v1/videos
**Description**: List videos with filtering

```python
# Query Parameters
?channel_id=uuid&status=completed&date_from=2025-01-01&date_to=2025-01-31

# Response (200 OK)
{
    "items": [
        {
            "id": "video-uuid",
            "channel_id": "channel-uuid",
            "title": "10 Python Tips for Beginners",
            "status": "published",
            "youtube_video_id": "dQw4w9WgXcQ",
            "cost": 2.35,
            "quality_score": 0.92,
            "generation_time_seconds": 485,
            "views": 1250,
            "created_at": "2025-01-15T09:00:00Z",
            "published_at": "2025-01-15T14:00:00Z"
        }
    ],
    "total": 150,
    "page": 1,
    "pages": 15,
    "aggregates": {
        "total_cost": 352.50,
        "average_cost": 2.35,
        "total_views": 187500
    }
}
```

### GET /api/v1/videos/{video_id}
**Description**: Get video details

```python
# Response (200 OK)
{
    "id": "video-uuid",
    "channel_id": "channel-uuid",
    "title": "10 Python Tips for Beginners",
    "description": "Full description...",
    "script": "Full script text...",
    "status": "published",
    "youtube_video_id": "dQw4w9WgXcQ",
    "youtube_url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "cost_breakdown": {
        "script": 0.35,
        "audio": 0.20,
        "video": 0.30,
        "thumbnail": 0.10,
        "total": 0.95
    },
    "quality_metrics": {
        "script_score": 0.95,
        "audio_quality": 0.92,
        "video_quality": 0.88,
        "overall": 0.92
    },
    "generation_time_seconds": 485,
    "file_paths": {
        "video": "/storage/videos/video-uuid.mp4",
        "thumbnail": "/storage/thumbnails/video-uuid.jpg",
        "audio": "/storage/audio/video-uuid.mp3"
    },
    "metadata": {
        "topic": "10 Python Tips for Beginners",
        "style": "educational",
        "optimization_level": "standard"
    },
    "tags": ["python", "programming", "tutorial"],
    "created_at": "2025-01-15T09:00:00Z",
    "published_at": "2025-01-15T14:00:00Z"
}
```

### GET /api/v1/videos/{video_id}/status
**Description**: Get video generation status

```python
# Response (200 OK)
{
    "video_id": "video-uuid",
    "status": "processing",
    "stage": "audio_synthesis",
    "progress": 45,
    "stages_completed": [
        {
            "stage": "script_generation",
            "completed_at": "2025-01-15T09:05:00Z",
            "duration_seconds": 25,
            "cost": 0.35
        }
    ],
    "current_stage": {
        "stage": "audio_synthesis",
        "started_at": "2025-01-15T09:05:30Z",
        "progress": 60,
        "estimated_remaining_seconds": 20
    },
    "estimated_completion": "2025-01-15T09:12:00Z",
    "cost_so_far": 0.55
}
```

### POST /api/v1/videos/{video_id}/retry
**Description**: Retry failed video generation

```python
# Request Body
{
    "optimization_level": "economy",  # Optional: reduce cost on retry
    "skip_to_stage": "audio_synthesis"  # Optional: skip completed stages
}

# Response (202 Accepted)
{
    "job_id": "new-job-uuid",
    "message": "Video generation retry queued",
    "estimated_completion": "2025-01-15T10:30:00Z"
}
```

### DELETE /api/v1/videos/{video_id}
**Description**: Cancel video generation or delete video

```python
# Response (200 OK)
{
    "message": "Video deleted successfully",
    "refunded_cost": 0.35  # If generation was incomplete
}
```

---

## Analytics Endpoints

### GET /api/v1/analytics/summary
**Description**: Get analytics summary for user

```python
# Query Parameters
?period=30d  # 7d, 30d, 90d, all

# Response (200 OK)
{
    "period": "30d",
    "date_range": {
        "from": "2024-12-16",
        "to": "2025-01-15"
    },
    "video_metrics": {
        "total_generated": 150,
        "total_published": 145,
        "success_rate": 0.97,
        "average_generation_time": 485,
        "total_cost": 352.50,
        "average_cost": 2.35
    },
    "channel_metrics": {
        "active_channels": 5,
        "total_views": 187500,
        "total_watch_time_hours": 3125,
        "subscribers_gained": 850,
        "revenue_generated": 1250.50
    },
    "cost_metrics": {
        "total_spent": 352.50,
        "by_service": {
            "openai": 125.50,
            "tts": 85.25,
            "storage": 15.75,
            "other": 126.00
        },
        "cost_per_view": 0.0019,
        "roi": 3.55
    },
    "top_performing": {
        "videos": [
            {
                "id": "video-uuid",
                "title": "Best Python Tips",
                "views": 15000,
                "revenue": 125.50
            }
        ],
        "channels": [
            {
                "id": "channel-uuid",
                "name": "Tech Tutorials",
                "total_views": 75000,
                "revenue": 625.25
            }
        ]
    }
}
```

### GET /api/v1/analytics/channels/{channel_id}
**Description**: Get detailed analytics for a channel

```python
# Query Parameters
?period=30d&metrics=views,revenue,subscribers

# Response (200 OK)
{
    "channel_id": "channel-uuid",
    "channel_name": "Tech Tutorials",
    "period": "30d",
    "metrics": {
        "views": {
            "total": 75000,
            "daily_average": 2500,
            "trend": "+15%",
            "chart_data": [
                {"date": "2024-12-16", "value": 2100},
                {"date": "2024-12-17", "value": 2350}
                // ... 30 data points
            ]
        },
        "revenue": {
            "total": 625.25,
            "daily_average": 20.84,
            "trend": "+22%",
            "chart_data": [...]
        },
        "subscribers": {
            "gained": 425,
            "lost": 28,
            "net": 397,
            "total": 5420,
            "chart_data": [...]
        }
    },
    "video_performance": {
        "published": 30,
        "average_views": 2500,
        "average_retention": 0.65,
        "best_performing": {...},
        "worst_performing": {...}
    }
}
```

### GET /api/v1/analytics/videos/{video_id}
**Description**: Get detailed analytics for a video

```python
# Response (200 OK)
{
    "video_id": "video-uuid",
    "title": "10 Python Tips",
    "metrics": {
        "views": 15000,
        "likes": 850,
        "dislikes": 23,
        "comments": 125,
        "shares": 45,
        "watch_time_hours": 250,
        "average_view_duration": 240,
        "retention_rate": 0.68,
        "ctr": 0.082
    },
    "revenue": {
        "ad_revenue": 125.50,
        "affiliate_revenue": 35.25,
        "total": 160.75
    },
    "audience": {
        "demographics": {
            "age_groups": {...},
            "geography": {...}
        },
        "traffic_sources": {
            "search": 0.45,
            "suggested": 0.30,
            "external": 0.15,
            "other": 0.10
        }
    },
    "cost_analysis": {
        "generation_cost": 2.35,
        "revenue": 160.75,
        "profit": 158.40,
        "roi": 67.4
    }
}
```

### POST /api/v1/analytics/export
**Description**: Export analytics data

```python
# Request Body
{
    "type": "channels",  # channels, videos, costs
    "format": "csv",  # csv, json, excel
    "period": "30d",
    "filters": {
        "channel_ids": ["channel-uuid"],
        "status": "published"
    }
}

# Response (202 Accepted)
{
    "export_id": "export-uuid",
    "status": "processing",
    "estimated_completion": "2025-01-15T10:05:00Z",
    "download_url": null  # Will be populated when ready
}
```

---

## Cost Management Endpoints

### GET /api/v1/costs/current
**Description**: Get current cost metrics

```python
# Response (200 OK)
{
    "daily": {
        "date": "2025-01-15",
        "spent": 47.85,
        "budget": 150.00,
        "remaining": 102.15,
        "percentage_used": 31.9,
        "videos_generated": 20,
        "average_cost": 2.39
    },
    "monthly": {
        "month": "2025-01",
        "spent": 523.45,
        "budget": 4500.00,
        "remaining": 3976.55,
        "percentage_used": 11.6,
        "projection": 1105.95  # Projected month-end
    },
    "alerts": [
        {
            "level": "info",
            "message": "Daily spending on track",
            "timestamp": "2025-01-15T10:00:00Z"
        }
    ]
}
```

### GET /api/v1/costs/breakdown
**Description**: Get detailed cost breakdown

```python
# Query Parameters
?period=7d&group_by=service

# Response (200 OK)
{
    "period": "7d",
    "total": 245.75,
    "breakdown": {
        "by_service": {
            "openai": {
                "amount": 95.50,
                "percentage": 38.9,
                "count": 140
            },
            "google_tts": {
                "amount": 42.25,
                "percentage": 17.2,
                "count": 140
            },
            "storage": {
                "amount": 8.00,
                "percentage": 3.3,
                "count": 140
            }
        },
        "by_channel": {
            "channel-uuid-1": {
                "name": "Tech Tutorials",
                "amount": 125.50,
                "videos": 35
            }
        },
        "by_optimization_level": {
            "economy": {
                "amount": 45.25,
                "videos": 25,
                "average": 1.81
            },
            "standard": {
                "amount": 150.50,
                "videos": 65,
                "average": 2.32
            },
            "premium": {
                "amount": 50.00,
                "videos": 10,
                "average": 5.00
            }
        }
    }
}
```

### POST /api/v1/costs/set-budget
**Description**: Set cost budget limits

```python
# Request Body
{
    "daily_limit": 200.00,
    "monthly_limit": 5000.00,
    "per_video_limit": 3.50,
    "alert_thresholds": {
        "warning": 0.80,  # 80% of limit
        "critical": 0.95  # 95% of limit
    }
}

# Response (200 OK)
{
    "message": "Budget limits updated successfully",
    "limits": {
        "daily": 200.00,
        "monthly": 5000.00,
        "per_video": 3.50
    }
}
```

### GET /api/v1/costs/optimization-suggestions
**Description**: Get cost optimization suggestions

```python
# Response (200 OK)
{
    "current_average_cost": 2.45,
    "potential_savings": 0.65,
    "suggestions": [
        {
            "action": "Use GPT-3.5 instead of GPT-4 for simple videos",
            "potential_savings": 0.30,
            "impact": "low",
            "implementation": "automatic"
        },
        {
            "action": "Enable audio caching for common phrases",
            "potential_savings": 0.15,
            "impact": "none",
            "implementation": "automatic"
        },
        {
            "action": "Batch process videos during off-peak hours",
            "potential_savings": 0.20,
            "impact": "medium",
            "implementation": "manual"
        }
    ],
    "recommended_settings": {
        "default_optimization_level": "economy",
        "batch_processing": true,
        "cache_enabled": true,
        "smart_scheduling": true
    }
}
```

---

## Webhook Endpoints

### POST /api/v1/webhooks/stripe
**Description**: Handle Stripe webhook events

```python
# Headers
Stripe-Signature: t=1492774577,v1=5257a869...

# Request Body (Stripe Event)
{
    "id": "evt_1234",
    "object": "event",
    "type": "invoice.payment_succeeded",
    "data": {
        "object": {
            "id": "in_1234",
            "customer": "cus_1234",
            "amount_paid": 29700
        }
    }
}

# Response (200 OK)
{
    "received": true
}

# Implementation
@router.post("/stripe")
async def handle_stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    # Verify signature
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(400, "Invalid signature")
    
    # Handle event
    if event["type"] == "invoice.payment_succeeded":
        await handle_payment_success(event["data"]["object"], db)
    elif event["type"] == "customer.subscription.deleted":
        await handle_subscription_cancelled(event["data"]["object"], db)
    
    return {"received": True}
```

### POST /api/v1/webhooks/youtube
**Description**: Handle YouTube API callbacks

```python
# Request Body
{
    "channel_id": "UC...",
    "event": "video.processing.complete",
    "video_id": "dQw4w9WgXcQ",
    "status": "success"
}

# Response (200 OK)
{
    "acknowledged": true
}
```

### POST /api/v1/webhooks/n8n/{workflow_id}
**Description**: N8N workflow completion callback

```python
# Request Body
{
    "workflow_id": "workflow-uuid",
    "execution_id": "exec-uuid",
    "status": "success",
    "data": {
        "video_id": "video-uuid",
        "stage": "completed",
        "cost": 2.35,
        "outputs": {
            "script": "s3://bucket/script.txt",
            "audio": "s3://bucket/audio.mp3",
            "video": "s3://bucket/video.mp4"
        }
    }
}

# Response (200 OK)
{
    "received": true,
    "next_action": "upload_to_youtube"
}
```

---

## N8N Integration Endpoints

### POST /api/v1/n8n/trigger-workflow
**Description**: Trigger N8N workflow from API

```python
# Request Body
{
    "workflow_name": "video_generation",
    "data": {
        "video_id": "video-uuid",
        "topic": "Python Tutorial",
        "style": "educational",
        "optimization_level": "standard"
    },
    "callback_url": "https://api.ytempire.com/api/v1/webhooks/n8n/callback"
}

# Response (200 OK)
{
    "workflow_id": "workflow-uuid",
    "execution_id": "exec-uuid",
    "status": "started",
    "webhook_url": "http://localhost:5678/webhook/workflow-uuid"
}
```

### GET /api/v1/n8n/workflow-status/{execution_id}
**Description**: Get N8N workflow execution status

```python
# Response (200 OK)
{
    "execution_id": "exec-uuid",
    "workflow_id": "workflow-uuid",
    "status": "running",
    "progress": 65,
    "current_node": "OpenAI Script Generation",
    "completed_nodes": [
        "Start",
        "Validate Input",
        "Check Cost"
    ],
    "data": {
        "cost_so_far": 0.35
    }
}
```

### POST /api/v1/n8n/track-cost
**Description**: N8N custom node for cost tracking

```python
# Request Body
{
    "video_id": "video-uuid",
    "service": "openai",
    "operation": "script_generation",
    "amount": 0.35
}

# Response (200 OK)
{
    "total_cost": 0.35,
    "within_budget": true,
    "warning": null,
    "continue": true
}
```

### POST /api/v1/n8n/check-quota
**Description**: N8N custom node for YouTube quota check

```python
# Request Body
{
    "operation": "video_upload",
    "account_id": "account-uuid"
}

# Response (200 OK)
{
    "available": true,
    "remaining_quota": 8400,
    "daily_uploads": 2,
    "next_best_account": "account-uuid-2"
}
```

---

## Standard Response Formats

### Success Response
```json
{
    "data": {},  // Actual response data
    "meta": {
        "request_id": "req-uuid",
        "timestamp": "2025-01-15T10:00:00Z",
        "version": "1.0.0"
    }
}
```

### Error Response
```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input data",
        "details": {
            "field": "email",
            "reason": "Invalid email format"
        },
        "request_id": "req-uuid"
    }
}
```

### Paginated Response
```json
{
    "items": [],
    "total": 100,
    "page": 1,
    "pages": 10,
    "limit": 10,
    "has_next": true,
    "has_prev": false
}
```

---

## Rate Limiting Headers

All API responses include rate limiting headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642248000
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
X-Process-Time: 0.125
```

---

## API Versioning

The API uses URL path versioning. Current version: `v1`

```
https://api.ytempire.com/api/v1/...
```

Future versions will be available at:
```
https://api.ytempire.com/api/v2/...
```

---

This completes the comprehensive API endpoints specification. Each endpoint includes request/response examples and implementation guidance for the API Development Engineer.