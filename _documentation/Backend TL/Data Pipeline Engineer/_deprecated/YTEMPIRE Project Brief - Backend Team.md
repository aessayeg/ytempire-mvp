# YTEMPIRE Project Brief - Backend Team

## Final Alignment & Clarifications

### 1. Team Size Correction (FINAL)
```
CORRECT TEAM STRUCTURE (17 people total):

CTO/Technical Director
├── Backend Team Lead (3 direct reports)
│   ├── API Developer Engineer
│   ├── Data Pipeline Engineer
│   └── Integration Specialist
│
├── Frontend Team Lead (3 direct reports)
│   ├── React Engineer
│   ├── Dashboard Specialist
│   └── UI/UX Designer
│
└── Platform Ops Lead (3 direct reports)
    ├── DevOps Engineer
    ├── Security Engineer
    └── QA Engineer

VP of AI (Separate)
├── AI/ML Team Lead (1 direct report)
│   └── ML Engineer
└── Data Team Lead (2 direct reports)
    ├── Data Engineer
    └── Analytics Engineer

Total: 17 people (12 Technical + 5 AI) ✓
```

### 2. Video Production Scale (CLARIFIED)
```yaml
video_production_final:
  target_per_user: 1 video per day per user
  
  daily_production:
    videos: 50 (50 users × 1 video/user/day)
    note: "Each user's 5 channels share 1 daily video quota"
    
  weekly_production: 350 videos
  
  total_mvp_12_weeks: 4,200 videos
  
  rationale: "Cost and API constraints limit us to 1 video per user per day, rotated across their 5 channels"
```

### 3. Cost Target (STANDARDIZED)
```yaml
official_cost_target:
  maximum_allowed: $3.00 per video  # Official MVP target
  operational_target: $2.50 per video  # What we aim for
  buffer: $0.50  # Safety margin
  
  breakdown:
    openai: $1.00
    elevenlabs: $0.50
    thumbnail: $0.20
    processing: $0.30
    safety_margin: $1.00
    total: $3.00  # Never exceed this
```

### 4. Performance Targets (CLARIFIED)
```python
class PerformanceMetrics:
    """ALL performance targets are for INTERNAL PROCESSING ONLY"""
    
    # What we measure and guarantee
    INTERNAL_API_P50 = 100  # ms (our code only)
    INTERNAL_API_P95 = 500  # ms (our code only)
    
    # Reality check (not part of SLA)
    EXTERNAL_API_REALITY = {
        'openai_gpt4': '2-5 seconds',
        'elevenlabs': '1-3 seconds',
        'youtube_upload': '10-30 seconds'
    }
    
    # Official SLA
    SLA = "500ms p95 for internal processing only, excluding external API calls"
```

### 5. Infrastructure Phases (EXPLICIT)
```yaml
infrastructure_timeline:
  mvp_phase_weeks_1_12:
    location: LOCAL ONLY
    services:
      - PostgreSQL (local)
      - Redis (local)
      - N8N (local)
      - File storage (local NVMe)
    external_apis:
      - YouTube API
      - OpenAI API
      - ElevenLabs API
      - Stripe API
    explicitly_excluded:
      - AWS S3
      - CloudFlare (except free DNS)
      - Any cloud compute
      - Any cloud storage
      
  phase_2_months_4_6:
    additions:
      - AWS S3 (backups only)
      - CloudFlare CDN (if needed)
      - Cloud GPU (peak processing)
    note: "Integration Specialist prepares for these in planning only"
```

### 6. Capacity vs Target Users (CLARIFIED)
```yaml
user_metrics:
  system_capacity: 100 users  # What the system can technically handle
  mvp_target: 50 users  # What we're actually launching with
  rationale: "Build with headroom but launch conservatively"
```

---

## Project Overview

YTEMPIRE is an automated YouTube content platform enabling creators to manage 5+ channels with 95% automation. The Backend Team provides the core infrastructure powering this automation through efficient APIs, smart data pipelines, and N8N-based integrations.

### Mission Statement
Build a cost-effective, locally-deployed platform achieving 95% automation for 50 beta users (with capacity for 100), producing 50 videos daily at <$3.00 per video within 12 weeks.

### Timeline: 12 WEEKS (Standardized)
- **NOT** "3 months" or "90 days"
- **Exactly** 84 days from start to beta launch
- **All** planning in weekly increments

## Backend Team Responsibilities

### Backend Team Lead
**Reports to**: CTO/Technical Director  
**Direct Reports**: 3 engineers

**Key Responsibilities**:
- Architecture decisions and technical standards
- Sprint planning (weekly cycles)
- Cost compliance (<$3.00/video)
- Cross-team coordination
- Performance monitoring

### API Developer Engineer
**Reports to**: Backend Team Lead

**Core Deliverables**:
```python
class APIDeveloper:
    """FastAPI REST-only implementation"""
    
    endpoints = {
        '/api/v1/channels': 'CRUD operations',
        '/api/v1/videos': 'Video management',
        '/api/v1/analytics': 'Performance metrics',
        '/api/v1/costs': 'Cost tracking',
        '/api/v1/webhooks/n8n': 'N8N callbacks'
    }
    
    performance_guarantee = {
        'internal_p95': 500,  # ms, our code only
        'cache_hit_rate': 0.40,  # 40% minimum
        'error_rate': 0.05  # 5% maximum
    }
    
    youtube_wrapper = {
        'ownership': 'You build the abstraction',
        'usage': 'Integration Specialist uses in N8N',
        'quota_management': 'Shared responsibility'
    }
```

### Data Pipeline Engineer
**Reports to**: Backend Team Lead

**Clear Scope for MVP**:
```python
class DataPipelineEngineer:
    """Local data processing only - NOT ML pipelines"""
    
    mvp_pipelines = {
        'youtube_analytics': {
            'frequency': 'daily',
            'volume': '500MB/day',  # 50 users data
            'storage': 'PostgreSQL'
        },
        'cost_tracking': {
            'frequency': 'per video',
            'alert': 'If approaching $3.00',
            'dashboard': 'Real-time updates'
        },
        'performance_metrics': {
            'frequency': 'hourly',
            'metrics': ['views', 'ctr', 'retention'],
            'cache': 'Redis with 1hr TTL'
        }
    }
    
    not_your_responsibility = [
        'ML training data',  # AI Team's Data Engineer
        'Feature engineering',  # AI Team
        'Model performance metrics'  # AI Team
    ]
```

### Integration Specialist
**Reports to**: Backend Team Lead

**N8N-First Approach (EXPLICIT)**:
```javascript
// PRIMARY RESPONSIBILITY: N8N Workflows (80% of time)
const n8nWorkflows = {
  contentGeneration: {
    trigger: 'Daily schedule for each user',
    nodes: [
      'Check user quota',
      'Select channel for today',
      'Generate content (OpenAI node)',
      'Create voice (ElevenLabs node)',
      'Process video (webhook to internal API)',
      'Upload to YouTube',
      'Update costs',
      'Handle errors'
    ],
    customNodes: [
      'ytempire-cost-tracker',
      'ytempire-quota-manager',
      'ytempire-cache-handler'
    ]
  }
};

// SECONDARY: Custom integrations (20% of time)
const customWork = {
  when: 'Only if N8N cannot handle',
  examples: ['Complex retry logic', 'Special auth flows'],
  language: 'Python preferred for consistency'
};
```

---

## 12-Week Implementation Plan

### Weeks 1-2: Foundation
**All Team Members**:
- Local server setup (Ryzen 9 9950X3D)
- Development environment configuration
- N8N installation (Integration Specialist leads)
- PostgreSQL/Redis setup (Data Pipeline Engineer leads)
- FastAPI initialization (API Developer leads)

### Weeks 3-4: Core Development
**API Developer**:
- Authentication system
- Channel CRUD APIs
- User management

**Data Pipeline Engineer**:
- Database schema implementation
- Redis caching strategy
- Cost tracking tables

**Integration Specialist**:
- First N8N workflow
- API credential management
- OpenAI test integration

### Weeks 5-6: Integration Phase
**Team Collaboration**:
- API Developer + Integration Specialist: YouTube wrapper integration
- Data Pipeline Engineer + Integration Specialist: Cost tracking workflow
- All: End-to-end test of first video generation

### Weeks 7-8: Optimization
**Focus**: Cost reduction and performance

**Integration Specialist**:
- Implement caching in N8N workflows
- Batch processing for API calls
- Fallback mechanisms

**API Developer**:
- Cache layer optimization
- Response time improvements

**Data Pipeline Engineer**:
- Query optimization
- Analytics pipeline tuning

### Weeks 9-10: Testing
**All Team**:
- Load testing (50 concurrent users)
- Cost verification (<$3.00/video)
- Performance validation (500ms p95 internal)
- N8N workflow reliability

### Weeks 11-12: Beta Launch
**Backend Team Lead** coordinates:
- Production deployment
- Beta user onboarding
- Issue resolution
- Performance monitoring

---

## Budget & Cost Management

### API Budget Reality (12 Weeks)
```yaml
total_api_budget: $35,000
total_videos: 4,200 (50 daily × 84 days)
budget_per_video: $8.33

actual_cost_target: $3.00/video
actual_api_spend: $12,600 (4,200 × $3.00)
remaining_buffer: $22,400 (for overages and testing)

mandatory_optimizations:
  cache_hit_rate: 40%  # Reduces API calls by 40%
  gpt_3.5_usage: 60%  # Use cheaper model when possible
  batch_processing: All ElevenLabs calls
  local_processing: Thumbnails when possible
```

### Daily Cost Monitoring
```python
class CostController:
    MAX_DAILY_SPEND = 150  # $35,000 ÷ 84 days ÷ 3 (safety)
    MAX_PER_VIDEO = 3.00  # Hard limit
    
    def enforce_limits(self):
        if daily_spend > self.MAX_DAILY_SPEND:
            stop_all_generation()
        if video_cost > self.MAX_PER_VIDEO:
            cancel_video()
```

---

## Technical Specifications

### System Capacity & Performance
```yaml
capacity_metrics:
  concurrent_users: 100  # System capacity
  active_users: 50  # MVP target
  daily_videos: 50  # 1 per active user
  
performance_sla:
  api_response_internal: <500ms p95
  video_generation_total: <5 minutes
  dashboard_load: <2 seconds
  error_rate: <5%
  
infrastructure_limits:
  gpu_concurrent_videos: 3-4
  database_connections: 100
  n8n_workflows: 20 concurrent
  memory_usage: <100GB
```

### YouTube API Quota Management
```python
class QuotaManager:
    """Shared between API Developer and Integration Specialist"""
    
    DAILY_QUOTA = 10000
    
    # Allocation for 50 users
    allocation = {
        'uploads': 2000,  # 50 videos × 40 units
        'metadata': 1000,  # Updates and descriptions
        'analytics': 2000,  # Daily pulls
        'read_operations': 2000,  # Checks and lists
        'emergency_buffer': 3000  # Never touch except crisis
    }
    
    def cost_per_operation(self):
        return {
            'video_upload': 1600,
            'thumbnail_set': 50,
            'metadata_update': 50,
            'playlist_insert': 50,
            'analytics_query': 1
        }
```

---

## Risk Management

### Critical Risks with Mitigations

| Risk | Owner | Mitigation | Success Metric |
|------|-------|------------|----------------|
| API costs exceed $3/video | Backend Team Lead | Daily monitoring, hard stops | Stay under $3.00 |
| N8N cannot handle complexity | Integration Specialist | Build custom nodes early | 90% in N8N |
| Local server failure | Platform Ops Lead | 4-hour recovery plan | <4hr RTO |
| YouTube quota exceeded | Integration Specialist | Cache aggressively | <7000 units/day |
| Performance degradation | API Developer | Profiling, optimization | 500ms p95 maintained |

---

## Communication & Reporting

### Daily Standup (15 min)
```markdown
Format:
1. Yesterday's progress (2 min/person)
2. Today's plan (1 min/person)
3. Blockers (5 min total)
4. Cost status (Integration Specialist)
```

### Weekly Backend Team Meeting (1 hour)
```markdown
Agenda:
1. Sprint review (15 min)
2. Technical decisions (15 min)
3. Cost analysis (10 min)
4. Dependencies (10 min)
5. Next sprint planning (10 min)
```

---

## Success Criteria (Week 12)

### Must Achieve
- ✅ 50 active beta users
- ✅ 250 total channels (5 per user)
- ✅ 50 videos generated daily
- ✅ <$3.00 cost per video
- ✅ 95% automation achieved
- ✅ <1 hour weekly user management
- ✅ 500ms p95 internal API response
- ✅ 90% N8N workflow success rate

### Nice to Have
- 60 active users (within 100 capacity)
- $2.50 cost per video
- 99% automation
- 300ms p95 internal response

---

## Clarifications Summary

1. **Team Size**: 17 people total (12 Technical + 5 AI) ✓
2. **Video Scale**: 50 videos daily (1 per user) ✓
3. **Cost Target**: $3.00 per video maximum ✓
4. **Timeline**: 12 weeks (not "3 months") ✓
5. **Infrastructure**: Local-only for MVP ✓
6. **Performance**: 500ms p95 for internal processing only ✓
7. **Integration**: N8N-first approach ✓
8. **Users**: 50 target, 100 capacity ✓

---

## Conclusion

The Backend Team has crystal-clear objectives: Build a locally-deployed MVP in exactly 12 weeks that supports 50 users (with 100-user capacity), generates 50 videos daily at <$3.00 per video, achieving 95% automation with N8N-based workflows.

Every decision should pass three tests:
1. Does it work within the $3.00/video constraint?
2. Can it be built in 12 weeks with our team?
3. Does it increase automation toward 95%?

Let's build this!