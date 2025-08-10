# 5. WORKFLOW AUTOMATION

## 5.1 N8N Platform Overview

### Introduction to N8N

N8N is the orchestration engine that powers YTEMPIRE's automation, connecting all services and enabling 95% hands-off operation. As the primary tool for the Integration Specialist (80% of role), N8N transforms complex multi-step processes into visual, maintainable workflows.

### Why N8N for YTEMPIRE

**Key Advantages:**
- **Visual Workflow Design**: Non-developers can understand and modify workflows
- **Self-Hosted Control**: Complete data sovereignty on local infrastructure
- **Extensive Integrations**: 200+ built-in nodes plus custom node support
- **Cost-Effective**: No per-execution fees unlike cloud alternatives
- **Real-Time Monitoring**: Built-in execution history and debugging
- **Scalability**: Handles 50+ concurrent workflows for MVP needs

### N8N Architecture in YTEMPIRE

```yaml
N8N Deployment:
  Infrastructure:
    - Container: Docker with 4GB RAM allocated
    - Database: PostgreSQL for workflow storage
    - Queue: Redis for execution queue
    - Storage: 50GB for execution history
    
  Integration Points:
    - Webhooks: Receive triggers from all services
    - API Calls: Connect to 50+ external services  
    - Database: Direct PostgreSQL connections
    - File System: Local video/audio processing
    
  Performance Targets:
    - Concurrent Workflows: 20
    - Execution Time: <5 minutes per video
    - Success Rate: >98%
    - Queue Depth: <100 items
```

### Environment Setup

```bash
# Docker Compose Configuration for N8N
version: '3.8'

services:
  n8n:
    image: n8nio/n8n:latest
    container_name: ytempire_n8n
    restart: always
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=ytempire_admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - NODE_ENV=production
      - EXECUTIONS_PROCESS=main
      - GENERIC_TIMEZONE=America/Los_Angeles
      - N8N_METRICS=true
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=${DB_PASSWORD}
      - EXECUTIONS_DATA_PRUNE=true
      - EXECUTIONS_DATA_MAX_AGE=168  # 7 days retention
    volumes:
      - n8n_data:/home/node/.n8n
      - ./custom-nodes:/home/node/.n8n/custom
    networks:
      - ytempire_network
```

## 5.2 Core Workflows

### Master Video Generation Workflow

The primary workflow handling end-to-end video creation:

```javascript
{
  "name": "YTEMPIRE_Video_Generation_Master",
  "triggers": ["webhook", "schedule", "manual"],
  "stages": {
    "validation": {
      "duration": "5 seconds",
      "actions": ["Check requirements", "Validate channel", "Verify quota"]
    },
    "content_generation": {
      "duration": "45 seconds", 
      "actions": ["Generate script", "Create voice", "Process media"]
    },
    "assembly": {
      "duration": "3 minutes",
      "actions": ["Render video", "Generate thumbnail", "Add metadata"]
    },
    "distribution": {
      "duration": "1 minute",
      "actions": ["Select account", "Upload to YouTube", "Update database"]
    }
  }
}
```

**Key Workflow Components:**

1. **Webhook Trigger**: Receives video generation requests
2. **Validation Node**: Ensures all requirements met
3. **Cost Tracker**: Monitors spending in real-time
4. **Script Generator**: Calls OpenAI with optimized prompts
5. **Voice Synthesizer**: Routes to cheapest TTS provider
6. **Video Processor**: Manages GPU/CPU rendering
7. **YouTube Uploader**: Handles multi-account distribution
8. **Database Updater**: Records all metrics

### Daily Scheduling Workflow

Manages the distribution of 50 videos across optimal time slots:

```javascript
{
  "name": "Daily_Video_Scheduler",
  "schedule": "0 6,12,18 * * *",  // 6 AM, 12 PM, 6 PM
  "distribution": {
    "morning": {"videos": 20, "accounts": 4},
    "afternoon": {"videos": 20, "accounts": 4},
    "evening": {"videos": 10, "accounts": 4}
  }
}
```

### Cost Monitoring Workflow

Real-time cost tracking with automatic interventions:

```javascript
{
  "name": "Cost_Monitor_Alert",
  "triggers": ["cost_update", "threshold_breach"],
  "thresholds": {
    "warning": "$2.50 per video",
    "critical": "$3.00 per video",
    "daily_limit": "$150.00"
  },
  "actions": {
    "warning": ["Send Slack alert", "Switch to economy mode"],
    "critical": ["Stop processing", "Alert team", "Fallback to cache"],
    "daily_limit": ["Halt all generation", "Emergency notification"]
  }
}
```

### Account Health Monitoring Workflow

Tracks YouTube account status and manages rotation:

```javascript
{
  "name": "YouTube_Account_Health",
  "schedule": "*/30 * * * *",  // Every 30 minutes
  "checks": [
    "Quota usage per account",
    "Upload success rate",
    "API errors",
    "Strike warnings"
  ],
  "actions": {
    "quota_high": "Switch to next account",
    "errors_detected": "Mark unhealthy, use reserve",
    "strike_warning": "Quarantine account"
  }
}
```

## 5.3 Custom Node Development

### YTEMPIRE Cost Tracker Node

```typescript
import {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from 'n8n-workflow';

export class YtempireCostTracker implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'YTEMPIRE Cost Tracker',
    name: 'ytempireCostTracker',
    group: ['transform'],
    version: 1,
    description: 'Track and monitor costs for YTEMPIRE videos',
    defaults: { name: 'Cost Tracker' },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        options: [
          { name: 'Track Cost', value: 'track' },
          { name: 'Get Total', value: 'getTotal' },
          { name: 'Check Threshold', value: 'checkThreshold' }
        ],
        default: 'track'
      },
      {
        displayName: 'Service',
        name: 'service',
        type: 'options',
        options: [
          { name: 'OpenAI', value: 'openai' },
          { name: 'Google TTS', value: 'google_tts' },
          { name: 'ElevenLabs', value: 'elevenlabs' }
        ],
        default: 'openai'
      },
      {
        displayName: 'Amount',
        name: 'amount',
        type: 'number',
        typeOptions: { numberPrecision: 4 },
        default: 0
      },
      {
        displayName: 'Threshold',
        name: 'threshold',
        type: 'number',
        default: 3.00,
        description: 'Maximum cost per video'
      }
    ]
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const operation = this.getNodeParameter('operation', 0) as string;
    
    if (operation === 'track') {
      // Track cost logic
      const service = this.getNodeParameter('service', 0) as string;
      const amount = this.getNodeParameter('amount', 0) as number;
      
      // Update Redis with new cost
      await this.trackCost(service, amount);
      
      // Check if approaching threshold
      const total = await this.getTotalCost();
      if (total > 2.50) {
        await this.sendWarning(total);
      }
    }
    
    return [items];
  }
}
```

### YouTube Account Selector Node

Custom node for intelligent account rotation:

```typescript
export class YtempireAccountSelector implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'YouTube Account Selector',
    name: 'ytempireAccountSelector',
    description: 'Select optimal YouTube account for upload',
    properties: [
      {
        displayName: 'Strategy',
        name: 'strategy',
        type: 'options',
        options: [
          { name: 'Health Score', value: 'health' },
          { name: 'Round Robin', value: 'roundrobin' },
          { name: 'Least Used', value: 'leastused' }
        ],
        default: 'health'
      }
    ]
  };

  async execute(): Promise<INodeExecutionData[][]> {
    const strategy = this.getNodeParameter('strategy', 0);
    const account = await this.selectAccount(strategy);
    
    return [{
      json: {
        selected_account: account.id,
        health_score: account.health,
        quota_remaining: account.quota
      }
    }];
  }
}
```

## 5.4 Error Handling & Recovery

### Error Classification System

```javascript
const ERROR_HANDLERS = {
  "quotaExceeded": {
    severity: "critical",
    action: "switch_account",
    retry: true,
    notification: "immediate"
  },
  "rateLimitExceeded": {
    severity: "warning",
    action: "exponential_backoff",
    retry: true,
    max_retries: 5
  },
  "authenticationError": {
    severity: "high",
    action: "refresh_token",
    retry: true,
    escalation: "after_2_failures"
  },
  "apiTimeout": {
    severity: "medium",
    action: "retry_with_backoff",
    retry: true,
    timeout_increase: 2
  },
  "contentViolation": {
    severity: "critical",
    action: "halt_and_review",
    retry: false,
    human_review: true
  }
};
```

### Global Error Handler Workflow

```javascript
{
  "name": "Global_Error_Handler",
  "trigger": "error_event",
  "process": [
    {
      "step": "classify_error",
      "determine": ["type", "severity", "source"]
    },
    {
      "step": "execute_recovery",
      "actions": {
        "low": "log_and_continue",
        "medium": "retry_with_delay",
        "high": "fallback_service",
        "critical": "stop_and_alert"
      }
    },
    {
      "step": "update_metrics",
      "track": ["error_rate", "recovery_time", "impact"]
    }
  ]
}
```

### Retry Strategies

```javascript
// Exponential Backoff Implementation
function calculateRetryDelay(attemptNumber) {
  const baseDelay = 1000; // 1 second
  const maxDelay = 300000; // 5 minutes
  const jitter = Math.random() * 1000; // 0-1 second
  
  return Math.min(
    baseDelay * Math.pow(2, attemptNumber) + jitter,
    maxDelay
  );
}

// Service-Specific Retry Configs
const RETRY_CONFIGS = {
  youtube: {
    maxAttempts: 5,
    backoffMultiplier: 2,
    maxDelay: 300000
  },
  openai: {
    maxAttempts: 3,
    backoffMultiplier: 1.5,
    maxDelay: 60000
  },
  payment: {
    maxAttempts: 3,
    backoffMultiplier: 2,
    maxDelay: 30000
  }
};
```

### Fallback Chains

```javascript
const FALLBACK_CHAINS = {
  script_generation: [
    { service: "openai_gpt4", cost: 0.30 },
    { service: "openai_gpt35", cost: 0.15 },
    { service: "cached_templates", cost: 0.00 }
  ],
  voice_synthesis: [
    { service: "google_tts", cost: 0.08 },
    { service: "elevenlabs", cost: 0.20 },
    { service: "cached_audio", cost: 0.00 }
  ],
  video_upload: [
    { service: "youtube_primary", accounts: 12 },
    { service: "youtube_reserve", accounts: 3 },
    { service: "queue_for_retry", delay: 3600 }
  ]
};
```

## 5.5 Workflow Optimization

### Performance Optimization Strategies

#### 1. Parallel Processing
```javascript
{
  "optimization": "parallel_execution",
  "implementation": {
    "split_batch": {
      "size": 10,
      "parallel_branches": 5
    },
    "benefits": "5x faster processing",
    "use_cases": ["bulk_generation", "analytics_fetch"]
  }
}
```

#### 2. Intelligent Caching
```javascript
{
  "cache_strategy": {
    "levels": {
      "L1": { "type": "memory", "ttl": 300 },
      "L2": { "type": "redis", "ttl": 3600 },
      "L3": { "type": "disk", "ttl": 86400 }
    },
    "cached_items": [
      "common_scripts",
      "frequent_audio",
      "api_responses"
    ],
    "hit_rate_target": ">60%"
  }
}
```

#### 3. Request Batching
```javascript
{
  "batching": {
    "youtube_analytics": {
      "batch_size": 50,
      "frequency": "5 minutes"
    },
    "cost_updates": {
      "batch_size": 100,
      "frequency": "1 minute"
    }
  }
}
```

### Monitoring & Metrics

```yaml
Key Workflow Metrics:
  Performance:
    - Execution time per workflow
    - Success rate per workflow type
    - Error rate by node
    - Queue depth over time
    
  Business:
    - Videos processed per hour
    - Cost per workflow execution
    - API quota consumption
    - Cache hit rates
    
  Alerts:
    Critical:
      - Workflow failure rate >10%
      - Execution time >15 minutes
      - Queue depth >100
      - Cost per video >$3.00
    
    Warning:
      - Single node failure >5%
      - Memory usage >80%
      - API errors increasing
      - Cache hit rate <40%
```

### Testing Workflows

```javascript
// Test Suite for Core Workflows
const WORKFLOW_TESTS = {
  video_generation: {
    test_data: {
      channel_id: "test_channel",
      topic: "Test Topic",
      style: "educational"
    },
    expected_results: {
      execution_time: "<5 minutes",
      cost: "<$3.00",
      outputs: ["video_url", "youtube_id"]
    }
  },
  
  cost_tracking: {
    test_data: {
      services: [
        { name: "openai", cost: 0.20 },
        { name: "google_tts", cost: 0.10 }
      ]
    },
    expected_results: {
      total_cost: 0.30,
      alert_triggered: false
    }
  }
};
```

### Optimization Checklist

- [ ] All workflows have error handlers
- [ ] Parallel processing implemented where possible
- [ ] Caching layer active for expensive operations
- [ ] Batch processing for bulk operations
- [ ] Rate limiting configured for all APIs
- [ ] Monitoring alerts configured
- [ ] Test coverage >80% for critical workflows
- [ ] Documentation complete for custom nodes
- [ ] Backup workflows for critical paths
- [ ] Performance benchmarks established