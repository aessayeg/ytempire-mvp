# N8N Workflow Development Manual

**Document Version**: 1.0  
**For**: Integration Specialist  
**Priority**: CRITICAL - 80% OF YOUR ROLE  
**Last Updated**: January 2025

---

## 🎯 N8N: Your Primary Tool

### Why N8N is Critical
N8N is the orchestration engine that connects all YTEMPIRE services. You'll spend 80% of your time building, optimizing, and maintaining N8N workflows that power our entire video generation pipeline.

---

## 🚀 N8N Environment Setup

### Docker Deployment

```bash
# docker-compose.yml for local N8N
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
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - NODE_ENV=production
      - WEBHOOK_URL=http://localhost:5678/
      - GENERIC_TIMEZONE=America/Los_Angeles
      - N8N_METRICS=true
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=${DB_PASSWORD}
      - EXECUTIONS_DATA_PRUNE=true
      - EXECUTIONS_DATA_MAX_AGE=168  # 7 days
    volumes:
      - n8n_data:/home/node/.n8n
      - ./custom-nodes:/home/node/.n8n/custom
    depends_on:
      - postgres
    networks:
      - ytempire_network

  postgres:
    image: postgres:14
    container_name: n8n_postgres
    restart: always
    environment:
      - POSTGRES_USER=n8n
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=n8n
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ytempire_network

volumes:
  n8n_data:
  postgres_data:

networks:
  ytempire_network:
    driver: bridge
```

### Initial Configuration

```bash
# Start N8N
docker-compose up -d

# Access N8N UI
# URL: http://localhost:5678
# Username: admin
# Password: [from environment]

# Verify installation
curl http://localhost:5678/healthz
```

---

## 📊 Core Workflows to Implement

### 1. Master Video Generation Workflow

```javascript
// Workflow: video-generation-pipeline
{
  "name": "Video Generation Pipeline",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300],
      "webhookId": "video-generation",
      "parameters": {
        "path": "video-generate",
        "responseMode": "responseNode",
        "options": {
          "responseData": "allEntries",
          "responsePropertyName": "data"
        }
      }
    },
    {
      "name": "Validate Request",
      "type": "n8n-nodes-base.function",
      "position": [450, 300],
      "parameters": {
        "functionCode": `
          // Validate incoming request
          const required = ['channel_id', 'topic', 'style'];
          const data = items[0].json;
          
          for (const field of required) {
            if (!data[field]) {
              throw new Error(\`Missing required field: \${field}\`);
            }
          }
          
          // Add metadata
          data.job_id = generateUUID();
          data.timestamp = new Date().toISOString();
          data.status = 'initialized';
          
          return [{json: data}];
        `
      }
    },
    {
      "name": "Check Compliance",
      "type": "n8n-nodes-base.httpRequest",
      "position": [650, 300],
      "parameters": {
        "url": "http://localhost:8000/api/v1/compliance/check",
        "method": "POST",
        "bodyParametersJson": "={{ $json }}",
        "options": {
          "timeout": 10000
        }
      }
    },
    {
      "name": "Compliance Router",
      "type": "n8n-nodes-base.switch",
      "position": [850, 300],
      "parameters": {
        "dataPropertyName": "compliance_status",
        "values": [
          {"value": "approved"},
          {"value": "rejected"},
          {"value": "needs_review"}
        ]
      }
    },
    {
      "name": "Generate Script",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1050, 200],
      "parameters": {
        "url": "http://localhost:8000/api/v1/ai/generate-script",
        "method": "POST",
        "authentication": "genericCredentialType",
        "genericAuthType": "httpHeaderAuth",
        "bodyParametersJson": "={{ $json }}",
        "options": {
          "timeout": 30000
        }
      }
    },
    {
      "name": "Cost Check",
      "type": "ytempire-cost-tracker",  // Custom node
      "position": [1250, 200],
      "parameters": {
        "service": "openai",
        "operation": "script_generation",
        "video_id": "={{ $json.job_id }}",
        "amount": "={{ $json.cost }}",
        "threshold": 0.40
      }
    },
    {
      "name": "Generate Audio",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1450, 200],
      "parameters": {
        "url": "http://localhost:8000/api/v1/tts/synthesize",
        "method": "POST",
        "bodyParametersJson": `{
          "text": "{{ $json.script }}",
          "voice": "{{ $json.voice_preference }}",
          "service": "google"
        }`,
        "options": {
          "timeout": 60000
        }
      }
    },
    {
      "name": "Process Video",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1650, 200],
      "parameters": {
        "url": "http://localhost:8000/api/v1/video/process",
        "method": "POST",
        "bodyParametersJson": "={{ $json }}",
        "options": {
          "timeout": 300000  // 5 minutes
        }
      }
    },
    {
      "name": "Select YouTube Account",
      "type": "ytempire-account-selector",  // Custom node
      "position": [1850, 200],
      "parameters": {
        "strategy": "health_score",
        "video_id": "={{ $json.job_id }}"
      }
    },
    {
      "name": "Upload to YouTube",
      "type": "n8n-nodes-base.httpRequest",
      "position": [2050, 200],
      "parameters": {
        "url": "http://localhost:8000/api/v1/youtube/upload",
        "method": "POST",
        "bodyParametersJson": `{
          "video_path": "{{ $json.video_path }}",
          "metadata": "{{ $json.metadata }}",
          "account_id": "{{ $json.selected_account }}"
        }`,
        "options": {
          "timeout": 120000  // 2 minutes
        }
      }
    },
    {
      "name": "Update Analytics",
      "type": "n8n-nodes-base.httpRequest",
      "position": [2250, 200],
      "parameters": {
        "url": "http://localhost:8000/api/v1/analytics/track",
        "method": "POST",
        "bodyParametersJson": "={{ $json }}"
      }
    },
    {
      "name": "Send Success Response",
      "type": "n8n-nodes-base.respondToWebhook",
      "position": [2450, 200],
      "parameters": {
        "responseCode": 200,
        "responseData": "={{ $json }}"
      }
    },
    {
      "name": "Handle Rejection",
      "type": "n8n-nodes-base.respondToWebhook",
      "position": [1050, 400],
      "parameters": {
        "responseCode": 400,
        "responseData": `{
          "error": "Content rejected",
          "reason": "{{ $json.rejection_reason }}",
          "suggestions": "{{ $json.alternatives }}"
        }`
      }
    },
    {
      "name": "Error Handler",
      "type": "n8n-nodes-base.errorTrigger",
      "position": [1450, 500],
      "parameters": {}
    },
    {
      "name": "Log Error",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1650, 500],
      "parameters": {
        "url": "http://localhost:8000/api/v1/errors/log",
        "method": "POST",
        "bodyParametersJson": `{
          "workflow": "video-generation-pipeline",
          "error": "{{ $json.error }}",
          "job_id": "{{ $json.job_id }}",
          "stage": "{{ $json.stage }}"
        }`
      }
    }
  ]
}
```

### 2. Daily Scheduling Workflow

```javascript
// Workflow: daily-video-scheduler
{
  "name": "Daily Video Scheduler",
  "nodes": [
    {
      "name": "Cron Trigger",
      "type": "n8n-nodes-base.cron",
      "position": [250, 300],
      "parameters": {
        "triggerTimes": {
          "item": [
            {"hour": 6, "minute": 0},   // Morning batch
            {"hour": 12, "minute": 0},  // Afternoon batch
            {"hour": 18, "minute": 0}   // Evening batch
          ]
        }
      }
    },
    {
      "name": "Get Scheduled Videos",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300],
      "parameters": {
        "url": "http://localhost:8000/api/v1/videos/scheduled",
        "method": "GET",
        "queryParametersJson": `{
          "time_window": "{{ $json.trigger_time }}",
          "limit": 20
        }`
      }
    },
    {
      "name": "Split Videos",
      "type": "n8n-nodes-base.splitInBatches",
      "position": [650, 300],
      "parameters": {
        "batchSize": 5,
        "options": {
          "reset": false
        }
      }
    },
    {
      "name": "Process Batch",
      "type": "n8n-nodes-base.executeWorkflow",
      "position": [850, 300],
      "parameters": {
        "workflowId": "video-generation-pipeline",
        "waitForSubWorkflow": true
      }
    },
    {
      "name": "Wait Between Batches",
      "type": "n8n-nodes-base.wait",
      "position": [1050, 300],
      "parameters": {
        "amount": 5,
        "unit": "minutes"
      }
    }
  ]
}
```

### 3. Cost Monitoring Workflow

```javascript
// Workflow: cost-monitoring-alerts
{
  "name": "Cost Monitoring & Alerts",
  "nodes": [
    {
      "name": "Webhook - Cost Update",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300],
      "webhookId": "cost-update",
      "parameters": {
        "path": "cost-track",
        "responseMode": "immediateResponse"
      }
    },
    {
      "name": "Calculate Running Total",
      "type": "n8n-nodes-base.function",
      "position": [450, 300],
      "parameters": {
        "functionCode": `
          const redis = require('redis');
          const client = redis.createClient();
          
          const videoId = items[0].json.video_id;
          const newCost = items[0].json.cost;
          const service = items[0].json.service;
          
          // Update running total
          const key = \`cost:\${videoId}\`;
          await client.hincrbyfloat(key, service, newCost);
          const total = await client.hget(key, 'total') || 0;
          const newTotal = parseFloat(total) + newCost;
          await client.hset(key, 'total', newTotal);
          
          return [{
            json: {
              video_id: videoId,
              service: service,
              cost: newCost,
              total_cost: newTotal,
              timestamp: new Date().toISOString()
            }
          }];
        `
      }
    },
    {
      "name": "Check Thresholds",
      "type": "n8n-nodes-base.switch",
      "position": [650, 300],
      "parameters": {
        "dataPropertyName": "total_cost",
        "rules": [
          {
            "operation": "larger",
            "value": 0.50,
            "output": "critical"
          },
          {
            "operation": "larger",
            "value": 0.45,
            "output": "high"
          },
          {
            "operation": "larger",
            "value": 0.40,
            "output": "warning"
          }
        ],
        "fallbackOutput": "normal"
      }
    },
    {
      "name": "Critical Alert",
      "type": "n8n-nodes-base.slack",
      "position": [850, 200],
      "parameters": {
        "channel": "#critical-alerts",
        "text": "🚨 COST ALERT: Video {{ $json.video_id }} at ${{ $json.total_cost }}",
        "attachments": [{
          "color": "danger",
          "fields": [
            {
              "title": "Total Cost",
              "value": "${{ $json.total_cost }}",
              "short": true
            },
            {
              "title": "Threshold",
              "value": "$0.50",
              "short": true
            }
          ]
        }]
      }
    },
    {
      "name": "Stop Processing",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1050, 200],
      "parameters": {
        "url": "http://localhost:8000/api/v1/video/stop",
        "method": "POST",
        "bodyParametersJson": `{
          "video_id": "{{ $json.video_id }}",
          "reason": "cost_exceeded",
          "total_cost": "{{ $json.total_cost }}"
        }`
      }
    }
  ]
}
```

---

## 🛠️ Custom N8N Nodes Development

### Custom Node Structure

```typescript
// custom-nodes/ytempire-cost-tracker/YtempireCostTracker.node.ts

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
    subtitle: '={{ $parameter["operation"] }}',
    description: 'Track and monitor costs for YTEMPIRE videos',
    defaults: {
      name: 'Cost Tracker',
    },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [
      {
        name: 'ytempireApi',
        required: true,
      },
    ],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        options: [
          {
            name: 'Track Cost',
            value: 'track',
            description: 'Track cost for a service',
          },
          {
            name: 'Get Total',
            value: 'getTotal',
            description: 'Get total cost for a video',
          },
          {
            name: 'Check Threshold',
            value: 'checkThreshold',
            description: 'Check if cost exceeds threshold',
          },
        ],
        default: 'track',
        noDataExpression: true,
      },
      {
        displayName: 'Video ID',
        name: 'videoId',
        type: 'string',
        default: '',
        required: true,
        displayOptions: {
          show: {
            operation: ['track', 'getTotal', 'checkThreshold'],
          },
        },
        description: 'The ID of the video to track costs for',
      },
      {
        displayName: 'Service',
        name: 'service',
        type: 'options',
        options: [
          { name: 'OpenAI', value: 'openai' },
          { name: 'Google TTS', value: 'google_tts' },
          { name: 'ElevenLabs', value: 'elevenlabs' },
          { name: 'Video Processing', value: 'video_processing' },
          { name: 'Storage', value: 'storage' },
        ],
        default: 'openai',
        displayOptions: {
          show: {
            operation: ['track'],
          },
        },
      },
      {
        displayName: 'Amount',
        name: 'amount',
        type: 'number',
        typeOptions: {
          numberPrecision: 4,
        },
        default: 0,
        displayOptions: {
          show: {
            operation: ['track'],
          },
        },
        description: 'Cost amount in dollars',
      },
      {
        displayName: 'Threshold',
        name: 'threshold',
        type: 'number',
        typeOptions: {
          numberPrecision: 2,
        },
        default: 0.50,
        displayOptions: {
          show: {
            operation: ['checkThreshold'],
          },
        },
        description: 'Cost threshold in dollars',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];
    const credentials = await this.getCredentials('ytempireApi');
    
    for (let i = 0; i < items.length; i++) {
      const operation = this.getNodeParameter('operation', i) as string;
      const videoId = this.getNodeParameter('videoId', i) as string;
      
      if (operation === 'track') {
        const service = this.getNodeParameter('service', i) as string;
        const amount = this.getNodeParameter('amount', i) as number;
        
        // Track the cost
        const response = await trackCost(videoId, service, amount, credentials);
        
        returnData.push({
          json: {
            ...items[i].json,
            cost_tracked: true,
            total_cost: response.total_cost,
            within_budget: response.total_cost < 0.50,
            service,
            amount,
          },
        });
      } else if (operation === 'getTotal') {
        const total = await getTotalCost(videoId, credentials);
        
        returnData.push({
          json: {
            ...items[i].json,
            video_id: videoId,
            total_cost: total,
          },
        });
      } else if (operation === 'checkThreshold') {
        const threshold = this.getNodeParameter('threshold', i) as number;
        const total = await getTotalCost(videoId, credentials);
        const exceeded = total > threshold;
        
        if (exceeded) {
          // Can trigger alerts or stop processing
          await sendCostAlert(videoId, total, threshold, credentials);
        }
        
        returnData.push({
          json: {
            ...items[i].json,
            video_id: videoId,
            total_cost: total,
            threshold,
            exceeded,
            action: exceeded ? 'stop_processing' : 'continue',
          },
        });
      }
    }
    
    return [returnData];
  }
}
```

### Installing Custom Nodes

```bash
# Build custom node
cd custom-nodes/ytempire-cost-tracker
npm run build

# Copy to N8N
cp -r dist/* ~/.n8n/custom/

# Restart N8N
docker-compose restart n8n

# Verify node appears in N8N UI
```

---

## 📝 Webhook Management

### Webhook Configuration

```javascript
// Webhook endpoints to configure
const WEBHOOKS = {
  // Incoming (N8N receives)
  incoming: {
    videoGeneration: {
      path: '/webhook/video-generate',
      method: 'POST',
      auth: 'bearer',
      response: 'immediate'
    },
    statusUpdate: {
      path: '/webhook/status-update',
      method: 'POST',
      auth: 'bearer',
      response: 'lastNode'
    },
    costTracking: {
      path: '/webhook/cost-track',
      method: 'POST',
      auth: 'none',  // Internal only
      response: 'immediate'
    }
  },
  
  // Outgoing (N8N calls)
  outgoing: {
    progressUpdate: {
      url: 'http://localhost:8000/api/v1/n8n/progress',
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ${API_TOKEN}',
        'Content-Type': 'application/json'
      }
    },
    errorNotification: {
      url: 'http://localhost:8000/api/v1/n8n/error',
      method: 'POST',
      retry: 3,
      timeout: 10000
    },
    completionCallback: {
      url: 'http://localhost:8000/api/v1/n8n/complete',
      method: 'POST',
      includeExecutionData: true
    }
  }
};
```

### Webhook Security

```javascript
// Implement webhook authentication
function validateWebhookRequest(req) {
  // 1. Check Bearer token
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token || token !== process.env.WEBHOOK_TOKEN) {
    throw new Error('Invalid webhook token');
  }
  
  // 2. Validate signature (for external webhooks)
  const signature = req.headers['x-webhook-signature'];
  const payload = JSON.stringify(req.body);
  const expectedSignature = crypto
    .createHmac('sha256', process.env.WEBHOOK_SECRET)
    .update(payload)
    .digest('hex');
  
  if (signature !== expectedSignature) {
    throw new Error('Invalid webhook signature');
  }
  
  // 3. Check timestamp (prevent replay attacks)
  const timestamp = req.headers['x-webhook-timestamp'];
  const currentTime = Date.now();
  const webhookTime = parseInt(timestamp);
  
  if (Math.abs(currentTime - webhookTime) > 300000) { // 5 minutes
    throw new Error('Webhook timestamp too old');
  }
  
  return true;
}
```

---

## 🔄 Error Handling & Recovery

### Error Handling Patterns

```javascript
// Global error handling workflow
{
  "name": "Global Error Handler",
  "nodes": [
    {
      "name": "Error Trigger",
      "type": "n8n-nodes-base.errorTrigger",
      "position": [250, 300]
    },
    {
      "name": "Classify Error",
      "type": "n8n-nodes-base.function",
      "position": [450, 300],
      "parameters": {
        "functionCode": `
          const error = items[0].json.error;
          const workflow = items[0].json.workflow;
          
          // Classify error severity
          let severity = 'low';
          let retryable = true;
          let action = 'retry';
          
          if (error.includes('quota')) {
            severity = 'high';
            action = 'switch_account';
          } else if (error.includes('authentication')) {
            severity = 'medium';
            action = 'refresh_token';
          } else if (error.includes('rate limit')) {
            severity = 'low';
            action = 'backoff';
          } else if (error.includes('cost exceeded')) {
            severity = 'critical';
            retryable = false;
            action = 'stop';
          }
          
          return [{
            json: {
              ...items[0].json,
              severity,
              retryable,
              action,
              timestamp: new Date().toISOString()
            }
          }];
        `
      }
    },
    {
      "name": "Route by Severity",
      "type": "n8n-nodes-base.switch",
      "position": [650, 300],
      "parameters": {
        "dataPropertyName": "severity",
        "values": [
          {"value": "critical"},
          {"value": "high"},
          {"value": "medium"},
          {"value": "low"}
        ]
      }
    },
    {
      "name": "Critical Handler",
      "type": "n8n-nodes-base.executeWorkflow",
      "position": [850, 200],
      "parameters": {
        "workflowId": "critical-error-handler"
      }
    },
    {
      "name": "Retry Handler",
      "type": "n8n-nodes-base.wait",
      "position": [850, 400],
      "parameters": {
        "amount": "={{ $json.retry_delay || 60 }}",
        "unit": "seconds"
      }
    }
  ]
}
```

### Retry Strategies

```javascript
// Exponential backoff implementation
function calculateRetryDelay(attemptNumber) {
  const baseDelay = 1000; // 1 second
  const maxDelay = 300000; // 5 minutes
  const jitter = Math.random() * 1000; // 0-1 second jitter
  
  const delay = Math.min(
    baseDelay * Math.pow(2, attemptNumber) + jitter,
    maxDelay
  );
  
  return delay;
}

// Retry configuration by service
const RETRY_CONFIGS = {
  youtube: {
    maxAttempts: 5,
    backoffMultiplier: 2,
    maxDelay: 300000,
    retryableErrors: ['rateLimitExceeded', 'processingFailed', 'networkError']
  },
  openai: {
    maxAttempts: 3,
    backoffMultiplier: 1.5,
    maxDelay: 60000,
    retryableErrors: ['timeout', 'serverError', 'rateLimitExceeded']
  },
  payment: {
    maxAttempts: 3,
    backoffMultiplier: 2,
    maxDelay: 30000,
    retryableErrors: ['networkError', 'timeout']
  }
};
```

---

## 📊 Monitoring & Optimization

### Workflow Metrics to Track

```yaml
key_metrics:
  performance:
    - Execution time per workflow
    - Success rate per workflow
    - Error rate by node
    - Queue depth
    - Memory usage
  
  business:
    - Videos processed per hour
    - Cost per video
    - Upload success rate
    - API quota usage
    - Account health scores
  
  alerts:
    critical:
      - Workflow failure rate > 10%
      - Execution time > 15 minutes
      - Cost per video > $0.50
      - All accounts quota > 80%
    
    warning:
      - Single node failure rate > 5%
      - Memory usage > 80%
      - Queue depth > 100
      - API errors increasing
```

### Performance Optimization Tips

```javascript
// 1. Use parallel processing where possible
{
  "name": "Parallel Processing",
  "type": "n8n-nodes-base.splitInBatches",
  "parameters": {
    "batchSize": 10,
    "options": {
      "parallel": true,
      "maxParallel": 5
    }
  }
}

// 2. Implement caching
{
  "name": "Cache Check",
  "type": "n8n-nodes-base.function",
  "parameters": {
    "functionCode": `
      const cacheKey = \`cache:\${items[0].json.topic}\`;
      const cached = await redis.get(cacheKey);
      
      if (cached) {
        return [{
          json: {
            ...JSON.parse(cached),
            from_cache: true
          }
        }];
      }
      
      // Continue to generation if not cached
      return items;
    `
  }
}

// 3. Batch API calls
{
  "name": "Batch API Calls",
  "type": "n8n-nodes-base.function",
  "parameters": {
    "functionCode": `
      // Collect multiple items
      const batch = [];
      for (const item of items) {
        batch.push(item.json);
        
        if (batch.length === 50) {
          // Send batch request
          await sendBatchRequest(batch);
          batch = [];
        }
      }
      
      // Send remaining
      if (batch.length > 0) {
        await sendBatchRequest(batch);
      }
    `
  }
}
```

---

## 🔧 Testing & Debugging

### Test Workflows

```javascript
// Test workflow for each component
const TEST_WORKFLOWS = {
  // Test YouTube upload
  testYouTubeUpload: {
    trigger: 'manual',
    data: {
      video_path: '/test/sample.mp4',
      title: 'Test Video',
      description: 'Test upload',
      account_id: 'ytempire_prod_01'
    },
    expectedResult: {
      youtube_id: 'string',
      upload_time: '<2 minutes'
    }
  },
  
  // Test cost tracking
  testCostTracking: {
    trigger: 'manual',
    data: {
      video_id: 'test_123',
      costs: [
        {service: 'openai', amount: 0.20},
        {service: 'google_tts', amount: 0.10},
        {service: 'video_processing', amount: 0.15}
      ]
    },
    expectedResult: {
      total_cost: 0.45,
      alert_triggered: true
    }
  }
};
```

### Debugging Tools

```javascript
// Debug node for logging
{
  "name": "Debug Logger",
  "type": "n8n-nodes-base.function",
  "parameters": {
    "functionCode": `
      console.log('=== DEBUG ===');
      console.log('Node:', $node.name);
      console.log('Execution ID:', $execution.id);
      console.log('Workflow:', $workflow.name);
      console.log('Data:', JSON.stringify(items[0].json, null, 2));
      console.log('=============');
      
      // Log to external service for persistence
      await logToElasticsearch({
        timestamp: new Date(),
        workflow: $workflow.name,
        node: $node.name,
        execution_id: $execution.id,
        data: items[0].json
      });
      
      return items;
    `
  }
}
```

---

## 🚀 Production Deployment

### Deployment Checklist

```yaml
pre_deployment:
  - [ ] All workflows tested in dev environment
  - [ ] Error handling implemented for all workflows
  - [ ] Webhook security configured
  - [ ] Custom nodes installed and tested
  - [ ] Monitoring dashboards configured
  - [ ] Backup and recovery procedures documented

deployment:
  - [ ] Export all workflows as JSON
  - [ ] Version control workflow files
  - [ ] Configure production credentials
  - [ ] Set production environment variables
  - [ ] Enable workflow execution logs
  - [ ] Configure alerting rules

post_deployment:
  - [ ] Verify all webhooks accessible
  - [ ] Test each workflow with production data
  - [ ] Monitor first 24 hours closely
  - [ ] Document any issues found
  - [ ] Optimize based on production metrics
```

### Backup & Recovery

```bash
# Backup N8N workflows and data
#!/bin/bash

# Export all workflows
n8n export:workflow --all --output=/backup/workflows_$(date +%Y%m%d).json

# Backup PostgreSQL database
pg_dump -h postgres -U n8n -d n8n > /backup/n8n_db_$(date +%Y%m%d).sql

# Backup credentials (encrypted)
n8n export:credentials --all --output=/backup/credentials_$(date +%Y%m%d).json

# Compress backups
tar -czf /backup/n8n_backup_$(date +%Y%m%d).tar.gz /backup/*.json /backup/*.sql

# Upload to S3 (optional)
aws s3 cp /backup/n8n_backup_$(date +%Y%m%d).tar.gz s3://ytempire-backups/n8n/
```

---

## 📞 Support & Resources

### N8N Resources
- Official Documentation: https://docs.n8n.io
- Community Forum: https://community.n8n.io
- Node Development: https://docs.n8n.io/integrations/creating-nodes/

### Internal Support
- Backend Team Lead: Architecture questions
- API Developers: Endpoint issues
- DevOps: Infrastructure problems
- On-call: Production emergencies

---

**Remember**: N8N is the heart of our automation. Every workflow you optimize directly impacts our ability to scale. Master N8N, and you master YTEMPIRE's future!