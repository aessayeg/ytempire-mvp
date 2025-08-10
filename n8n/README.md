# YTEmpire N8N Backend Integration
**Owner: Integration Specialist**

## Overview

This N8N integration provides workflow automation for YTEmpire's video generation pipeline, enabling seamless orchestration of AI services, YouTube API interactions, and backend system communications.

## Architecture

### Services
- **N8N Workflow Engine**: Main automation platform (Port: 5678)
- **PostgreSQL**: N8N data persistence (Port: 5433)
- **Redis**: N8N queue management (Port: 6380)

### Integration Points
- **Backend API**: Direct communication with YTEmpire backend
- **AI Services**: OpenAI, ElevenLabs integration
- **YouTube API**: Video upload and management
- **Webhook System**: Event-driven communication

## Deployment

### Quick Start
```bash
# Deploy N8N integration
./deploy-n8n.sh

# Or manually with Docker Compose
docker-compose up -d
```

### Manual Setup
1. Start services: `docker-compose up -d`
2. Access N8N at http://localhost:5678
3. Login with admin/ytempire_n8n_2025
4. Import workflow templates
5. Configure API credentials

## Workflows

### 1. Video Generation Workflow
**File**: `workflows/video-generation-workflow.json`

**Trigger**: Webhook `/webhook/video-generation`

**Process Flow**:
1. **Validate Request** - Extract and validate video generation parameters
2. **Update Status** - Notify backend of processing start
3. **Generate Content** - Create script using OpenAI GPT-4
4. **Synthesize Voice** - Convert script to audio using ElevenLabs
5. **Generate Visuals** - Create images/thumbnails using DALL-E
6. **Combine Assets** - Merge audio and visuals
7. **Compile Video** - Create final video file
8. **Complete Workflow** - Update backend with results

**Error Handling**:
- Automatic error detection and logging
- Backend notification on failures
- Cost tracking and budget enforcement

### 2. YouTube Upload Workflow
**File**: `workflows/youtube-upload-workflow.json`

**Trigger**: Webhook `/webhook/youtube-upload`

**Process Flow**:
1. **Process Upload Request** - Validate video and channel data
2. **Upload to YouTube** - Use YouTube API v3 for upload
3. **Process Response** - Handle success/failure cases
4. **Send Callback** - Notify backend of upload status

## Backend Integration

### N8N Service Class
**File**: `backend/app/services/n8n_service.py`

**Key Methods**:
- `trigger_video_generation_workflow()` - Start video generation
- `trigger_youtube_upload_workflow()` - Start YouTube upload
- `trigger_cost_monitoring_workflow()` - Monitor costs
- `process_video_completion_webhook()` - Handle completion callbacks
- `process_youtube_callback_webhook()` - Handle YouTube callbacks

### Webhook Endpoints
**File**: `backend/app/api/v1/endpoints/webhooks.py`

**Available Endpoints**:
```
POST /api/v1/webhooks/n8n/video-complete
POST /api/v1/webhooks/n8n/trigger-upload
POST /api/v1/webhooks/n8n/cost-alert
POST /api/v1/webhooks/youtube/callback
POST /api/v1/webhooks/n8n/analytics-complete
POST /api/v1/webhooks/n8n/optimization-complete
GET  /api/v1/webhooks/n8n/health
```

### Configuration
**File**: `backend/app/core/config.py`

**Environment Variables**:
```bash
N8N_BASE_URL=http://n8n:5678
N8N_API_KEY=your-n8n-api-key
N8N_WEBHOOK_SECRET=ytempire-n8n-secret
BACKEND_URL=http://backend:8000
```

## API Integration

### Video Generation Trigger
When a user creates a video via `/api/v1/videos/generate`, the backend:
1. Creates video record in database
2. Calls `n8n_service.trigger_video_generation_workflow()`
3. N8N processes the video generation
4. N8N calls back to `/webhooks/n8n/video-complete`
5. Backend updates video status and file paths

### Cost Monitoring
- Real-time cost tracking during workflow execution
- Automatic alerts when thresholds are exceeded
- Budget enforcement to prevent overspend

### Error Handling
- Comprehensive error logging in N8N and backend
- Automatic retry mechanisms for transient failures
- User notification system for permanent failures

## Credentials Management

### Required API Keys
1. **OpenAI API Key** - For script generation and image creation
2. **ElevenLabs API Key** - For voice synthesis
3. **YouTube OAuth2** - For video uploads
4. **YTEmpire Backend API Key** - For webhook authentication

### Security Features
- HMAC signature verification for webhooks
- Encrypted credential storage in N8N
- Environment variable configuration
- Role-based access control

## Monitoring & Observability

### Health Checks
- N8N service health monitoring
- Database connectivity checks
- Backend integration status
- Workflow execution monitoring

### Metrics Collection
- Workflow execution times
- Success/failure rates
- Cost per video generation
- API usage statistics

### Logging
- Structured logging in both N8N and backend
- Error tracking and alerting
- Performance monitoring
- Audit trail for all operations

## Workflow Development

### Creating New Workflows
1. Design workflow in N8N visual editor
2. Test with sample data
3. Export workflow JSON
4. Add to version control
5. Update deployment scripts

### Best Practices
- Use environment variables for configuration
- Implement proper error handling
- Add logging at key steps
- Validate input data
- Monitor execution costs
- Test thoroughly before deployment

### Debugging
- Use N8N execution logs
- Monitor webhook delivery
- Check backend logs for integration issues
- Verify API credentials and permissions

## Production Deployment

### Infrastructure Requirements
- Docker environment
- PostgreSQL database
- Redis cache
- Network connectivity between services

### Scaling Considerations
- N8N can run in clustered mode
- Queue-based execution for high volume
- Database connection pooling
- API rate limiting compliance

### Backup & Recovery
- Regular N8N workflow exports
- Database backups
- Credential backup (encrypted)
- Disaster recovery procedures

## Troubleshooting

### Common Issues

1. **Workflow Not Triggering**
   ```bash
   # Check webhook configuration
   curl -X POST http://localhost:5678/webhook/video-generation \
     -H "Content-Type: application/json" \
     -d '{"test": "data"}'
   
   # Check N8N logs
   docker logs ytempire_n8n
   ```

2. **API Authentication Failures**
   - Verify API keys in N8N credentials
   - Check token expiration
   - Validate OAuth2 setup for YouTube

3. **Backend Integration Issues**
   - Verify webhook URLs are accessible
   - Check HMAC signature generation
   - Monitor backend logs for errors

4. **High Processing Costs**
   - Review AI service usage
   - Optimize prompt engineering
   - Implement cost controls

### Log Analysis
- N8N execution logs: Available in N8N dashboard
- Backend logs: Check FastAPI application logs
- Database logs: Monitor PostgreSQL query performance

## Development & Testing

### Local Development
1. Start services: `docker-compose up -d`
2. Access N8N at http://localhost:5678
3. Import development workflows
4. Configure test credentials
5. Run test scenarios

### Testing Strategy
- Unit tests for N8N service methods
- Integration tests for webhook endpoints
- End-to-end workflow testing
- Performance testing under load

### CI/CD Integration
- Automated workflow validation
- Credential security scanning
- Integration test execution
- Deployment automation

## Support & Maintenance

### Regular Tasks
- Monitor workflow execution rates
- Review and optimize costs
- Update API credentials
- Performance tuning
- Security patches

### Monitoring Alerts
- Workflow failure notifications
- Cost threshold alerts
- API quota warnings
- System resource monitoring

### Documentation Updates
- Keep workflow documentation current
- Update API integration guides
- Maintain troubleshooting guides
- Version control all changes