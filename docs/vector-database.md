# Vector Database Implementation

## Overview

The YTEmpire platform uses **Qdrant** as its vector database to enable semantic search, content recommendations, and similarity matching across videos, scripts, and channels.

## Features

### 🔍 Semantic Search
- Search videos by meaning, not just keywords
- Find similar content across your video library
- Natural language queries

### 🎯 Personalized Recommendations
- AI-powered content suggestions based on user history
- Dynamic recommendation engine
- Context-aware suggestions

### 📊 Content Analysis
- Trending topic detection
- Content clustering and categorization
- Similarity scoring

### 🚀 Real-time Indexing
- Automatic content indexing on video completion
- Bulk reindexing capabilities
- Background processing with Celery

## Architecture

### Components

1. **Qdrant Vector Database**
   - High-performance vector search engine
   - REST and gRPC APIs
   - Persistent storage with Docker volumes

2. **Sentence Transformers**
   - `all-MiniLM-L6-v2` embedding model
   - 384-dimensional vectors
   - Fast inference and good quality

3. **Vector Service Layer**
   - `VectorService` class for database operations
   - Connection management and health checks
   - Error handling and retry logic

4. **Celery Tasks**
   - Automatic content indexing
   - Bulk operations
   - Background processing

### Collections

| Collection | Purpose | Content Type |
|------------|---------|--------------|
| `video_embeddings` | Video content search | Video titles, descriptions |
| `script_embeddings` | Script similarity | Generated video scripts |
| `channel_embeddings` | Channel analysis | Channel metadata |
| `keyword_embeddings` | Keyword matching | Tags and keywords |

## API Endpoints

### Search & Recommendations

```
POST /api/v1/search/semantic-search
GET  /api/v1/search/recommendations
GET  /api/v1/search/trending-topics
GET  /api/v1/search/similar-videos/{video_id}
```

### Content Management

```
POST /api/v1/search/index-content/{content_id}
DELETE /api/v1/search/content/{content_id}
GET  /api/v1/search/vector-stats
```

## Configuration

### Environment Variables

```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=your-api-key  # Optional for production

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
SIMILARITY_THRESHOLD=0.7
```

### Docker Configuration

```yaml
# docker-compose.qdrant.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
```

## Deployment

### 1. Deploy Qdrant

```bash
# Run deployment script
./scripts/deploy-vector-db.sh

# Or manually
docker-compose -f docker-compose.qdrant.yml up -d
```

### 2. Install Dependencies

```bash
pip install qdrant-client sentence-transformers
```

### 3. Initialize Service

The vector service is automatically initialized during application startup.

### 4. Index Content

```bash
# Index all existing videos
curl -X POST "http://localhost:8000/api/v1/search/index-content/bulk" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Usage Examples

### Semantic Search

```python
from app.services.vector_service import vector_service

# Find similar videos
results = await vector_service.find_similar_content(
    query_text="machine learning tutorial",
    content_type="video",
    limit=10
)
```

### Content Indexing

```python
# Index a video
success = await vector_service.store_content_embedding(
    content_id="video-123",
    content_type="video",
    content="Amazing Python Tutorial for Beginners",
    metadata={
        "title": "Python Tutorial",
        "channel_id": "channel-456",
        "tags": ["python", "programming", "tutorial"]
    }
)
```

### Recommendations

```python
# Get personalized recommendations
recommendations = await vector_service.get_content_recommendations(
    user_id="user-123",
    content_history=["python", "web development", "APIs"],
    limit=20
)
```

## API Examples

### Search for Similar Content

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic-search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "query": "artificial intelligence machine learning",
    "content_type": "video",
    "limit": 10,
    "min_score": 0.7
  }'
```

### Get Recommendations

```bash
curl -X GET "http://localhost:8000/api/v1/search/recommendations?limit=20" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Index Content

```bash
curl -X POST "http://localhost:8000/api/v1/search/index-content/video-123?content_type=video" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Monitoring

### Health Checks

The vector database health is monitored through:

```bash
# Application health endpoint
curl http://localhost:8000/health

# Direct Qdrant health
curl http://localhost:6333/health
```

### Metrics

Key metrics tracked:

- `vector_embeddings_stored`: Number of embeddings stored
- `vector_searches_performed`: Number of searches executed
- `vector_errors`: Number of errors encountered

### Collection Statistics

```bash
curl -X GET "http://localhost:8000/api/v1/search/vector-stats" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Performance Considerations

### Embedding Generation
- Model loading: ~2-3 seconds on startup
- Embedding generation: ~10-50ms per text
- Batch processing recommended for bulk operations

### Search Performance
- Small collections (<10K): <10ms
- Medium collections (10K-100K): <50ms
- Large collections (>100K): <200ms

### Storage Requirements
- 384 dimensions × 4 bytes = ~1.5KB per vector
- Plus metadata and indexes
- Estimate: ~3KB per video embedding

### Scaling Recommendations

1. **Horizontal Scaling**: Use Qdrant cluster mode
2. **Caching**: Cache frequent queries
3. **Batch Operations**: Group indexing operations
4. **Index Optimization**: Regular collection optimization

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if Qdrant is running
   docker ps | grep qdrant
   curl http://localhost:6333/health
   ```

2. **Collections Not Found**
   ```bash
   # Restart application to recreate collections
   # Or manually create via API
   ```

3. **High Memory Usage**
   - Reduce batch sizes
   - Optimize collection settings
   - Consider quantization for large deployments

4. **Slow Searches**
   - Check collection size
   - Optimize similarity threshold
   - Consider approximate search settings

### Logs

Vector service logs are available in the application logs:

```bash
# Docker logs
docker logs ytempire-backend

# Search for vector-related logs
docker logs ytempire-backend 2>&1 | grep -i vector
```

## Development

### Testing

```bash
# Run vector service tests
pytest tests/test_vector_service.py

# Test search endpoints
pytest tests/test_search_endpoints.py
```

### Local Development

```bash
# Start Qdrant locally
docker-compose -f docker-compose.qdrant.yml up -d

# Set environment variables
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# Run application
uvicorn app.main:app --reload
```

## Security

### Production Considerations

1. **API Key Authentication**
   ```env
   QDRANT_API_KEY=secure-random-key
   ```

2. **Network Security**
   - Use private networks
   - Configure firewalls
   - Enable TLS for gRPC

3. **Data Privacy**
   - Encrypt sensitive embeddings
   - Regular security audits
   - Access logging

4. **Backup Strategy**
   - Regular volume backups
   - Collection snapshots
   - Disaster recovery plan

## Future Enhancements

### Planned Features

1. **Multi-modal Embeddings**
   - Image and video frame embeddings
   - Cross-modal search capabilities

2. **Advanced Analytics**
   - Content trend analysis
   - User behavior insights
   - Performance optimization

3. **Federated Search**
   - Search across multiple vector databases
   - Content federation

4. **Real-time Updates**
   - WebSocket notifications
   - Live recommendation updates

### Performance Improvements

1. **Quantization**
   - Reduce vector dimensions
   - Faster search with acceptable accuracy loss

2. **Approximate Search**
   - HNSW algorithm optimization
   - Trade accuracy for speed

3. **Caching Layer**
   - Redis for frequent queries
   - Result caching strategies

## Support

For issues with the vector database implementation:

1. Check this documentation
2. Review application logs
3. Check Qdrant documentation
4. File issues in the project repository

---

**Owner**: VP of AI  
**Last Updated**: 2025-08-10  
**Version**: 1.0.0