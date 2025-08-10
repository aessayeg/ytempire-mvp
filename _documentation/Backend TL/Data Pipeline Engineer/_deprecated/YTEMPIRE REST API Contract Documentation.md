# YTEMPIRE REST API Contract Documentation
**Version 1.0 | January 2025**  
**Owner: API Development Engineer**  
**Approved By: Backend Team Lead**  
**Status: Official API Contract**

---

## Executive Summary

This document defines the REST API contract for YTEMPIRE's automated YouTube content platform. It establishes the standards, patterns, and specifications that all API endpoints must follow to ensure consistency, reliability, and scalability across our platform serving 100+ concurrent users managing 500+ YouTube channels.

**Core API Principles:**
- **RESTful Design**: Follow REST architectural constraints
- **JSON First**: All requests and responses in JSON format
- **Versioned**: API versioning through URL path
- **Secure by Default**: All endpoints require authentication
- **Rate Limited**: Protect platform stability
- **Idempotent**: Safe retry mechanisms

---

## 1. API Design Standards

### 1.1 URL Structure

```
https://api.ytempire.com/{version}/{resource}/{resource-id}/{sub-resource}
```

**Examples:**
```
GET    https://api.ytempire.com/v1/channels
GET    https://api.ytempire.com/v1/channels/ch_123abc
POST   https://api.ytempire.com/v1/channels/ch_123abc/videos
GET    https://api.ytempire.com/v1/videos/vid_456def
PUT    https://api.ytempire.com/v1/videos/vid_456def/publish
```

### 1.2 Resource Naming Conventions

```yaml
naming_rules:
  collections: 
    - Use plural nouns: /channels, /videos, /analytics
    - Lowercase only: /subscriptions not /Subscriptions
    - Hyphenate multi-word: /content-templates
    
  identifiers:
    - Format: {resource_prefix}_{unique_id}
    - Examples:
      - Channel: ch_1234567890abcdef
      - Video: vid_0987654321fedcba
      - User: usr_abcdef1234567890
      - Template: tpl_fedcba0987654321
    
  actions:
    - Use resource-based URLs with HTTP verbs
    - For custom actions, use descriptive names:
      - POST /videos/{id}/generate-thumbnail
      - POST /channels/{id}/analyze-performance
```

### 1.3 HTTP Methods

| Method | Usage | Idempotent | Safe | Example |
|--------|-------|------------|------|---------|
| GET | Retrieve resource(s) | Yes | Yes | GET /channels |
| POST | Create new resource | No | No | POST /videos |
| PUT | Full update | Yes | No | PUT /channels/{id} |
| PATCH | Partial update | No | No | PATCH /videos/{id} |
| DELETE | Remove resource | Yes | No | DELETE /templates/{id} |

### 1.4 Standard HTTP Status Codes

```yaml
success_codes:
  200: OK - Successful GET, PUT, PATCH
  201: Created - Successful POST with new resource
  202: Accepted - Request accepted for async processing
  204: No Content - Successful DELETE

client_error_codes:
  400: Bad Request - Invalid request format
  401: Unauthorized - Missing or invalid authentication
  403: Forbidden - Valid auth but insufficient permissions
  404: Not Found - Resource doesn't exist
  409: Conflict - Resource state conflict
  422: Unprocessable Entity - Validation errors
  429: Too Many Requests - Rate limit exceeded

server_error_codes:
  500: Internal Server Error - Unexpected server error
  502: Bad Gateway - Upstream service error
  503: Service Unavailable - Temporary unavailability
  504: Gateway Timeout - Upstream service timeout
```

---

## 2. Request/Response Format

### 2.1 Standard Request Headers

```http
POST /v1/channels HTTP/1.1
Host: api.ytempire.com
Content-Type: application/json
Accept: application/json
Authorization: Bearer {jwt-token}
X-API-Version: 1.0
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
X-Client-ID: web-app-v2.1.0
```

### 2.2 Request Body Structure

```json
{
  "data": {
    "type": "channel",
    "attributes": {
      "name": "Tech Reviews Daily",
      "description": "Daily technology product reviews",
      "niche": "technology",
      "language": "en-US",
      "targetAudience": {
        "ageRange": "18-34",
        "interests": ["gadgets", "smartphones", "laptops"]
      }
    },
    "relationships": {
      "owner": {
        "type": "user",
        "id": "usr_1234567890abcdef"
      }
    }
  }
}
```

### 2.3 Response Envelope

```json
{
  "data": {
    "type": "channel",
    "id": "ch_0987654321fedcba",
    "attributes": {
      "name": "Tech Reviews Daily",
      "description": "Daily technology product reviews",
      "niche": "technology",
      "status": "active",
      "subscriberCount": 0,
      "videoCount": 0,
      "createdAt": "2025-01-15T10:30:00Z",
      "updatedAt": "2025-01-15T10:30:00Z"
    },
    "relationships": {
      "owner": {
        "type": "user",
        "id": "usr_1234567890abcdef"
      },
      "videos": {
        "links": {
          "related": "/v1/channels/ch_0987654321fedcba/videos"
        }
      }
    },
    "links": {
      "self": "/v1/channels/ch_0987654321fedcba"
    }
  },
  "meta": {
    "requestId": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "1.0"
  }
}
```

### 2.4 Error Response Format

```json
{
  "error": {
    "id": "err_1234567890abcdef",
    "status": 422,
    "code": "VALIDATION_ERROR",
    "title": "Validation Failed",
    "detail": "The request contains invalid data",
    "source": {
      "pointer": "/data/attributes/name",
      "parameter": "name"
    },
    "meta": {
      "timestamp": "2025-01-15T10:30:00Z",
      "requestId": "550e8400-e29b-41d4-a716-446655440000",
      "validationErrors": [
        {
          "field": "name",
          "code": "too_short",
          "message": "Channel name must be at least 3 characters"
        }
      ]
    }
  }
}
```

---

## 3. Pagination

### 3.1 Pagination Parameters

```yaml
query_parameters:
  page[size]: 
    - Number of items per page
    - Default: 20
    - Maximum: 100
    
  page[number]:
    - Page number (1-based)
    - Default: 1
    
  sort:
    - Comma-separated list of fields
    - Prefix with - for descending
    - Example: sort=-createdAt,name
    
  filter:
    - Field-specific filtering
    - Example: filter[status]=active
```

### 3.2 Paginated Response

```json
{
  "data": [...],
  "meta": {
    "pagination": {
      "total": 234,
      "count": 20,
      "perPage": 20,
      "currentPage": 2,
      "totalPages": 12
    }
  },
  "links": {
    "first": "/v1/channels?page[number]=1&page[size]=20",
    "prev": "/v1/channels?page[number]=1&page[size]=20",
    "self": "/v1/channels?page[number]=2&page[size]=20",
    "next": "/v1/channels?page[number]=3&page[size]=20",
    "last": "/v1/channels?page[number]=12&page[size]=20"
  }
}
```

---

## 4. Filtering and Searching

### 4.1 Filter Syntax

```http
GET /v1/videos?filter[status]=published&filter[channel]=ch_123abc&filter[createdAt][gte]=2025-01-01
```

**Supported Operators:**
- `[eq]` - Equals (default)
- `[ne]` - Not equals
- `[gt]` - Greater than
- `[gte]` - Greater than or equal
- `[lt]` - Less than
- `[lte]` - Less than or equal
- `[in]` - In array
- `[contains]` - Contains substring
- `[startsWith]` - Starts with
- `[endsWith]` - Ends with

### 4.2 Search Capabilities

```http
GET /v1/videos?search=technology+reviews&searchFields=title,description,tags
```

---

## 5. Bulk Operations

### 5.1 Bulk Create

```json
POST /v1/videos/bulk

{
  "data": [
    {
      "type": "video",
      "attributes": {
        "title": "iPhone 15 Review",
        "channelId": "ch_123abc"
      }
    },
    {
      "type": "video",
      "attributes": {
        "title": "Samsung S24 Review",
        "channelId": "ch_123abc"
      }
    }
  ]
}
```

### 5.2 Bulk Update

```json
PATCH /v1/videos/bulk

{
  "data": [
    {
      "type": "video",
      "id": "vid_123",
      "attributes": {
        "status": "published"
      }
    },
    {
      "type": "video",
      "id": "vid_456",
      "attributes": {
        "status": "published"
      }
    }
  ]
}
```

---

## 6. Async Operations

### 6.1 Async Request

```json
POST /v1/videos/generate
Content-Type: application/json
Prefer: respond-async

{
  "data": {
    "type": "video-generation",
    "attributes": {
      "channelId": "ch_123abc",
      "topic": "iPhone 15 Pro Review",
      "style": "detailed-review",
      "duration": "10-12 minutes"
    }
  }
}

Response:
HTTP/1.1 202 Accepted
Location: /v1/jobs/job_789xyz
```

### 6.2 Job Status Polling

```json
GET /v1/jobs/job_789xyz

{
  "data": {
    "type": "job",
    "id": "job_789xyz",
    "attributes": {
      "status": "processing",
      "progress": 45,
      "createdAt": "2025-01-15T10:30:00Z",
      "estimatedCompletion": "2025-01-15T10:40:00Z"
    },
    "relationships": {
      "result": {
        "type": "video",
        "id": "vid_pending"
      }
    }
  }
}
```

---

## 7. Versioning Strategy

### 7.1 Version in URL Path

```yaml
versioning_rules:
  current_version: v1
  format: /v{major_version}/
  
  examples:
    - /v1/channels
    - /v2/channels  # Future version
    
  deprecation_policy:
    - Minimum 6 months notice
    - Sunset headers in responses
    - Migration guides provided
```

### 7.2 Version Negotiation

```http
# Explicit version in header (optional)
X-API-Version: 1.0

# Response includes version info
X-API-Version: 1.0
X-API-Deprecated: false
X-API-Sunset: 2026-01-01  # If applicable
```

---

## 8. Field Selection

### 8.1 Sparse Fieldsets

```http
GET /v1/channels/ch_123abc?fields[channel]=name,status,subscriberCount
```

### 8.2 Including Related Resources

```http
GET /v1/channels/ch_123abc?include=videos,analytics&fields[videos]=title,status
```

---

## 9. Data Types and Formats

### 9.1 Standard Data Types

```yaml
data_types:
  strings:
    - UTF-8 encoded
    - Max length specified per field
    
  numbers:
    - Integers: 64-bit signed
    - Decimals: IEEE 754 double precision
    - Money: Integer cents (e.g., 1999 = $19.99)
    
  dates:
    - ISO 8601 format
    - UTC timezone
    - Example: 2025-01-15T10:30:00Z
    
  booleans:
    - true/false (lowercase)
    
  arrays:
    - JSON arrays
    - Homogeneous types preferred
    
  objects:
    - JSON objects
    - Nested depth limit: 5 levels
```

### 9.2 Common Field Formats

```yaml
field_formats:
  id: 
    pattern: ^[a-z]+_[a-f0-9]{16}$
    example: ch_1234567890abcdef
    
  email:
    pattern: RFC 5322
    example: user@example.com
    
  url:
    pattern: Valid HTTP(S) URL
    example: https://youtube.com/watch?v=abc123
    
  currency:
    pattern: ISO 4217
    example: USD
    
  language:
    pattern: ISO 639-1
    example: en
    
  country:
    pattern: ISO 3166-1 alpha-2
    example: US
```

---

## 10. API Contract Testing

### 10.1 Contract Test Example

```python
import pytest
from jsonschema import validate

class TestChannelAPIContract:
    """Contract tests for Channel API"""
    
    def test_create_channel_request_schema(self):
        """Validate create channel request format"""
        schema = {
            "type": "object",
            "required": ["data"],
            "properties": {
                "data": {
                    "type": "object",
                    "required": ["type", "attributes"],
                    "properties": {
                        "type": {"enum": ["channel"]},
                        "attributes": {
                            "type": "object",
                            "required": ["name", "niche"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "minLength": 3,
                                    "maxLength": 100
                                },
                                "niche": {
                                    "type": "string",
                                    "enum": ["technology", "gaming", "education", "lifestyle"]
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Test valid request
        valid_request = {
            "data": {
                "type": "channel",
                "attributes": {
                    "name": "Tech Reviews",
                    "niche": "technology"
                }
            }
        }
        
        validate(instance=valid_request, schema=schema)
```

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: API Development Engineer
- **Review Cycle**: Bi-weekly
- **Approval**: Backend Team Lead

**Change Log:**
- v1.0 - Initial REST API contract definition

---