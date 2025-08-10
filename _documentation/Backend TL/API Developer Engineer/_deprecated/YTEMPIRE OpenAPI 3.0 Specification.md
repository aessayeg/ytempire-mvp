# YTEMPIRE OpenAPI 3.0 Specification
**Version 1.0 | January 2025**  
**Owner: API Development Engineer**  
**Format: OpenAPI 3.0.3**

---

## Complete OpenAPI Specification

```yaml
openapi: 3.0.3
info:
  title: YTEMPIRE API
  description: |
    YTEMPIRE REST API for automated YouTube content creation and channel management.
    
    ## Overview
    The YTEMPIRE API enables automated creation and management of YouTube channels,
    video generation, analytics tracking, and monetization optimization.
    
    ## Authentication
    All endpoints require JWT Bearer token authentication.
    
    ## Rate Limiting
    - Standard tier: 100 requests/minute
    - Premium tier: 1000 requests/minute
    
    ## Support
    - Email: api-support@ytempire.com
    - Documentation: https://docs.ytempire.com
  version: 1.0.0
  contact:
    name: YTEMPIRE API Support
    email: api-support@ytempire.com
    url: https://support.ytempire.com
  license:
    name: Proprietary
    url: https://ytempire.com/terms

servers:
  - url: https://api.ytempire.com/v1
    description: Production server
  - url: https://staging-api.ytempire.com/v1
    description: Staging server
  - url: http://localhost:8000/v1
    description: Local development

tags:
  - name: Authentication
    description: User authentication and authorization
  - name: Channels
    description: YouTube channel management
  - name: Videos
    description: Video generation and management
  - name: Analytics
    description: Performance analytics and insights
  - name: Templates
    description: Content templates management
  - name: Webhooks
    description: Webhook subscriptions
  - name: Jobs
    description: Async job management

paths:
  # Authentication Endpoints
  /auth/login:
    post:
      tags:
        - Authentication
      summary: User login
      description: Authenticate user and receive JWT tokens
      operationId: login
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/LoginRequest'
      responses:
        '200':
          description: Successful authentication
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '422':
          $ref: '#/components/responses/ValidationError'
      security: []

  /auth/refresh:
    post:
      tags:
        - Authentication
      summary: Refresh access token
      description: Exchange refresh token for new access token
      operationId: refreshToken
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RefreshTokenRequest'
      responses:
        '200':
          description: Token refreshed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
      security: []

  /auth/logout:
    post:
      tags:
        - Authentication
      summary: User logout
      description: Invalidate current tokens
      operationId: logout
      responses:
        '204':
          description: Logout successful
        '401':
          $ref: '#/components/responses/UnauthorizedError'

  # Channel Endpoints
  /channels:
    get:
      tags:
        - Channels
      summary: List channels
      description: Get paginated list of user's channels
      operationId: listChannels
      parameters:
        - $ref: '#/components/parameters/PageSize'
        - $ref: '#/components/parameters/PageNumber'
        - $ref: '#/components/parameters/Sort'
        - name: filter[status]
          in: query
          schema:
            type: string
            enum: [active, paused, suspended]
        - name: filter[niche]
          in: query
          schema:
            type: string
      responses:
        '200':
          description: Channel list retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Channel'
                  meta:
                    $ref: '#/components/schemas/PaginationMeta'
                  links:
                    $ref: '#/components/schemas/PaginationLinks'
        '401':
          $ref: '#/components/responses/UnauthorizedError'

    post:
      tags:
        - Channels
      summary: Create channel
      description: Create a new YouTube channel
      operationId: createChannel
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - data
              properties:
                data:
                  $ref: '#/components/schemas/ChannelCreate'
      responses:
        '201':
          description: Channel created successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Channel'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '422':
          $ref: '#/components/responses/ValidationError'

  /channels/{channelId}:
    parameters:
      - $ref: '#/components/parameters/ChannelId'
    
    get:
      tags:
        - Channels
      summary: Get channel details
      description: Retrieve specific channel information
      operationId: getChannel
      parameters:
        - name: include
          in: query
          description: Include related resources
          schema:
            type: string
            example: videos,analytics
      responses:
        '200':
          description: Channel details retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Channel'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '404':
          $ref: '#/components/responses/NotFoundError'

    put:
      tags:
        - Channels
      summary: Update channel
      description: Update channel information
      operationId: updateChannel
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                data:
                  $ref: '#/components/schemas/ChannelUpdate'
      responses:
        '200':
          description: Channel updated successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Channel'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '404':
          $ref: '#/components/responses/NotFoundError'
        '422':
          $ref: '#/components/responses/ValidationError'

    delete:
      tags:
        - Channels
      summary: Delete channel
      description: Soft delete a channel
      operationId: deleteChannel
      responses:
        '204':
          description: Channel deleted successfully
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '404':
          $ref: '#/components/responses/NotFoundError'

  # Video Endpoints
  /videos:
    get:
      tags:
        - Videos
      summary: List all videos
      description: Get paginated list of all user's videos
      operationId: listAllVideos
      parameters:
        - $ref: '#/components/parameters/PageSize'
        - $ref: '#/components/parameters/PageNumber'
        - $ref: '#/components/parameters/Sort'
        - name: filter[status]
          in: query
          schema:
            type: string
            enum: [draft, processing, published, failed]
        - name: filter[channelId]
          in: query
          schema:
            type: string
      responses:
        '200':
          description: Video list retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Video'
                  meta:
                    $ref: '#/components/schemas/PaginationMeta'
                  links:
                    $ref: '#/components/schemas/PaginationLinks'

  /channels/{channelId}/videos:
    parameters:
      - $ref: '#/components/parameters/ChannelId'
    
    get:
      tags:
        - Videos
      summary: List channel videos
      description: Get videos for specific channel
      operationId: listChannelVideos
      parameters:
        - $ref: '#/components/parameters/PageSize'
        - $ref: '#/components/parameters/PageNumber'
        - $ref: '#/components/parameters/Sort'
      responses:
        '200':
          description: Video list retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Video'
                  meta:
                    $ref: '#/components/schemas/PaginationMeta'

    post:
      tags:
        - Videos
      summary: Create video
      description: Generate a new video for the channel
      operationId: createVideo
      parameters:
        - name: Prefer
          in: header
          description: Request async processing
          schema:
            type: string
            enum: [respond-async]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                data:
                  $ref: '#/components/schemas/VideoCreate'
      responses:
        '201':
          description: Video created (sync)
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Video'
        '202':
          description: Video generation started (async)
          headers:
            Location:
              description: Job status URL
              schema:
                type: string
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Job'

  /videos/{videoId}:
    parameters:
      - $ref: '#/components/parameters/VideoId'
    
    get:
      tags:
        - Videos
      summary: Get video details
      description: Retrieve specific video information
      operationId: getVideo
      responses:
        '200':
          description: Video details retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Video'
        '404':
          $ref: '#/components/responses/NotFoundError'

    patch:
      tags:
        - Videos
      summary: Update video
      description: Update video metadata
      operationId: updateVideo
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                data:
                  $ref: '#/components/schemas/VideoUpdate'
      responses:
        '200':
          description: Video updated successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Video'

  /videos/{videoId}/publish:
    parameters:
      - $ref: '#/components/parameters/VideoId'
    
    post:
      tags:
        - Videos
      summary: Publish video
      description: Publish video to YouTube
      operationId: publishVideo
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                scheduledAt:
                  type: string
                  format: date-time
                  description: Optional scheduled publish time
      responses:
        '202':
          description: Publish job started
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Job'

  # Analytics Endpoints
  /analytics/channels/{channelId}:
    parameters:
      - $ref: '#/components/parameters/ChannelId'
    
    get:
      tags:
        - Analytics
      summary: Get channel analytics
      description: Retrieve channel performance metrics
      operationId: getChannelAnalytics
      parameters:
        - name: startDate
          in: query
          required: true
          schema:
            type: string
            format: date
        - name: endDate
          in: query
          required: true
          schema:
            type: string
            format: date
        - name: metrics
          in: query
          schema:
            type: array
            items:
              type: string
              enum: [views, subscribers, revenue, watchTime]
      responses:
        '200':
          description: Analytics data retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Analytics'

  # Template Endpoints
  /templates:
    get:
      tags:
        - Templates
      summary: List templates
      description: Get available content templates
      operationId: listTemplates
      parameters:
        - name: filter[type]
          in: query
          schema:
            type: string
            enum: [video, thumbnail, description]
        - name: filter[niche]
          in: query
          schema:
            type: string
      responses:
        '200':
          description: Template list retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Template'

  # Webhook Endpoints
  /webhooks:
    get:
      tags:
        - Webhooks
      summary: List webhook subscriptions
      description: Get all webhook subscriptions
      operationId: listWebhooks
      responses:
        '200':
          description: Webhook list retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Webhook'

    post:
      tags:
        - Webhooks
      summary: Create webhook
      description: Subscribe to webhook events
      operationId: createWebhook
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                data:
                  $ref: '#/components/schemas/WebhookCreate'
      responses:
        '201':
          description: Webhook created
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Webhook'

  # Job Endpoints
  /jobs/{jobId}:
    parameters:
      - $ref: '#/components/parameters/JobId'
    
    get:
      tags:
        - Jobs
      summary: Get job status
      description: Check async job status
      operationId: getJobStatus
      responses:
        '200':
          description: Job status retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Job'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  parameters:
    PageSize:
      name: page[size]
      in: query
      description: Number of items per page
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 20

    PageNumber:
      name: page[number]
      in: query
      description: Page number
      schema:
        type: integer
        minimum: 1
        default: 1

    Sort:
      name: sort
      in: query
      description: Sort order (prefix with - for descending)
      schema:
        type: string
        example: -createdAt

    ChannelId:
      name: channelId
      in: path
      required: true
      description: Channel identifier
      schema:
        type: string
        pattern: '^ch_[a-f0-9]{16}$'

    VideoId:
      name: videoId
      in: path
      required: true
      description: Video identifier
      schema:
        type: string
        pattern: '^vid_[a-f0-9]{16}$'

    JobId:
      name: jobId
      in: path
      required: true
      description: Job identifier
      schema:
        type: string
        pattern: '^job_[a-f0-9]{16}$'

  schemas:
    # Authentication Schemas
    LoginRequest:
      type: object
      required:
        - email
        - password
      properties:
        email:
          type: string
          format: email
        password:
          type: string
          format: password
          minLength: 8

    RefreshTokenRequest:
      type: object
      required:
        - refreshToken
      properties:
        refreshToken:
          type: string

    AuthResponse:
      type: object
      properties:
        data:
          type: object
          properties:
            type:
              type: string
              enum: [auth]
            attributes:
              type: object
              properties:
                accessToken:
                  type: string
                refreshToken:
                  type: string
                expiresIn:
                  type: integer
                  description: Seconds until expiration
                tokenType:
                  type: string
                  enum: [Bearer]
            relationships:
              type: object
              properties:
                user:
                  type: object
                  properties:
                    type:
                      type: string
                      enum: [user]
                    id:
                      type: string

    # Channel Schemas
    Channel:
      type: object
      properties:
        type:
          type: string
          enum: [channel]
        id:
          type: string
          pattern: '^ch_[a-f0-9]{16}$'
        attributes:
          type: object
          properties:
            name:
              type: string
              minLength: 3
              maxLength: 100
            description:
              type: string
              maxLength: 1000
            niche:
              type: string
              enum: [technology, gaming, education, lifestyle, finance, health]
            status:
              type: string
              enum: [active, paused, suspended]
            youtubeChannelId:
              type: string
            subscriberCount:
              type: integer
            videoCount:
              type: integer
            totalViews:
              type: integer
            monthlyRevenue:
              type: number
              format: float
            automationEnabled:
              type: boolean
            settings:
              type: object
              properties:
                uploadSchedule:
                  type: object
                  properties:
                    frequency:
                      type: string
                      enum: [daily, weekly, biweekly]
                    times:
                      type: array
                      items:
                        type: string
                        format: time
                defaultVideoLength:
                  type: string
                  enum: [short, medium, long]
                monetizationEnabled:
                  type: boolean
            createdAt:
              type: string
              format: date-time
            updatedAt:
              type: string
              format: date-time
        relationships:
          type: object
          properties:
            owner:
              type: object
              properties:
                type:
                  type: string
                  enum: [user]
                id:
                  type: string
            videos:
              type: object
              properties:
                links:
                  type: object
                  properties:
                    related:
                      type: string
        links:
          type: object
          properties:
            self:
              type: string

    ChannelCreate:
      type: object
      required:
        - type
        - attributes
      properties:
        type:
          type: string
          enum: [channel]
        attributes:
          type: object
          required:
            - name
            - niche
          properties:
            name:
              type: string
              minLength: 3
              maxLength: 100
            description:
              type: string
              maxLength: 1000
            niche:
              type: string
              enum: [technology, gaming, education, lifestyle, finance, health]
            targetAudience:
              type: object
              properties:
                ageRange:
                  type: string
                interests:
                  type: array
                  items:
                    type: string
            settings:
              $ref: '#/components/schemas/Channel/properties/attributes/properties/settings'

    ChannelUpdate:
      type: object
      required:
        - type
        - id
      properties:
        type:
          type: string
          enum: [channel]
        id:
          type: string
        attributes:
          type: object
          properties:
            name:
              type: string
            description:
              type: string
            status:
              type: string
              enum: [active, paused]
            settings:
              $ref: '#/components/schemas/Channel/properties/attributes/properties/settings'

    # Video Schemas
    Video:
      type: object
      properties:
        type:
          type: string
          enum: [video]
        id:
          type: string
          pattern: '^vid_[a-f0-9]{16}$'
        attributes:
          type: object
          properties:
            title:
              type: string
            description:
              type: string
            tags:
              type: array
              items:
                type: string
            status:
              type: string
              enum: [draft, processing, published, failed]
            youtubeVideoId:
              type: string
            thumbnailUrl:
              type: string
              format: uri
            duration:
              type: integer
              description: Duration in seconds
            publishedAt:
              type: string
              format: date-time
            metrics:
              type: object
              properties:
                views:
                  type: integer
                likes:
                  type: integer
                comments:
                  type: integer
                watchTime:
                  type: integer
            generationCost:
              type: number
              format: float
            createdAt:
              type: string
              format: date-time
            updatedAt:
              type: string
              format: date-time
        relationships:
          type: object
          properties:
            channel:
              type: object
              properties:
                type:
                  type: string
                  enum: [channel]
                id:
                  type: string

    VideoCreate:
      type: object
      required:
        - type
        - attributes
      properties:
        type:
          type: string
          enum: [video]
        attributes:
          type: object
          required:
            - topic
          properties:
            topic:
              type: string
            style:
              type: string
              enum: [review, tutorial, news, entertainment]
            duration:
              type: string
              enum: [1-3, 3-5, 5-10, 10-15]
            templateId:
              type: string
            customPrompt:
              type: string
            scheduledPublishAt:
              type: string
              format: date-time

    VideoUpdate:
      type: object
      required:
        - type
        - id
      properties:
        type:
          type: string
          enum: [video]
        id:
          type: string
        attributes:
          type: object
          properties:
            title:
              type: string
            description:
              type: string
            tags:
              type: array
              items:
                type: string
            thumbnailUrl:
              type: string
              format: uri

    # Analytics Schemas
    Analytics:
      type: object
      properties:
        type:
          type: string
          enum: [analytics]
        attributes:
          type: object
          properties:
            startDate:
              type: string
              format: date
            endDate:
              type: string
              format: date
            metrics:
              type: object
              properties:
                views:
                  type: object
                  properties:
                    total:
                      type: integer
                    average:
                      type: number
                    change:
                      type: number
                subscribers:
                  type: object
                  properties:
                    gained:
                      type: integer
                    lost:
                      type: integer
                    net:
                      type: integer
                revenue:
                  type: object
                  properties:
                    total:
                      type: number
                    adRevenue:
                      type: number
                    affiliateRevenue:
                      type: number
                watchTime:
                  type: object
                  properties:
                    total:
                      type: integer
                    average:
                      type: number

    # Template Schemas
    Template:
      type: object
      properties:
        type:
          type: string
          enum: [template]
        id:
          type: string
          pattern: '^tpl_[a-f0-9]{16}$'
        attributes:
          type: object
          properties:
            name:
              type: string
            type:
              type: string
              enum: [video, thumbnail, description]
            niche:
              type: string
            content:
              type: string
            variables:
              type: array
              items:
                type: string
            isPublic:
              type: boolean

    # Webhook Schemas
    Webhook:
      type: object
      properties:
        type:
          type: string
          enum: [webhook]
        id:
          type: string
          pattern: '^wh_[a-f0-9]{16}$'
        attributes:
          type: object
          properties:
            url:
              type: string
              format: uri
            events:
              type: array
              items:
                type: string
                enum: [video.created, video.published, video.failed, channel.suspended]
            secret:
              type: string
            isActive:
              type: boolean
            lastDeliveredAt:
              type: string
              format: date-time

    WebhookCreate:
      type: object
      required:
        - type
        - attributes
      properties:
        type:
          type: string
          enum: [webhook]
        attributes:
          type: object
          required:
            - url
            - events
          properties:
            url:
              type: string
              format: uri
            events:
              type: array
              items:
                type: string
              minItems: 1

    # Job Schemas
    Job:
      type: object
      properties:
        type:
          type: string
          enum: [job]
        id:
          type: string
          pattern: '^job_[a-f0-9]{16}$'
        attributes:
          type: object
          properties:
            status:
              type: string
              enum: [pending, processing, completed, failed]
            progress:
              type: integer
              minimum: 0
              maximum: 100
            resultType:
              type: string
            resultId:
              type: string
            error:
              type: object
              properties:
                code:
                  type: string
                message:
                  type: string
            createdAt:
              type: string
              format: date-time
            startedAt:
              type: string
              format: date-time
            completedAt:
              type: string
              format: date-time
            estimatedCompletion:
              type: string
              format: date-time

    # Common Schemas
    Error:
      type: object
      properties:
        id:
          type: string
        status:
          type: integer
        code:
          type: string
        title:
          type: string
        detail:
          type: string
        source:
          type: object
          properties:
            pointer:
              type: string
            parameter:
              type: string
        meta:
          type: object

    PaginationMeta:
      type: object
      properties:
        pagination:
          type: object
          properties:
            total:
              type: integer
            count:
              type: integer
            perPage:
              type: integer
            currentPage:
              type: integer
            totalPages:
              type: integer

    PaginationLinks:
      type: object
      properties:
        first:
          type: string
        prev:
          type: string
        self:
          type: string
        next:
          type: string
        last:
          type: string

  responses:
    UnauthorizedError:
      description: Authentication required
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                $ref: '#/components/schemas/Error'
          example:
            error:
              status: 401
              code: UNAUTHORIZED
              title: Authentication Required
              detail: Please provide valid authentication credentials

    ForbiddenError:
      description: Insufficient permissions
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                $ref: '#/components/schemas/Error'

    NotFoundError:
      description: Resource not found
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                $ref: '#/components/schemas/Error'

    ValidationError:
      description: Validation failed
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                $ref: '#/components/schemas/Error'

    RateLimitError:
      description: Rate limit exceeded
      headers:
        X-RateLimit-Limit:
          schema:
            type: integer
        X-RateLimit-Remaining:
          schema:
            type: integer
        X-RateLimit-Reset:
          schema:
            type: integer
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                $ref: '#/components/schemas/Error'

security:
  - bearerAuth: []
```

---

## OpenAPI Implementation Guide

### 1. Code Generation

```bash
# Generate Python FastAPI server
openapi-generator generate \
  -i ytempire-openapi.yaml \
  -g python-fastapi \
  -o ./backend/generated

# Generate TypeScript client
openapi-generator generate \
  -i ytempire-openapi.yaml \
  -g typescript-axios \
  -o ./frontend/src/api/generated

# Generate API documentation
openapi-generator generate \
  -i ytempire-openapi.yaml \
  -g html2 \
  -o ./docs/api
```

### 2. Validation Middleware

```python
from fastapi import FastAPI, Request
from openapi_core import create_spec
from openapi_core.validation.request.validators import RequestValidator
from openapi_core.validation.response.validators import ResponseValidator

app = FastAPI()

# Load OpenAPI spec
with open('ytempire-openapi.yaml', 'r') as f:
    spec_dict = yaml.safe_load(f)
    spec = create_spec(spec_dict)

@app.middleware("http")
async def validate_openapi(request: Request, call_next):
    """Validate requests and responses against OpenAPI spec"""
    
    # Validate request
    request_validator = RequestValidator(spec)
    result = request_validator.validate(request)
    
    if result.errors:
        return JSONResponse(
            status_code=400,
            content={"error": "Request validation failed", "details": str(result.errors)}
        )
    
    # Process request
    response = await call_next(request)
    
    # Validate response in development
    if settings.ENVIRONMENT == "development":
        response_validator = ResponseValidator(spec)
        # Validation logic here
    
    return response
```

### 3. API Documentation UI

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    # Load our OpenAPI spec
    with open('ytempire-openapi.yaml', 'r') as f:
        openapi_schema = yaml.safe_load(f)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Documentation will be available at:
# - /docs (Swagger UI)
# - /redoc (ReDoc)
```

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: API Development Engineer
- **Format**: OpenAPI 3.0.3
- **Validation**: Automated via CI/CD

**Note**: This specification serves as the single source of truth for API contracts. Any changes must be reviewed and approved by the Backend Team Lead.