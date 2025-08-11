import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const videoGenTime = new Trend('video_generation_time');
const websocketConnections = new Gauge('websocket_connections');
const successfulVideos = new Counter('successful_videos');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const WS_URL = __ENV.WS_URL || 'ws://localhost:8000';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '1m',
      tags: { test_type: 'smoke' },
    },
    
    // Load test
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },  // Ramp up
        { duration: '5m', target: 10 },  // Stay at 10 users
        { duration: '2m', target: 20 },  // Ramp to 20
        { duration: '5m', target: 20 },  // Stay at 20
        { duration: '2m', target: 0 },   // Ramp down
      ],
      tags: { test_type: 'load' },
      startTime: '10m',
    },
    
    // Stress test
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 50 },
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 150 },
        { duration: '5m', target: 150 },
        { duration: '5m', target: 0 },
      ],
      tags: { test_type: 'stress' },
      startTime: '30m',
    },
    
    // Spike test
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 100 },
        { duration: '1m', target: 100 },
        { duration: '10s', target: 0 },
      ],
      tags: { test_type: 'spike' },
      startTime: '55m',
    },
    
    // Soak test (endurance)
    soak: {
      executor: 'constant-vus',
      vus: 20,
      duration: '30m',
      tags: { test_type: 'soak' },
      startTime: '60m',
    },
  },
  
  thresholds: {
    'http_req_duration': ['p(95)<500', 'p(99)<1000'],
    'http_req_failed': ['rate<0.1'],
    'errors': ['rate<0.05'],
    'api_latency': ['p(95)<300'],
    'video_generation_time': ['p(95)<600000'], // 10 minutes
  },
};

// Helper functions
function getAuthToken() {
  const loginRes = http.post(`${BASE_URL}/api/v1/auth/login`, JSON.stringify({
    email: 'test@ytempire.com',
    password: 'testpassword123'
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(loginRes, {
    'login successful': (r) => r.status === 200,
  });
  
  if (loginRes.status === 200) {
    const token = loginRes.json('access_token');
    return token;
  }
  return null;
}

function getAuthHeaders(token) {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };
}

// Test scenarios
export function setup() {
  // Setup code - create test user
  const signupRes = http.post(`${BASE_URL}/api/v1/auth/signup`, JSON.stringify({
    email: 'test@ytempire.com',
    password: 'testpassword123',
    full_name: 'Test User'
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const token = getAuthToken();
  return { token };
}

export default function (data) {
  const token = data.token;
  const headers = getAuthHeaders(token);
  
  // Test groups
  group('Authentication Tests', () => {
    const startTime = Date.now();
    
    const res = http.get(`${BASE_URL}/api/v1/users/me`, { headers });
    
    apiLatency.add(Date.now() - startTime);
    
    check(res, {
      'user profile retrieved': (r) => r.status === 200,
      'has user data': (r) => r.json('email') !== undefined,
    });
    
    errorRate.add(res.status !== 200);
  });
  
  group('Channel Operations', () => {
    // Create channel
    const createChannelRes = http.post(
      `${BASE_URL}/api/v1/channels`,
      JSON.stringify({
        name: `Test Channel ${randomString(8)}`,
        description: 'Performance test channel',
        youtube_channel_id: `UC${randomString(22)}`,
      }),
      { headers }
    );
    
    check(createChannelRes, {
      'channel created': (r) => r.status === 201,
    });
    
    if (createChannelRes.status === 201) {
      const channelId = createChannelRes.json('id');
      
      // Get channel details
      const getChannelRes = http.get(
        `${BASE_URL}/api/v1/channels/${channelId}`,
        { headers }
      );
      
      check(getChannelRes, {
        'channel retrieved': (r) => r.status === 200,
      });
      
      // Update channel
      const updateChannelRes = http.put(
        `${BASE_URL}/api/v1/channels/${channelId}`,
        JSON.stringify({
          description: 'Updated description',
        }),
        { headers }
      );
      
      check(updateChannelRes, {
        'channel updated': (r) => r.status === 200,
      });
    }
  });
  
  group('Video Generation', () => {
    const startTime = Date.now();
    
    const videoRes = http.post(
      `${BASE_URL}/api/v1/videos/generate`,
      JSON.stringify({
        title: `Test Video ${randomString(10)}`,
        topic: 'Technology trends',
        duration: randomIntBetween(300, 600),
        style: 'educational',
        channel_id: 'test-channel-1',
      }),
      { headers }
    );
    
    const genTime = Date.now() - startTime;
    videoGenTime.add(genTime);
    
    check(videoRes, {
      'video generation started': (r) => r.status === 202,
      'has video id': (r) => r.json('video_id') !== undefined,
    });
    
    if (videoRes.status === 202) {
      successfulVideos.add(1);
      const videoId = videoRes.json('video_id');
      
      // Poll for status
      for (let i = 0; i < 10; i++) {
        sleep(2);
        const statusRes = http.get(
          `${BASE_URL}/api/v1/videos/${videoId}/status`,
          { headers }
        );
        
        if (statusRes.json('status') === 'completed') {
          break;
        }
      }
    }
    
    errorRate.add(videoRes.status !== 202);
  });
  
  group('Analytics Queries', () => {
    // Revenue analytics
    const revenueRes = http.get(
      `${BASE_URL}/api/v1/analytics/revenue?period=7d`,
      { headers }
    );
    
    check(revenueRes, {
      'revenue data retrieved': (r) => r.status === 200,
    });
    
    // View analytics
    const viewsRes = http.get(
      `${BASE_URL}/api/v1/analytics/views?period=30d`,
      { headers }
    );
    
    check(viewsRes, {
      'views data retrieved': (r) => r.status === 200,
    });
    
    // Performance metrics
    const perfRes = http.get(
      `${BASE_URL}/api/v1/analytics/performance`,
      { headers }
    );
    
    check(perfRes, {
      'performance metrics retrieved': (r) => r.status === 200,
    });
  });
  
  group('Search and Filter', () => {
    // Search videos
    const searchRes = http.get(
      `${BASE_URL}/api/v1/videos/search?q=technology&limit=10`,
      { headers }
    );
    
    check(searchRes, {
      'search results returned': (r) => r.status === 200,
      'has results': (r) => Array.isArray(r.json('results')),
    });
    
    // Filter by date
    const filterRes = http.get(
      `${BASE_URL}/api/v1/videos?start_date=2024-01-01&end_date=2024-12-31`,
      { headers }
    );
    
    check(filterRes, {
      'filtered results returned': (r) => r.status === 200,
    });
  });
  
  group('Concurrent Operations', () => {
    const requests = [
      ['GET', `${BASE_URL}/api/v1/channels`],
      ['GET', `${BASE_URL}/api/v1/videos`],
      ['GET', `${BASE_URL}/api/v1/analytics/summary`],
      ['GET', `${BASE_URL}/api/v1/users/me`],
    ];
    
    const responses = http.batch(
      requests.map(([method, url]) => ({
        method,
        url,
        params: { headers },
      }))
    );
    
    responses.forEach((res, index) => {
      check(res, {
        [`request ${index} successful`]: (r) => r.status === 200,
      });
    });
  });
  
  sleep(randomIntBetween(1, 3));
}

export function teardown(data) {
  // Cleanup code
  console.log('Test completed');
}

// WebSocket test (separate scenario)
export function websocketTest() {
  const url = `${WS_URL}/ws/test-client-${randomString(8)}`;
  const params = { tags: { test_type: 'websocket' } };
  
  const res = ws.connect(url, params, function (socket) {
    socket.on('open', () => {
      websocketConnections.add(1);
      
      // Send test messages
      socket.send(JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString(),
      }));
      
      socket.setInterval(() => {
        socket.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: new Date().toISOString(),
        }));
      }, 5000);
    });
    
    socket.on('message', (data) => {
      const message = JSON.parse(data);
      check(message, {
        'valid message format': (m) => m.type !== undefined,
      });
    });
    
    socket.on('close', () => {
      websocketConnections.add(-1);
    });
    
    socket.on('error', (e) => {
      console.error('WebSocket error:', e);
      errorRate.add(1);
    });
    
    // Keep connection open for test duration
    socket.setTimeout(() => {
      socket.close();
    }, 60000);
  });
  
  check(res, {
    'websocket connected': (r) => r && r.status === 101,
  });
}

// Custom summary
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'summary.json': JSON.stringify(data),
    'summary.html': htmlReport(data),
  };
}