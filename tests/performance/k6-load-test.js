/**
 * k6 Load Testing Script for YTEmpire API
 * Performance testing setup
 */
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { randomString, randomItem, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const apiDuration = new Trend('api_duration');
const videoCreations = new Counter('video_creations');
const activeSessions = new Gauge('active_sessions');

// Test configuration
export const options = {
  scenarios: {
    // Smoke test - minimal load
    smoke: {
      executor: 'constant-vus',
      vus: 2,
      duration: '1m',
      tags: { test_type: 'smoke' },
    },
    
    // Load test - normal load
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
      gracefulRampDown: '30s',
      tags: { test_type: 'load' },
    },
    
    // Stress test - beyond normal load
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 },
        { duration: '5m', target: 20 },
        { duration: '2m', target: 40 },
        { duration: '5m', target: 40 },
        { duration: '2m', target: 60 },
        { duration: '5m', target: 60 },
        { duration: '2m', target: 0 },
      ],
      gracefulRampDown: '30s',
      tags: { test_type: 'stress' },
    },
    
    // Spike test - sudden load increase
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 5 },
        { duration: '1m', target: 5 },
        { duration: '10s', target: 100 },  // Spike!
        { duration: '3m', target: 100 },
        { duration: '10s', target: 5 },
        { duration: '3m', target: 5 },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '30s',
      tags: { test_type: 'spike' },
    },
    
    // Soak test - extended duration
    soak: {
      executor: 'constant-vus',
      vus: 10,
      duration: '30m',
      tags: { test_type: 'soak' },
    },
  },
  
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // 95% of requests under 500ms
    http_req_failed: ['rate<0.1'],  // Error rate under 10%
    errors: ['rate<0.1'],
    api_duration: ['p(95)<1000'],
  },
};

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_PREFIX = '/api/v1';

// Test data
const testUsers = [
  { email: 'test1@example.com', password: 'Test123!@#' },
  { email: 'test2@example.com', password: 'Test123!@#' },
  { email: 'test3@example.com', password: 'Test123!@#' },
];

const videoTitles = [
  'Top 10 JavaScript Frameworks',
  'Ultimate Gaming PC Build',
  'Easy Recipe Tutorial',
  'Travel Vlog Episode',
  'Tech Review 2024',
];

const categories = ['Technology', 'Gaming', 'Food', 'Travel', 'Education'];

// Helper functions
function authenticate() {
  const user = randomItem(testUsers);
  const loginRes = http.post(
    `${BASE_URL}${API_PREFIX}/auth/login`,
    JSON.stringify({
      email: user.email,
      password: user.password,
    }),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );
  
  check(loginRes, {
    'login successful': (r) => r.status === 200,
    'token received': (r) => r.json('access_token') !== '',
  });
  
  if (loginRes.status === 200) {
    return loginRes.json('access_token');
  }
  return null;
}

function makeAuthHeaders(token) {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };
}

// Test scenarios
export default function () {
  const token = authenticate();
  if (!token) {
    errorRate.add(1);
    return;
  }
  
  activeSessions.add(1);
  const headers = makeAuthHeaders(token);
  
  // Dashboard requests
  group('Dashboard', () => {
    const start = Date.now();
    
    const dashboardRes = http.get(
      `${BASE_URL}${API_PREFIX}/dashboard/metrics`,
      { headers }
    );
    
    check(dashboardRes, {
      'dashboard loaded': (r) => r.status === 200,
      'metrics received': (r) => r.json('totalVideos') !== undefined,
    });
    
    apiDuration.add(Date.now() - start);
    errorRate.add(dashboardRes.status !== 200);
  });
  
  sleep(randomIntBetween(1, 3));
  
  // Channel operations
  group('Channels', () => {
    const channelsRes = http.get(
      `${BASE_URL}${API_PREFIX}/channels`,
      { headers }
    );
    
    check(channelsRes, {
      'channels fetched': (r) => r.status === 200,
      'channels array': (r) => Array.isArray(r.json()),
    });
    
    errorRate.add(channelsRes.status !== 200);
  });
  
  sleep(randomIntBetween(1, 2));
  
  // Video operations
  group('Videos', () => {
    // Get video queue
    const queueRes = http.get(
      `${BASE_URL}${API_PREFIX}/videos/queue`,
      { headers }
    );
    
    check(queueRes, {
      'queue fetched': (r) => r.status === 200,
    });
    
    // Create new video (20% chance)
    if (Math.random() < 0.2) {
      const createVideoRes = http.post(
        `${BASE_URL}${API_PREFIX}/videos`,
        JSON.stringify({
          title: randomItem(videoTitles) + ' ' + randomString(5),
          description: 'Test video description',
          category: randomItem(categories),
          channelId: 'test-channel-id',
          scheduledDate: new Date(Date.now() + 86400000).toISOString(),
        }),
        { headers }
      );
      
      check(createVideoRes, {
        'video created': (r) => r.status === 201,
        'video id returned': (r) => r.json('id') !== undefined,
      });
      
      if (createVideoRes.status === 201) {
        videoCreations.add(1);
      } else {
        errorRate.add(1);
      }
    }
  });
  
  sleep(randomIntBetween(2, 5));
  
  // Cost tracking
  group('Cost Tracking', () => {
    const costRes = http.get(
      `${BASE_URL}${API_PREFIX}/cost/realtime`,
      { headers }
    );
    
    check(costRes, {
      'costs fetched': (r) => r.status === 200,
      'cost data valid': (r) => r.json('total_cost') !== undefined,
    });
    
    // Check if threshold exceeded
    if (costRes.status === 200) {
      const costData = costRes.json();
      check(costData, {
        'cost under threshold': (c) => c.total_cost < 1000,
      });
    }
  });
  
  sleep(randomIntBetween(1, 3));
  
  // Analytics
  group('Analytics', () => {
    const analyticsRes = http.get(
      `${BASE_URL}${API_PREFIX}/analytics/overview?days=7`,
      { headers }
    );
    
    check(analyticsRes, {
      'analytics loaded': (r) => r.status === 200,
    });
  });
  
  // WebSocket test (simulated)
  group('WebSocket', () => {
    const wsCheckRes = http.get(
      `${BASE_URL}${API_PREFIX}/ws/status`,
      { headers }
    );
    
    check(wsCheckRes, {
      'websocket endpoint available': (r) => r.status < 500,
    });
  });
  
  activeSessions.add(-1);
  sleep(randomIntBetween(3, 7));
}

// Lifecycle hooks
export function setup() {
  console.log('Setting up test environment...');
  
  // Check if API is available
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'API is healthy': (r) => r.status === 200,
  });
  
  if (healthCheck.status !== 200) {
    throw new Error('API is not healthy, aborting test');
  }
  
  // Create test users if needed
  testUsers.forEach(user => {
    http.post(
      `${BASE_URL}${API_PREFIX}/auth/register`,
      JSON.stringify({
        email: user.email,
        password: user.password,
        name: `Test User ${user.email}`,
      }),
      { headers: { 'Content-Type': 'application/json' } }
    );
  });
  
  return { startTime: Date.now() };
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration} seconds`);
  
  // Clean up test data if needed
  // This would typically involve calling cleanup endpoints
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'summary.json': JSON.stringify(data),
    'summary.html': htmlReport(data),
  };
}