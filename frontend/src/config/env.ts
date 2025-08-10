/**
 * Environment Configuration
 * Owner: Frontend Team Lead
 */

const getEnvVar = (key: string, defaultValue = ''): string => {
  return import.meta.env[key] || defaultValue
}

export const env = {
  // API Configuration
  API_URL: getEnvVar('VITE_API_URL', 'http://localhost:8000'),
  API_VERSION: getEnvVar('VITE_API_VERSION', 'v1'),
  API_TIMEOUT: parseInt(getEnvVar('VITE_API_TIMEOUT', '30000')),

  // WebSocket Configuration
  WS_URL: getEnvVar('VITE_WS_URL', 'ws://localhost:8000'),
  WS_RECONNECT_INTERVAL: parseInt(getEnvVar('VITE_WS_RECONNECT_INTERVAL', '5000')),

  // App Configuration
  APP_NAME: getEnvVar('VITE_APP_NAME', 'YTEmpire'),
  APP_VERSION: getEnvVar('VITE_APP_VERSION', '0.1.0'),
  
  // Feature Flags
  ENABLE_ANALYTICS: getEnvVar('VITE_ENABLE_ANALYTICS', 'false') === 'true',
  ENABLE_DEBUG: getEnvVar('VITE_ENABLE_DEBUG', 'true') === 'true',
  ENABLE_MOCK_DATA: getEnvVar('VITE_ENABLE_MOCK_DATA', 'false') === 'true',

  // External Services
  STRIPE_PUBLIC_KEY: getEnvVar('VITE_STRIPE_PUBLIC_KEY', ''),
  GOOGLE_CLIENT_ID: getEnvVar('VITE_GOOGLE_CLIENT_ID', ''),
  
  // Limits
  MAX_CHANNELS_PER_USER: parseInt(getEnvVar('VITE_MAX_CHANNELS', '5')),
  MAX_VIDEOS_PER_DAY: parseInt(getEnvVar('VITE_MAX_VIDEOS_PER_DAY', '10')),
  MAX_VIDEO_DURATION: parseInt(getEnvVar('VITE_MAX_VIDEO_DURATION', '3600')),
  
  // Cost Limits
  MAX_COST_PER_VIDEO: parseFloat(getEnvVar('VITE_MAX_COST_PER_VIDEO', '3.0')),
  COST_WARNING_THRESHOLD: parseFloat(getEnvVar('VITE_COST_WARNING_THRESHOLD', '2.5')),

  // Environment
  NODE_ENV: import.meta.env.MODE,
  IS_PRODUCTION: import.meta.env.PROD,
  IS_DEVELOPMENT: import.meta.env.DEV,
}

// Validate required environment variables
const requiredVars = ['VITE_API_URL']
const missingVars = requiredVars.filter(key => !import.meta.env[key])

if (missingVars.length > 0 && env.IS_PRODUCTION) {
  throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`)
}

export default env