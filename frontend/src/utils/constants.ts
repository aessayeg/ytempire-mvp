/**
 * Constants - Application-wide constants
 * Owner: Frontend Team Lead
 */

// API Constants
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    FORGOT_PASSWORD: '/auth/forgot-password',
    RESET_PASSWORD: '/auth/reset-password',
    VERIFY_EMAIL: '/auth/verify-email',
    CHANGE_PASSWORD: '/auth/change-password'
  },
  USERS: {
    ME: '/users/me',
    USAGE: '/users/me/usage',
    PERMISSIONS: '/users/me/permissions'
  },
  CHANNELS: {
    LIST: '/channels',
    CREATE: '/channels',
    DETAIL: (id: string) => `/channels/${id}`,
    UPDATE: (id: string) => `/channels/${id}`,
    DELETE: (id: string) => `/channels/${id}`,
    STATS: (id: string) => `/channels/${id}/stats`,
    ANALYTICS: (id: string) => `/channels/${id}/analytics`,
    VIDEOS: (id: string) => `/channels/${id}/videos`,
    CONNECT_YOUTUBE: (id: string) => `/channels/${id}/youtube/connect`,
    DISCONNECT_YOUTUBE: (id: string) => `/channels/${id}/youtube/disconnect`,
    SYNC: (id: string) => `/channels/${id}/sync`,
    AVATAR: (id: string) => `/channels/${id}/avatar`,
    BANNER: (id: string) => `/channels/${id}/banner`
  },
  VIDEOS: {
    LIST: '/videos',
    GENERATE: '/videos/generate',
    DETAIL: (id: string) => `/videos/${id}`,
    UPDATE: (id: string) => `/videos/${id}`,
    DELETE: (id: string) => `/videos/${id}`,
    RETRY: (id: string) => `/videos/${id}/retry`,
    PUBLISH: (id: string) => `/videos/${id}/publish`,
    COST: (id: string) => `/videos/${id}/cost-breakdown`,
    ANALYTICS: (id: string) => `/videos/${id}/analytics`,
    DOWNLOAD: (id: string) => `/videos/${id}/download`,
    THUMBNAIL: (id: string) => `/videos/${id}/thumbnail`,
    QUEUE: '/videos/queue',
    QUEUE_ITEM: (id: string) => `/videos/queue/${id}`,
    BULK_DELETE: '/videos/bulk/delete',
    BULK_PUBLISH: '/videos/bulk/publish'
  },
  ANALYTICS: {
    DASHBOARD: '/analytics/dashboard',
    COSTS: '/analytics/costs',
    WEEKLY_REPORT: '/analytics/reports/weekly',
    COMPARE_CHANNELS: '/analytics/compare/channels',
    TRENDS: '/analytics/trends',
    INSIGHTS: '/analytics/insights',
    REALTIME: '/analytics/realtime'
  }
} as const

// HTTP Status Codes
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  METHOD_NOT_ALLOWED: 405,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
  GATEWAY_TIMEOUT: 504
} as const

// Application Routes
export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  REGISTER: '/register',
  FORGOT_PASSWORD: '/forgot-password',
  RESET_PASSWORD: '/reset-password',
  VERIFY_EMAIL: '/verify-email',
  DASHBOARD: '/dashboard',
  CHANNELS: '/channels',
  CHANNEL_DETAIL: (id: string) => `/channels/${id}`,
  CHANNEL_CREATE: '/channels/create',
  CHANNEL_EDIT: (id: string) => `/channels/${id}/edit`,
  VIDEOS: '/videos',
  VIDEO_DETAIL: (id: string) => `/videos/${id}`,
  VIDEO_GENERATE: '/videos/generate',
  ANALYTICS: '/analytics',
  SETTINGS: '/settings',
  PROFILE: '/profile',
  BILLING: '/billing',
  HELP: '/help',
  PRIVACY: '/privacy',
  TERMS: '/terms'
} as const

// Local Storage Keys
export const STORAGE_KEYS = {
  ACCESS_TOKEN: 'access_token',
  REFRESH_TOKEN: 'refresh_token',
  USER: 'user',
  THEME: 'theme',
  LANGUAGE: 'language',
  UI_PREFERENCES: 'ui_preferences',
  CHANNEL_FILTERS: 'channel_filters',
  VIDEO_FILTERS: 'video_filters',
  LAST_VISITED_CHANNEL: 'last_visited_channel',
  ONBOARDING_COMPLETED: 'onboarding_completed',
  NOTIFICATION_SETTINGS: 'notification_settings'
} as const

// Event Names
export const EVENTS = {
  // Auth events
  USER_LOGIN: 'user_login',
  USER_LOGOUT: 'user_logout',
  USER_REGISTER: 'user_register',
  TOKEN_REFRESH: 'token_refresh',
  
  // Channel events
  CHANNEL_CREATED: 'channel_created',
  CHANNEL_UPDATED: 'channel_updated',
  CHANNEL_DELETED: 'channel_deleted',
  CHANNEL_CONNECTED: 'channel_connected',
  CHANNEL_DISCONNECTED: 'channel_disconnected',
  
  // Video events
  VIDEO_GENERATION_STARTED: 'video_generation_started',
  VIDEO_GENERATION_COMPLETED: 'video_generation_completed',
  VIDEO_GENERATION_FAILED: 'video_generation_failed',
  VIDEO_PUBLISHED: 'video_published',
  VIDEO_DELETED: 'video_deleted',
  
  // Analytics events
  ANALYTICS_UPDATED: 'analytics_updated',
  REPORT_GENERATED: 'report_generated',
  
  // System events
  ERROR_OCCURRED: 'error_occurred',
  NOTIFICATION_SHOWN: 'notification_shown',
  PAGE_VIEW: 'page_view'
} as const

// WebSocket Event Types
export const WS_EVENTS = {
  // Connection
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  
  // Video processing
  VIDEO_STATUS_UPDATE: 'video_status_update',
  QUEUE_UPDATE: 'queue_update',
  GENERATION_PROGRESS: 'generation_progress',
  
  // Channel updates
  CHANNEL_ANALYTICS_UPDATE: 'channel_analytics_update',
  CHANNEL_STATUS_UPDATE: 'channel_status_update',
  
  // System notifications
  SYSTEM_NOTIFICATION: 'system_notification',
  SYSTEM_MAINTENANCE: 'system_maintenance',
  
  // Real-time metrics
  METRICS_UPDATE: 'metrics_update',
  COST_ALERT: 'cost_alert'
} as const

// File Types and Extensions
export const FILE_TYPES = {
  IMAGES: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
  VIDEOS: ['video/mp4', 'video/webm', 'video/ogg'],
  AUDIO: ['audio/mp3', 'audio/wav', 'audio/ogg'],
  DOCUMENTS: ['application/pdf', 'text/plain', 'application/msword']
} as const

export const FILE_EXTENSIONS = {
  IMAGES: ['.jpg', '.jpeg', '.png', '.gif', '.webp'],
  VIDEOS: ['.mp4', '.webm', '.ogg'],
  AUDIO: ['.mp3', '.wav', '.ogg'],
  DOCUMENTS: ['.pdf', '.txt', '.doc', '.docx']
} as const

// Validation Constants
export const VALIDATION = {
  PASSWORD_MIN_LENGTH: 8,
  USERNAME_MIN_LENGTH: 3,
  USERNAME_MAX_LENGTH: 30,
  CHANNEL_NAME_MIN_LENGTH: 3,
  CHANNEL_NAME_MAX_LENGTH: 50,
  VIDEO_TITLE_MIN_LENGTH: 10,
  VIDEO_TITLE_MAX_LENGTH: 100,
  VIDEO_DESCRIPTION_MIN_LENGTH: 50,
  VIDEO_DESCRIPTION_MAX_LENGTH: 5000,
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB
  MAX_IMAGE_SIZE: 10 * 1024 * 1024, // 10MB
  MAX_TAGS: 10,
  MIN_TAGS: 3
} as const

// Business Logic Constants
export const LIMITS = {
  MAX_CHANNELS_FREE: 2,
  MAX_CHANNELS_PRO: 10,
  MAX_CHANNELS_ENTERPRISE: 50,
  MAX_VIDEOS_PER_DAY_FREE: 3,
  MAX_VIDEOS_PER_DAY_PRO: 15,
  MAX_VIDEOS_PER_DAY_ENTERPRISE: 100,
  MAX_VIDEO_DURATION: 3600, // 1 hour in seconds
  MIN_VIDEO_DURATION: 30, // 30 seconds
  MAX_COST_PER_VIDEO: 3.00,
  COST_WARNING_THRESHOLD: 2.50
} as const

// UI Constants
export const UI = {
  DEBOUNCE_DELAY: 300,
  TOAST_DURATION: 5000,
  LOADING_TIMEOUT: 30000,
  PAGINATION_SIZES: [10, 25, 50, 100],
  DEFAULT_PAGE_SIZE: 25,
  ANIMATION_DURATION: 200,
  BREAKPOINTS: {
    XS: 0,
    SM: 600,
    MD: 900,
    LG: 1200,
    XL: 1536
  }
} as const

// Theme Constants
export const THEME = {
  MODES: ['light', 'dark', 'system'] as const,
  COLORS: {
    PRIMARY: '#1976d2',
    SECONDARY: '#dc004e',
    SUCCESS: '#2e7d32',
    WARNING: '#ed6c02',
    ERROR: '#d32f2f',
    INFO: '#0288d1'
  },
  SPACING: {
    XS: 4,
    SM: 8,
    MD: 16,
    LG: 24,
    XL: 32,
    XXL: 48
  }
} as const

// Video Generation Constants
export const VIDEO_GENERATION = {
  STATUSES: {
    PENDING: 'pending',
    PROCESSING: 'processing',
    COMPLETED: 'completed',
    FAILED: 'failed',
    CANCELLED: 'cancelled'
  },
  STAGES: {
    SCRIPT_GENERATION: 'script_generation',
    VOICE_SYNTHESIS: 'voice_synthesis',
    IMAGE_GENERATION: 'image_generation',
    VIDEO_COMPILATION: 'video_compilation',
    THUMBNAIL_CREATION: 'thumbnail_creation',
    UPLOAD_PREPARATION: 'upload_preparation'
  },
  STYLES: {
    EDUCATIONAL: 'educational',
    ENTERTAINMENT: 'entertainment',
    NEWS: 'news',
    TUTORIAL: 'tutorial',
    REVIEW: 'review',
    DOCUMENTARY: 'documentary'
  },
  VOICES: {
    MALE_PROFESSIONAL: 'male_professional',
    FEMALE_PROFESSIONAL: 'female_professional',
    MALE_CASUAL: 'male_casual',
    FEMALE_CASUAL: 'female_casual',
    NARRATOR: 'narrator'
  }
} as const

// Channel Constants
export const CHANNEL = {
  STATUSES: {
    ACTIVE: 'active',
    INACTIVE: 'inactive',
    CONNECTING: 'connecting',
    FAILED: 'failed',
    SUSPENDED: 'suspended'
  },
  CATEGORIES: [
    'Entertainment',
    'Education',
    'Technology',
    'Gaming',
    'Music',
    'Sports',
    'News',
    'Comedy',
    'Lifestyle',
    'Travel',
    'Food',
    'Health',
    'Business',
    'Science',
    'Art'
  ],
  LANGUAGES: [
    'en',
    'es',
    'fr',
    'de',
    'it',
    'pt',
    'ru',
    'ja',
    'ko',
    'zh',
    'hi',
    'ar'
  ],
  TONES: [
    'Professional',
    'Casual',
    'Friendly',
    'Authoritative',
    'Humorous',
    'Inspirational',
    'Educational',
    'Conversational'
  ]
} as const

// Analytics Constants
export const ANALYTICS = {
  TIME_RANGES: {
    LAST_7_DAYS: '7d',
    LAST_30_DAYS: '30d',
    LAST_90_DAYS: '90d',
    LAST_YEAR: '1y',
    CUSTOM: 'custom'
  },
  METRICS: {
    VIEWS: 'views',
    SUBSCRIBERS: 'subscribers',
    REVENUE: 'revenue',
    COSTS: 'costs',
    ROI: 'roi',
    ENGAGEMENT: 'engagement',
    WATCH_TIME: 'watch_time'
  },
  CHART_TYPES: {
    LINE: 'line',
    BAR: 'bar',
    AREA: 'area',
    PIE: 'pie',
    DONUT: 'donut'
  }
} as const

// Error Codes
export const ERROR_CODES = {
  // Authentication
  INVALID_CREDENTIALS: 'INVALID_CREDENTIALS',
  TOKEN_EXPIRED: 'TOKEN_EXPIRED',
  UNAUTHORIZED: 'UNAUTHORIZED',
  
  // Validation
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  REQUIRED_FIELD: 'REQUIRED_FIELD',
  INVALID_FORMAT: 'INVALID_FORMAT',
  
  // Business Logic
  CHANNEL_LIMIT_EXCEEDED: 'CHANNEL_LIMIT_EXCEEDED',
  VIDEO_LIMIT_EXCEEDED: 'VIDEO_LIMIT_EXCEEDED',
  COST_LIMIT_EXCEEDED: 'COST_LIMIT_EXCEEDED',
  INSUFFICIENT_CREDITS: 'INSUFFICIENT_CREDITS',
  
  // External Services
  YOUTUBE_API_ERROR: 'YOUTUBE_API_ERROR',
  OPENAI_API_ERROR: 'OPENAI_API_ERROR',
  ELEVENLABS_API_ERROR: 'ELEVENLABS_API_ERROR',
  
  // System
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT: 'TIMEOUT'
} as const

// Regular Expressions
export const REGEX = {
  EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PASSWORD: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/,
  PHONE: /^[\+]?[1-9][\d]{0,15}$/,
  URL: /^https?:\/\/.+/,
  YOUTUBE_URL: /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/,
  YOUTUBE_VIDEO_ID: /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/,
  CHANNEL_NAME: /^[a-zA-Z0-9\s\-]+$/,
  HEX_COLOR: /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/,
  SLUG: /^[a-z0-9-]+$/
} as const

// Feature Flags
export const FEATURES = {
  ANALYTICS_EXPORT: 'analytics_export',
  BULK_OPERATIONS: 'bulk_operations',
  ADVANCED_SCHEDULING: 'advanced_scheduling',
  CUSTOM_THUMBNAILS: 'custom_thumbnails',
  A_B_TESTING: 'a_b_testing',
  WHITE_LABEL: 'white_label',
  API_ACCESS: 'api_access',
  WEBHOOK_SUPPORT: 'webhook_support'
} as const

// Subscription Plans
export const PLANS = {
  FREE: {
    id: 'free',
    name: 'Free',
    price: 0,
    features: ['2 channels', '3 videos/day', 'Basic analytics']
  },
  PRO: {
    id: 'pro',
    name: 'Pro',
    price: 29,
    features: ['10 channels', '15 videos/day', 'Advanced analytics', 'Custom thumbnails']
  },
  ENTERPRISE: {
    id: 'enterprise',
    name: 'Enterprise',
    price: 99,
    features: ['50 channels', '100 videos/day', 'Full analytics', 'API access', 'White label']
  }
} as const

export default {
  API_ENDPOINTS,
  HTTP_STATUS,
  ROUTES,
  STORAGE_KEYS,
  EVENTS,
  WS_EVENTS,
  FILE_TYPES,
  FILE_EXTENSIONS,
  VALIDATION,
  LIMITS,
  UI,
  THEME,
  VIDEO_GENERATION,
  CHANNEL,
  ANALYTICS,
  ERROR_CODES,
  REGEX,
  FEATURES,
  PLANS
}