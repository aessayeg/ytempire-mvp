# YTEMPIRE Documentation - Business Logic

## 6.1 Cost Calculations

### Cost Per Video Breakdown

```typescript
// Cost calculation formula and components
interface VideoCostBreakdown {
  // AI Content Generation (GPT-4)
  scriptGeneration: {
    tokensUsed: number;        // ~2000-4000 tokens per script
    costPerToken: 0.00003;     // $0.03 per 1K tokens
    totalCost: number;         // tokensUsed * costPerToken
  };
  
  // Voice Synthesis (ElevenLabs)
  voiceSynthesis: {
    characterCount: number;    // ~3000-5000 chars per video
    costPerChar: 0.00003;      // $0.30 per 10K characters
    totalCost: number;         // characterCount * costPerChar
  };
  
  // Video Rendering (Local GPU)
  videoRendering: {
    renderMinutes: number;     // 3-5 minutes per video
    gpuCostPerMinute: 0.02;    // Electricity + amortized hardware
    totalCost: number;         // renderMinutes * gpuCostPerMinute
  };
  
  // Storage & CDN
  storage: {
    videoSizeMB: number;       // 50-200MB per video
    storageCostPerGB: 0.023;   // S3 pricing
    cdnTransferGB: number;     // Estimated based on views
    cdnCostPerGB: 0.085;       // CloudFront pricing
    totalCost: number;         // (videoSizeMB/1024 * storageCostPerGB) + (cdnTransferGB * cdnCostPerGB)
  };
  
  // YouTube API
  youtubeApi: {
    quotaUnitsUsed: 1600;      // Upload = 1600 units
    costPerMillion: 0;         // Free within quota
    totalCost: 0;              // Only cost if exceeding 10K uploads/day
  };
  
  // Total Cost Calculation
  totalCost: number;           // Sum of all components (target: <$0.50)
}
```

### Cost Calculation Implementation

```typescript
export const calculateVideoCost = (params: VideoGenerationParams): VideoCostBreakdown => {
  const breakdown: VideoCostBreakdown = {
    scriptGeneration: {
      tokensUsed: estimateTokens(params.length),
      costPerToken: 0.00003,
      totalCost: 0
    },
    voiceSynthesis: {
      characterCount: estimateCharacters(params.length),
      costPerChar: 0.00003,
      totalCost: 0
    },
    videoRendering: {
      renderMinutes: estimateRenderTime(params.length),
      gpuCostPerMinute: 0.02,
      totalCost: 0
    },
    storage: {
      videoSizeMB: estimateVideoSize(params.length, params.quality),
      storageCostPerGB: 0.023,
      cdnTransferGB: 0,
      cdnCostPerGB: 0.085,
      totalCost: 0
    },
    youtubeApi: {
      quotaUnitsUsed: 1600,
      costPerMillion: 0,
      totalCost: 0
    },
    totalCost: 0
  };
  
  // Calculate individual costs
  breakdown.scriptGeneration.totalCost = 
    breakdown.scriptGeneration.tokensUsed * breakdown.scriptGeneration.costPerToken;
    
  breakdown.voiceSynthesis.totalCost = 
    breakdown.voiceSynthesis.characterCount * breakdown.voiceSynthesis.costPerChar;
    
  breakdown.videoRendering.totalCost = 
    breakdown.videoRendering.renderMinutes * breakdown.videoRendering.gpuCostPerMinute;
    
  breakdown.storage.totalCost = 
    (breakdown.storage.videoSizeMB / 1024 * breakdown.storage.storageCostPerGB);
  
  // Sum total cost
  breakdown.totalCost = 
    breakdown.scriptGeneration.totalCost +
    breakdown.voiceSynthesis.totalCost +
    breakdown.videoRendering.totalCost +
    breakdown.storage.totalCost +
    breakdown.youtubeApi.totalCost;
  
  return breakdown;
};

// Helper functions for estimation
const estimateTokens = (videoLength: 'short' | 'medium' | 'long'): number => {
  const tokenMap = {
    short: 2000,   // 3-5 minute video
    medium: 3000,  // 5-8 minute video
    long: 4000     // 8-12 minute video
  };
  return tokenMap[videoLength];
};

const estimateCharacters = (videoLength: 'short' | 'medium' | 'long'): number => {
  const charMap = {
    short: 3000,   // ~500 words
    medium: 4500,  // ~750 words
    long: 6000     // ~1000 words
  };
  return charMap[videoLength];
};
```

### Cost Alert Thresholds

```typescript
// Cost alert configuration and logic
interface CostAlertConfig {
  daily: {
    warning: 40,    // $40 - Yellow alert
    critical: 45,   // $45 - Orange alert
    limit: 50       // $50 - Red alert, pause generation
  };
  perVideo: {
    warning: 0.40,  // $0.40 - Optimization suggested
    critical: 0.45, // $0.45 - Review required
    limit: 0.50     // $0.50 - Block video
  };
  monthly: {
    warning: 1000,  // $1000 - Budget review
    critical: 1250, // $1250 - Throttle generation
    limit: 1500     // $1500 - Hard stop
  };
}

export const checkCostAlerts = (
  currentCosts: CurrentCosts
): CostAlert[] => {
  const alerts: CostAlert[] = [];
  const config = getCostAlertConfig();
  
  // Daily cost check
  if (currentCosts.daily >= config.daily.limit) {
    alerts.push({
      id: generateAlertId(),
      type: 'cost',
      severity: 'critical',
      level: 'limit',
      title: 'Daily Cost Limit Reached',
      message: `Daily cost of $${currentCosts.daily} exceeds limit of $${config.daily.limit}`,
      suggestedAction: 'Video generation paused. Review and optimize costs.',
      actionRequired: true
    });
  } else if (currentCosts.daily >= config.daily.critical) {
    alerts.push({
      id: generateAlertId(),
      type: 'cost',
      severity: 'high',
      level: 'critical',
      title: 'Daily Cost Critical',
      message: `Daily cost of $${currentCosts.daily} approaching limit`,
      suggestedAction: 'Consider pausing low-performing channels',
      actionRequired: false
    });
  }
  
  return alerts;
};
```

## 6.2 Revenue Tracking

### Revenue Sources & Attribution

```typescript
// Revenue tracking system
interface RevenueSource {
  youtube: {
    adRevenue: number;         // YouTube Partner Program
    premiumRevenue: number;    // YouTube Premium views
    superThanks: number;       // Direct viewer support
  };
  affiliate: {
    amazonAssociates: number;  // Amazon affiliate links
    otherAffiliates: number;   // Other affiliate programs
    sponsorships: number;      // Direct sponsorships
  };
  calculated: {
    cpm: number;              // Cost per 1000 views
    rpm: number;              // Revenue per 1000 views
    affiliateCtr: number;     // Affiliate link click rate
    conversionRate: number;   // Affiliate conversion rate
  };
}

// Revenue calculation per video
export const calculateVideoRevenue = (
  video: Video,
  analytics: VideoAnalytics
): VideoRevenue => {
  const revenue: VideoRevenue = {
    // YouTube ad revenue (simplified calculation)
    adRevenue: calculateAdRevenue(analytics.views, analytics.watchTime),
    
    // Affiliate revenue
    affiliateRevenue: calculateAffiliateRevenue(
      analytics.clicks,
      analytics.conversions
    ),
    
    // Total revenue
    totalRevenue: 0,
    
    // Metrics
    rpm: 0,
    roi: 0,
    profitMargin: 0
  };
  
  revenue.totalRevenue = revenue.adRevenue + revenue.affiliateRevenue;
  revenue.rpm = (revenue.totalRevenue / analytics.views) * 1000;
  revenue.roi = ((revenue.totalRevenue - video.cost) / video.cost) * 100;
  revenue.profitMargin = ((revenue.totalRevenue - video.cost) / revenue.totalRevenue) * 100;
  
  return revenue;
};

// YouTube ad revenue calculation
const calculateAdRevenue = (views: number, watchTimeMinutes: number): number => {
  // Typical CPM ranges by niche
  const cpmByNiche = {
    tech: 8.0,        // $8 per 1000 views
    finance: 12.0,    // $12 per 1000 views
    gaming: 4.0,      // $4 per 1000 views
    education: 6.0,   // $6 per 1000 views
    lifestyle: 5.0,   // $5 per 1000 views
    default: 5.0
  };
  
  // YouTube takes 45% cut
  const creatorShare = 0.55;
  
  // Adjust CPM based on watch time (better retention = higher CPM)
  const avgWatchTime = watchTimeMinutes / views;
  const watchTimeMultiplier = Math.min(avgWatchTime / 5, 1.5); // Cap at 1.5x
  
  const effectiveCpm = cpmByNiche.default * watchTimeMultiplier * creatorShare;
  
  return (views / 1000) * effectiveCpm;
};
```

### Revenue Forecasting

```typescript
export const forecastRevenue = (
  historicalData: HistoricalRevenue[],
  channels: Channel[],
  period: number = 30 // days
): RevenueForecast => {
  // Calculate growth trends
  const growthRate = calculateGrowthRate(historicalData);
  
  // Factor in seasonality
  const seasonalityFactor = getSeasonalityFactor(new Date());
  
  // Channel-specific projections
  const channelProjections = channels.map(channel => {
    const channelHistory = historicalData.filter(d => d.channelId === channel.id);
    const avgDailyRevenue = calculateAvgDailyRevenue(channelHistory);
    const channelGrowth = calculateChannelGrowth(channelHistory);
    
    return {
      channelId: channel.id,
      projectedRevenue: avgDailyRevenue * period * (1 + channelGrowth) * seasonalityFactor,
  ## 6.3 Automation Metrics

### Automation Percentage Calculation

```typescript
// Automation metrics calculation
interface AutomationMetrics {
  overallPercentage: number;       // 0-100 (target: 95%)
  byCategory: {
    contentCreation: number;       // Script, thumbnail, video generation
    publishing: number;            // Upload, scheduling, metadata
    optimization: number;          // Title/description updates, A/B testing
    monitoring: number;            // Analytics tracking, alerts
    monetization: number;          // Ad placement, affiliate links
  };
  manualInterventions: {
    count: number;
    reasons: string[];
    timeSpent: number;             // minutes
  };
}

export const calculateAutomationPercentage = (
  channel: Channel,
  period: { start: Date; end: Date }
): AutomationMetrics => {
  const metrics: AutomationMetrics = {
    overallPercentage: 0,
    byCategory: {
      contentCreation: 100,        // Fully automated
      publishing: 95,              // Manual review occasionally
      optimization: 90,            // Some manual A/B testing
      monitoring: 100,             // Fully automated alerts
      monetization: 85             // Manual sponsor negotiations
    },
    manualInterventions: {
      count: 0,
      reasons: [],
      timeSpent: 0
    }
  };
  
  // Calculate based on actual interventions
  const interventions = getManualInterventions(channel.id, period);
  
  interventions.forEach(intervention => {
    metrics.manualInterventions.count++;
    metrics.manualInterventions.reasons.push(intervention.reason);
    metrics.manualInterventions.timeSpent += intervention.duration;
    
    // Reduce automation percentage based on intervention type
    switch (intervention.type) {
      case 'content_review':
        metrics.byCategory.contentCreation -= 5;
        break;
      case 'manual_upload':
        metrics.byCategory.publishing -= 10;
        break;
      case 'optimization_tweak':
        metrics.byCategory.optimization -= 3;
        break;
    }
  });
  
  // Calculate weighted overall percentage
  const weights = {
    contentCreation: 0.30,
    publishing: 0.25,
    optimization: 0.20,
    monitoring: 0.15,
    monetization: 0.10
  };
  
  metrics.overallPercentage = Object.entries(metrics.byCategory)
    .reduce((sum, [category, percentage]) => {
      return sum + (percentage * weights[category]);
    }, 0);
  
  return metrics;
};

// Time saved calculation
export const calculateTimeSaved = (
  automationMetrics: AutomationMetrics,
  videoCount: number
): TimeSavings => {
  // Manual video creation time estimates (minutes)
  const manualTime = {
    research: 60,
    scriptWriting: 90,
    recording: 30,
    editing: 120,
    thumbnailCreation: 30,
    uploading: 15,
    optimization: 20,
    total: 365  // ~6 hours per video
  };
  
  const automatedTime = {
    setup: 5,
    review: 10,
    tweaks: 5,
    total: 20  // 20 minutes per video with automation
  };
  
  const timeSavedPerVideo = manualTime.total - automatedTime.total;
  const totalTimeSaved = timeSavedPerVideo * videoCount;
  
  return {
    perVideo: {
      manual: manualTime.total,
      automated: automatedTime.total,
      saved: timeSavedPerVideo,
      percentage: (timeSavedPerVideo / manualTime.total) * 100
    },
    total: {
      minutes: totalTimeSaved,
      hours: Math.round(totalTimeSaved / 60),
      days: Math.round(totalTimeSaved / 60 / 8), // 8-hour work days
      value: totalTimeSaved * 0.50  // $30/hour / 60 minutes
    }
  };
};
```

## 6.4 Video Generation Workflow

### Video Generation State Machine

```typescript
// Video generation state machine
interface VideoGenerationStates {
  queued: {
    description: 'Video is waiting to be processed';
    duration: '0-30 minutes';
    actions: ['cancel', 'prioritize'];
    nextStates: ['processing', 'cancelled'];
  };
  
  processing: {
    stages: {
      topic_research: {
        description: 'AI researching trending topics';
        duration: '10-30 seconds';
        percentage: 5;
      };
      script_generation: {
        description: 'GPT-4 writing video script';
        duration: '20-40 seconds';
        percentage: 20;
      };
      voice_synthesis: {
        description: 'ElevenLabs generating narration';
        duration: '30-60 seconds';
        percentage: 35;
      };
      video_assembly: {
        description: 'Combining assets into video';
        duration: '2-4 minutes';
        percentage: 60;
      };
      thumbnail_creation: {
        description: 'AI generating thumbnail';
        duration: '10-20 seconds';
        percentage: 70;
      };
      quality_check: {
        description: 'Automated quality validation';
        duration: '5-10 seconds';
        percentage: 80;
      };
      youtube_upload: {
        description: 'Uploading to YouTube';
        duration: '30-90 seconds';
        percentage: 95;
      };
      final_optimization: {
        description: 'Setting tags, cards, end screens';
        duration: '10-20 seconds';
        percentage: 100;
      };
    };
    nextStates: ['completed', 'failed'];
  };
  
  completed: {
    description: 'Video successfully published';
    actions: ['view', 'analytics', 'share'];
    metrics: ['views', 'revenue', 'engagement'];
  };
  
  failed: {
    description: 'Video generation failed';
    reasons: [
      'script_generation_failed',
      'voice_synthesis_error',
      'rendering_timeout',
      'upload_quota_exceeded',
      'quality_check_failed',
      'cost_limit_exceeded'
    ];
    actions: ['retry', 'debug', 'cancel'];
    nextStates: ['queued', 'cancelled'];
  };
  
  cancelled: {
    description: 'Video generation cancelled by user or system';
    reasons: ['user_cancelled', 'cost_limit', 'channel_paused'];
    actions: ['delete'];
  };
}

// Progress tracking implementation
export const trackVideoProgress = (
  videoId: string,
  stage: string,
  progress: number
): VideoProgress => {
  const timestamp = new Date();
  const elapsedTime = calculateElapsedTime(videoId);
  const estimatedRemaining = estimateRemainingTime(stage, progress);
  
  return {
    videoId,
    stage,
    progress,
    timestamp,
    elapsedTime,
    estimatedRemaining,
    isStalled: checkIfStalled(videoId, timestamp),
    currentCost: calculateCurrentCost(videoId, stage)
  };
};

// Failure recovery logic
export const handleVideoFailure = async (
  video: FailedVideo,
  error: VideoError
): Promise<RecoveryAction> => {
  const recovery: RecoveryAction = {
    videoId: video.id,
    action: 'none',
    reason: ''
  };
  
  // Determine recovery strategy based on error type
  switch (error.code) {
    case 'SCRIPT_GENERATION_TIMEOUT':
      if (video.retryCount < 3) {
        recovery.action = 'retry';
        recovery.delay = 60000; // 1 minute
        recovery.reason = 'Temporary AI service issue';
      }
      break;
      
    case 'VOICE_SYNTHESIS_QUOTA':
      recovery.action = 'queue_for_tomorrow';
      recovery.reason = 'Daily voice quota exceeded';
      break;
      
    case 'COST_LIMIT_EXCEEDED':
      recovery.action = 'pause_channel';
      recovery.reason = 'Cost limit reached';
      recovery.alert = {
        type: 'cost',
        severity: 'critical',
        message: 'Channel paused due to cost limit'
      };
      break;
      
    case 'YOUTUBE_UPLOAD_FAILED':
      if (error.details?.quotaExceeded) {
        recovery.action = 'queue_for_tomorrow';
        recovery.reason = 'YouTube quota exceeded';
      } else {
        recovery.action = 'retry';
        recovery.delay = 300000; // 5 minutes
        recovery.reason = 'YouTube API temporary issue';
      }
      break;
      
    case 'QUALITY_CHECK_FAILED':
      recovery.action = 'manual_review';
      recovery.reason = `Quality issue: ${error.details?.qualityIssue}`;
      recovery.notification = {
        type: 'quality',
        message: 'Video requires manual review'
      };
      break;
      
    default:
      recovery.action = 'manual_review';
      recovery.reason = 'Unknown error requires investigation';
  }
  
  return recovery;
};
```

### Quality Control Metrics

```typescript
// Video quality scoring system
interface VideoQualityScore {
  overall: number;           // 0-100 (minimum 75 for auto-publish)
  components: {
    scriptQuality: {
      score: number;         // 0-100
      factors: {
        grammar: number;
        coherence: number;
        engagement: number;
        seoOptimization: number;
      };
    };
    audioQuality: {
      score: number;         // 0-100
      factors: {
        clarity: number;
        pacing: number;
        pronunciation: number;
        naturalness: number;
      };
    };
    videoQuality: {
      score: number;         // 0-100
      factors: {
        resolution: number;
        transitions: number;
        relevance: number;
        branding: number;
      };
    };
    thumbnailQuality: {
      score: number;         // 0-100
      factors: {
        clickability: number;
        clarity: number;
        branding: number;
        textReadability: number;
      };
    };
  };
  flags: QualityFlag[];
}

export const calculateQualityScore = (
  video: ProcessedVideo
): VideoQualityScore => {
  const score: VideoQualityScore = {
    overall: 0,
    components: {
      scriptQuality: evaluateScript(video.script),
      audioQuality: evaluateAudio(video.audio),
      videoQuality: evaluateVideo(video.video),
      thumbnailQuality: evaluateThumbnail(video.thumbnail)
    },
    flags: []
  };
  
  // Calculate weighted overall score
  const weights = {
    scriptQuality: 0.30,
    audioQuality: 0.25,
    videoQuality: 0.25,
    thumbnailQuality: 0.20
  };
  
  score.overall = Object.entries(score.components)
    .reduce((sum, [component, data]) => {
      return sum + (data.score * weights[component]);
    }, 0);
  
  // Add quality flags
  if (score.components.scriptQuality.factors.grammar < 80) {
    score.flags.push({
      type: 'warning',
      component: 'script',
      issue: 'Grammar issues detected',
      suggestion: 'Review script for grammar corrections'
    });
  }
  
  // Minimum quality threshold
  const MIN_QUALITY_SCORE = 75;
  if (score.overall < MIN_QUALITY_SCORE) {
    score.flags.push({
      type: 'error',
      component: 'overall',
      issue: `Quality score ${score.overall} below minimum ${MIN_QUALITY_SCORE}`,
      suggestion: 'Video requires regeneration or manual review'
    });
  }
  
  return score;
};
```

## 6.5 Channel Management Logic

### Channel Status Transitions

```typescript
// Channel status state machine
interface ChannelStatusMachine {
  states: {
    active: {
      description: 'Channel is generating videos automatically';
      transitions: {
        pause: 'paused';
        error: 'error';
      };
      conditions: {
        hasValidAuth: true;
        withinQuota: true;
        noCriticalErrors: true;
      };
    };
    paused: {
      description: 'Channel is manually paused by user';
      transitions: {
        resume: 'active';
        error: 'error';
      };
      conditions: {
        userInitiated: true;
      };
    };
    error: {
      description: 'Channel has encountered an error';
      transitions: {
        retry: 'active';
        pause: 'paused';
      };
      conditions: {
        authExpired: boolean;
        quotaExceeded: boolean;
        apiError: boolean;
        costLimitReached: boolean;
      };
    };
  };
}

// Channel status manager
export const channelStatusManager = {
  canTransition: (
    currentStatus: ChannelStatus,
    targetStatus: ChannelStatus,
    context: ChannelContext
  ): boolean => {
    // Define valid transitions
    const validTransitions = {
      active: ['paused', 'error'],
      paused: ['active', 'error'],
      error: ['active', 'paused']
    };
    
    if (!validTransitions[currentStatus].includes(targetStatus)) {
      return false;
    }
    
    // Check specific conditions
    if (targetStatus === 'active') {
      return (
        context.hasValidAuth &&
        context.withinQuota &&
        context.costWithinLimit &&
        !context.hasCriticalErrors
      );
    }
    
    return true;
  },
  
  getStatusColor: (status: ChannelStatus): string => {
    const colors = {
      active: '#4CAF50',   // Green
      paused: '#FF9800',   // Orange
      error: '#F44336'     // Red
    };
    return colors[status];
  },
  
  getRecommendedAction: (
    status: ChannelStatus,
    context: ChannelContext
  ): ChannelAction | null => {
    if (status === 'error') {
      if (context.authExpired) {
        return {
          type: 'reauth',
          label: 'Reconnect YouTube',
          priority: 'high'
        };
      }
      if (context.costLimitReached) {
        return {
          type: 'review_costs',
          label: 'Review Costs',
          priority: 'critical'
        };
      }
    }
    
    if (status === 'paused' && context.daysInactive > 7) {
      return {
        type: 'resume',
        label: 'Resume Channel',
        priority: 'medium'
      };
    }
    
    return null;
  }
};
```

### Channel Performance Scoring

```typescript
// Channel performance evaluation
interface ChannelPerformanceScore {
  overall: number;              // 0-100
  metrics: {
    revenue: {
      score: number;
      actual: number;
      target: number;
      trend: 'up' | 'down' | 'stable';
    };
    growth: {
      score: number;
      subscriberGrowth: number;
      viewGrowth: number;
      trend: 'up' | 'down' | 'stable';
    };
    engagement: {
      score: number;
      avgCtr: number;
      avgRetention: number;
      avgLikeRatio: number;
    };
    efficiency: {
      score: number;
      costPerView: number;
      revenuePerVideo: number;
      roi: number;
    };
  };
  recommendations: ChannelRecommendation[];
}

export const evaluateChannelPerformance = (
  channel: Channel,
  analytics: ChannelAnalytics,
  period: number = 30 // days
): ChannelPerformanceScore => {
  const performance: ChannelPerformanceScore = {
    overall: 0,
    metrics: {
      revenue: evaluateRevenue(channel, analytics),
      growth: evaluateGrowth(channel, analytics),
      engagement: evaluateEngagement(analytics),
      efficiency: evaluateEfficiency(channel, analytics)
    },
    recommendations: []
  };
  
  // Calculate weighted overall score
  const weights = {
    revenue: 0.35,
    growth: 0.25,
    engagement: 0.25,
    efficiency: 0.15
  };
  
  performance.overall = Object.entries(performance.metrics)
    .reduce((sum, [metric, data]) => {
      return sum + (data.score * weights[metric]);
    }, 0);
  
  // Generate recommendations based on scores
  performance.recommendations = generateRecommendations(performance);
  
  return performance;
};

// Business rule validation
export const businessRules = {
  // Cost limits
  costs: {
    maxPerVideo: 0.50,
    maxDaily: 50.00,
    maxMonthly: 1500.00,
    alertThresholds: {
      warning: 0.80,   // 80% of limit
      critical: 0.90,  // 90% of limit
      stop: 1.00       // 100% of limit
    }
  },
  
  // Channel limits
  channels: {
    maxPerUser: 5,
    maxDailyVideosPerChannel: 3,
    maxTotalDailyVideos: 15,
    minQualityScore: 75
  },
  
  // Performance thresholds
  performance: {
    minROI: 100,              // 100% minimum ROI
    minCTR: 2.0,              // 2% minimum CTR
    minRetention: 30,         // 30% minimum retention
    evaluationPeriod: 7       // Days before evaluation
  },
  
  // Automation rules
  automation: {
    minAutomationRate: 85,    // 85% automation required
    maxManualTime: 60,        // 60 minutes per week max
    autoPublishQuality: 75,   // Min quality for auto-publish
    retryAttempts: 3          // Max retry attempts
  }
};
``` calculateConfidence(channelHistory),
      assumptions: {
        avgDailyVideos: channel.settings.dailyVideoLimit,
        avgRevenuePerVideo: avgDailyRevenue / channel.settings.dailyVideoLimit,
        growthRate: channelGrowth,
        seasonalityImpact: seasonalityFactor
      }
    };
  });
  
  return {
    period,
    totalProjected: channelProjections.reduce((sum, p) => sum + p.projectedRevenue, 0),
    byChannel: channelProjections,
    confidence: calculateOverallConfidence(channelProjections),
    assumptions: {
      basedOnDays: historicalData.length,
      growthRate,
      seasonalityFactor
    }
  };
};

// Seasonality factors by month
const getSeasonalityFactor = (date: Date): number => {
  const month = date.getMonth();
  const seasonalityMap = {
    0: 0.85,   // January - Post-holiday slump
    1: 0.90,   // February
    2: 0.95,   // March
    3: 1.00,   // April
    4: 1.00,   // May
    5: 0.95,   // June
    6: 0.90,   // July - Summer slump
    7: 0.90,   // August
    8: 1.05,   // September - Back to school
    9: 1.10,   // October - Pre-holiday
    10: 1.20,  // November - Black Friday
    11: 1.25   // December - Holiday peak
  };
  
  return seasonalityMap[month] || 1.0;
};