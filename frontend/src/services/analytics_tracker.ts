/**
 * Analytics Tracking Service
 * Tracks user behavior and sends events to the backend
 */
import { apiClient } from './api';
import { v4 as uuidv4 } from 'uuid';

interface EventData {
  [key: string]: unknown;
}

interface TrackingEvent {
  event_type: string;
  event_data: EventData;
  session_id?: string;
  client_timestamp?: string;
  page_url?: string;
  referrer?: string;
}

class AnalyticsTracker {
  private sessionId: string;
  private userId: number | null = null;
  private eventQueue: TrackingEvent[] = [];
  private batchTimer: NodeJS.Timeout | null = null;
  private isEnabled: boolean = true;
  private debug: boolean = false;
  private batchSize: number = 10;
  private batchDelay: number = 5000; // 5 seconds
  private sessionTimeout: number = 30 * 60 * 1000; // 30 minutes
  private lastActivityTime: number = Date.now();
  private pageStartTime: number = Date.now();

  constructor() {
    this.sessionId = this.getOrCreateSessionId();
    this.initializeTracking();
  }

  /**
   * Initialize tracking listeners
   */
  private initializeTracking(): void {
    if (typeof window === 'undefined') return;

    // Track page views
    this.trackPageView();

    // Track page unload
    window.addEventListener('beforeunload', () => {
      this.flush(); // Send any pending events
      this.trackEvent('session_end', {
        duration: Date.now() - this.pageStartTime,
      });
    });

    // Track clicks
    document.addEventListener('click', (_e) => {
      const target = e.target as HTMLElement;
      if (target.dataset.track) {
        this.trackEvent('click', {
          element: target.dataset.track,
          text: target.textContent?.substring(0, 100),
          class: target.className,
          id: target.id,
        });
      }
    });

    // Track form submissions
    document.addEventListener('submit', (_e) => {
      const form = e.target as HTMLFormElement;
      if (form.dataset.track) {
        this.trackEvent('form_submit', {
          form_name: form.dataset.track,
          form_id: form.id,
          action: form.action,
        });
      }
    });

    // Track errors
    window.addEventListener('error', (_e) => {
      this.trackEvent('error', {
        _message: e.message,
        filename: e.filename,
        line: e.lineno,
        column: e.colno,
        stack: e.error?.stack?.substring(0, 500),
      });
    });

    // Track session activity
    ['mousedown', 'keydown', 'scroll', 'touchstart'].forEach((eventType) => {
      document.addEventListener(eventType, () => {
        this.updateActivity();
      }, { passive: true });
    });

    // Check session timeout
    setInterval(() => {
      if (Date.now() - this.lastActivityTime > this.sessionTimeout) {
        this.startNewSession();
      }
    }, 60000); // Check every minute
  }

  /**
   * Get or create session ID
   */
  private getOrCreateSessionId(): string {
    const stored = localStorage.getItem('analytics_session_id');
    const sessionExpiry = localStorage.getItem('analytics_session_expiry');

    if (stored && sessionExpiry && Date.now() < parseInt(sessionExpiry)) {
      return stored;
    }

    const newSessionId = uuidv4();
    localStorage.setItem('analytics_session_id', newSessionId);
    localStorage.setItem('analytics_session_expiry', (Date.now() + this.sessionTimeout).toString());
    
    return newSessionId;
  }

  /**
   * Start a new session
   */
  private startNewSession(): void {
    this.trackEvent('session_end', {
      duration: Date.now() - this.pageStartTime,
    });
    
    this.sessionId = uuidv4();
    localStorage.setItem('analytics_session_id', this.sessionId);
    localStorage.setItem('analytics_session_expiry', (Date.now() + this.sessionTimeout).toString());
    
    this.trackEvent('session_start', {
      previous_session_duration: Date.now() - this.pageStartTime,
    });
    
    this.pageStartTime = Date.now();
  }

  /**
   * Update activity timestamp
   */
  private updateActivity(): void {
    this.lastActivityTime = Date.now();
    localStorage.setItem('analytics_session_expiry', (Date.now() + this.sessionTimeout).toString());
  }

  /**
   * Set user ID for tracking
   */
  public setUserId(userId: number | null): void {
    this.userId = userId;
    if (userId) {
      this.trackEvent('identify', { user_id: userId });
    }
  }

  /**
   * Enable/disable tracking
   */
  public setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    if (!enabled) {
      this.flush(); // Send any pending events before disabling
    }
  }

  /**
   * Set debug mode
   */
  public setDebug(debug: boolean): void {
    this.debug = debug;
  }

  /**
   * Track a page view
   */
  public trackPageView(customData?: EventData): void {
    if (!this.isEnabled) return;

    const data = {
      url: window.location.href,
      path: window.location.pathname,
      search: window.location.search,
      hash: window.location.hash,
      title: document.title,
      referrer: document.referrer,
      screen_width: window.screen.width,
      screen_height: window.screen.height,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
      ...customData,
    };

    this.trackEvent('page_view', data);
  }

  /**
   * Track a custom event
   */
  public trackEvent(eventType: string, eventData: EventData = {}): void {
    if (!this.isEnabled) return;

    const _event: TrackingEvent = {
      event_type: eventType,
      event_data: {
        ...eventData,
        timestamp: new Date().toISOString(),
        session_duration: Date.now() - this.pageStartTime,
      },
      session_id: this.sessionId,
      client_timestamp: new Date().toISOString(),
      page_url: window.location.href,
      referrer: document.referrer,
    };

    if (this.debug) {
      console.log('[Analytics]', _event);
    }

    this.addToQueue(_event);
  }

  /**
   * Track feature usage
   */
  public trackFeature(featureName: string, metadata?: EventData): void {
    this.trackEvent('feature_use', {
      feature_name: featureName,
      ...metadata,
    });
  }

  /**
   * Track timing (performance)
   */
  public trackTiming(category: string, variable: string, timeMs: number, label?: string): void {
    this.trackEvent('timing', {
      category,
      variable,
      time_ms: timeMs,
      label,
    });
  }

  /**
   * Track user journey step
   */
  public trackJourneyStep(step: string, metadata?: EventData): void {
    this.trackEvent('journey_step', {
      step,
      ...metadata,
    });
  }

  /**
   * Track conversion
   */
  public trackConversion(conversionType: string, value?: number, metadata?: EventData): void {
    this.trackEvent('conversion', {
      conversion_type: conversionType,
      value,
      ...metadata,
    });
  }

  /**
   * Add event to queue
   */
  private addToQueue(_event: TrackingEvent): void {
    this.eventQueue.push(_event);

    // Send immediately if queue is full
    if (this.eventQueue.length >= this.batchSize) {
      this.flush();
    } else {
      // Schedule batch send
      this.scheduleBatch();
    }
  }

  /**
   * Schedule batch sending
   */
  private scheduleBatch(): void {
    if (this.batchTimer) return;

    this.batchTimer = setTimeout(() => {
      this.flush();
    }, this.batchDelay);
  }

  /**
   * Flush event queue
   */
  public async flush(): Promise<void> {
    if (this.eventQueue.length === 0) return;

    const events = [...this.eventQueue];
    this.eventQueue = [];

    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }

    try {
      if (events.length === 1) {
        // Send single event
        await apiClient.post('/api/v1/analytics/events', events[0]);
      } else {
        // Send batch
        await apiClient.post('/api/v1/analytics/events/batch', events);
      }
    } catch (_error) {
      // Re-add events to queue on failure
      this.eventQueue = [...events, ...this.eventQueue];
      
      if (this.debug) {
        console.error('[Analytics] Failed to send events:', _error);
      }
    }
  }

  /**
   * Track performance metrics
   */
  public trackPerformance(): void {
    if (!this.isEnabled || typeof window === 'undefined') return;

    // Use Performance API if available
    if (window.performance && window.performance.timing) {
      const timing = window.performance.timing;
      const loadTime = timing.loadEventEnd - timing.navigationStart;
      const domReadyTime = timing.domContentLoadedEventEnd - timing.navigationStart;
      const firstPaintTime = timing.responseStart - timing.navigationStart;

      this.trackTiming('performance', 'page_load', loadTime);
      this.trackTiming('performance', 'dom_ready', domReadyTime);
      this.trackTiming('performance', 'first_paint', firstPaintTime);

      // Track Core Web Vitals if available
      if ('PerformanceObserver' in window) {
        try {
          // Largest Contentful Paint
          const lcpObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            this.trackTiming('web_vitals', 'lcp', lastEntry.startTime);
          });
          lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

          // First Input Delay
          const fidObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry: unknown) => {
              this.trackTiming('web_vitals', 'fid', entry.processingStart - entry.startTime);
            });
          });
          fidObserver.observe({ entryTypes: ['first-input'] });

          // Cumulative Layout Shift
          const clsObserver = new PerformanceObserver((list) => {
            let clsScore = 0;
            list.getEntries().forEach((entry: unknown) => {
              if (!entry.hadRecentInput) {
                clsScore += entry.value;
              }
            });
            this.trackTiming('web_vitals', 'cls', clsScore * 1000); // Convert to ms
          });
          clsObserver.observe({ entryTypes: ['layout-shift'] });
        } catch (_e) {
          // Silently fail if observers are not supported
        }
      }
    }
  }

  /**
   * Get current session ID
   */
  public getSessionId(): string {
    return this.sessionId;
  }

  /**
   * Get tracking status
   */
  public isTrackingEnabled(): boolean {
    return this.isEnabled;
  }
}

// Create singleton instance
export const analyticsTracker = new AnalyticsTracker();

// Export convenience functions
export const trackEvent = (eventType: string, eventData?: EventData) => 
  analyticsTracker.trackEvent(eventType, eventData);

export const trackFeature = (featureName: string, metadata?: EventData) => 
  analyticsTracker.trackFeature(featureName, metadata);

export const trackPageView = (customData?: EventData) => 
  analyticsTracker.trackPageView(customData);

export const trackTiming = (category: string, variable: string, timeMs: number, label?: string) => 
  analyticsTracker.trackTiming(category, variable, timeMs, label);

export const trackConversion = (conversionType: string, value?: number, metadata?: EventData) => 
  analyticsTracker.trackConversion(conversionType, value, metadata);