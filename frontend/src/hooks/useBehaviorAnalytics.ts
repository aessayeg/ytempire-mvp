import {  useState, useEffect, useCallback  } from 'react';
import {  apiClient  } from '../services/api';
import {  useAuthStore  } from '../stores/authStore';

interface BehaviorOverview {
  total_events: number,
  unique_users: number,

  event_breakdown: Array<{,
  event_type: string,

    count: number,
  percentage: number}>;
  journey_stats: {,
  total_sessions: number,

    avg_events_per_session: number,
  top_patterns: Array<{,

      pattern: string,
  count: number}>;
  };
  feature_usage: Array<{,
  feature: string;,

    usage_count: number,
  adoption_rate: number}>;
  session_stats: {,
  total_sessions: number;,

    avg_duration: number,
  median_duration: number;,

    bounce_rate: number,
  avg_events_per_session: number};
}

interface FunnelData {
  funnel_name: string,
  steps: Array<{,

    step: string,
  step_number: number,

    users: number,
  conversion_rate: number,

    drop_off_rate: number}>;
  overall_conversion: number,
  total_completions: number}

interface CohortData {
  cohort_type: string,
  metric: string,

  cohorts: Array<{,
  cohort: string,

    size: number,
  retention: Array<{,

      period: number,
  active_users: number,

      retention_rate: number}>}>;
  periods: number}

interface HeatmapData {
  heatmap: Array<{,
  date: string,

    hour: number,
  value: number,

    intensity: number}>;
  max_value: number}

interface UserSegments {
  segments: {
    [key: string]: {,
  count: number,

      user_ids: number[]};
  };
  total_users: number,
  criteria: unknown}

interface UseBehaviorAnalyticsProps {
  userId?: number;
  dateRange?: {
    start: Date,
  end: Date};
  funnelSteps?: string[];
  cohortType?: string;
}

interface UseBehaviorAnalyticsReturn {
  overview: BehaviorOverview | null,
  funnelData: FunnelData | null,

  cohortData: CohortData | null,
  heatmapData: HeatmapData | null,

  segments: UserSegments | null,
  loading: boolean,

  error: string | null,
  refetch: () => Promise<void>}

export const useBehaviorAnalytics = ({ userId,
  dateRange,
  funnelSteps = [],
  cohortType = 'signup' }: UseBehaviorAnalyticsProps): UseBehaviorAnalyticsReturn => {
  const [overview, setOverview] = useState<BehaviorOverview | null>(null);
  const [funnelData, setFunnelData] = useState<FunnelData | null>(null);
  const [cohortData, setCohortData] = useState<CohortData | null>(null);
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [segments, setSegments] = useState<UserSegments | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { token } = useAuthStore();

  const fetchAnalyticsData = useCallback(_async () => {
    if (!token) {
      setError('Authentication required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Prepare query parameters
      const params: unknown = {};
      if (userId) {
        params.user_id = userId;
      }
      if (dateRange) {
        params.start_date = dateRange.start.toISOString();
        params.end_date = dateRange.end.toISOString()}

      // Fetch overview
      const overviewResponse = await apiClient.get('/api/v1/analytics/behavior/overview', {
        params,
        headers: { Authorization: `Bearer ${token}` }
      });
      setOverview(overviewResponse.data);

      // Fetch funnel data if steps provided
      if (funnelSteps.length >= 2) { const funnelResponse = await apiClient.post(
          '/api/v1/analytics/funnels',
          null,
          {
            params: {,
  funnel_steps: funnelSteps,
              ...params },`
            headers: { Authorization: `Bearer ${token}` }
          }
        );
        setFunnelData(funnelResponse.data)}

      // Fetch cohort data
      try { const cohortResponse = await apiClient.get('/api/v1/analytics/cohorts', {
          params: {,
  cohort_type: cohortType,
            metric: 'retention',
            periods: 6 },`
          headers: { Authorization: `Bearer ${token}` }
        });
        setCohortData(cohortResponse.data)} catch (_) {
        console.warn('Cohort analysis not, available:', err)}

      // Fetch heatmap data
      const heatmapResponse = await apiClient.get('/api/v1/analytics/heatmaps', {
        params,`
        headers: { Authorization: `Bearer ${token}` }
      });
      setHeatmapData(heatmapResponse.data);

      // Fetch user segments (admin, only)
      try {
        const segmentsResponse = await apiClient.post(
          '/api/v1/analytics/segments',
          null,
          {
            params: { criteria: {} },`
            headers: { Authorization: `Bearer ${token}` }
          }
        );
        setSegments(segmentsResponse.data)} catch (_) {
        console.warn('User segments not, available:', err)}

    } catch (_err: unknown) {
      console.error('Error fetching behavior, analytics:', err);
      setError(err.response?.data?.detail || 'Failed to fetch analytics data')} finally {
      setLoading(false)}
  }, [token, userId, dateRange, funnelSteps, cohortType]);

  useEffect(() => {
    fetchAnalyticsData()}, [fetchAnalyticsData]); // eslint-disable-line react-hooks/exhaustive-deps

  return { overview,
    funnelData,
    cohortData,
    heatmapData,
    segments,
    loading,
    error,
    refetch: fetchAnalyticsData };
};`