import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../services/api';
import { useAuthStore } from '../stores/authStore';

interface RevenueOverview {
  total_revenue: number;
  average_revenue_per_video: number;
  highest_revenue_video: number;
  lowest_revenue_video: number;
  total_videos_monetized: number;
  daily_revenue: Array<{ date: string; revenue: number }>;
  channel_breakdown: Array<{ channel_id: number; channel_name: string; revenue: number }>;
  cpm: number;
  rpm: number;
  forecast: {
    next_7_days: number;
    next_30_days: number;
    confidence: string;
    trend_factor: number;
  };
  revenue_growth?: number;
  cpm_trend?: number;
  rpm_trend?: number;
}

interface RevenueTrend {
  period: string;
  revenue: number;
  views: number;
  video_count: number;
  rpm: number;
  growth_rate?: number;
}

interface RevenueForecast {
  date: string;
  predicted_revenue: number;
  confidence_lower: number;
  confidence_upper: number;
}

interface RevenueBreakdown {
  source?: string;
  content_type?: string;
  length_range?: string;
  time_period?: string;
  revenue: number;
  percentage?: number;
}

interface ChannelRevenue {
  channel_id: number;
  channel_name: string;
  total_revenue: number;
  video_count: number;
  average_revenue_per_video: number;
  total_views: number;
}

interface UseRevenueDataProps {
  userId?: number;
  channelId?: number;
  dateRange?: {
    start: Date;
    end: Date;
  };
  period?: 'daily' | 'weekly' | 'monthly';
  breakdownBy?: string;
}

interface UseRevenueDataReturn {
  overview: RevenueOverview | null;
  trends: RevenueTrend[] | null;
  forecast: RevenueForecast[] | null;
  breakdown: RevenueBreakdown[] | null;
  channelRevenue: ChannelRevenue[] | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  exportData: (format: 'csv' | 'json') => Promise<void>;
}

export const useRevenueData = ({
  userId,
  channelId,
  dateRange,
  period = 'daily',
  breakdownBy = 'source',
}: UseRevenueDataProps): UseRevenueDataReturn => {
  const [overview, setOverview] = useState<RevenueOverview | null>(null);
  const [trends, setTrends] = useState<RevenueTrend[] | null>(null);
  const [forecast, setForecast] = useState<RevenueForecast[] | null>(null);
  const [breakdown, setBreakdown] = useState<RevenueBreakdown[] | null>(null);
  const [channelRevenue, setChannelRevenue] = useState<ChannelRevenue[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { token } = useAuthStore();

  const fetchRevenueData = useCallback(async () => {
    if (!token) {
      setError('Authentication required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Prepare query parameters
      const params: any = {};
      if (dateRange) {
        params.start_date = dateRange.start.toISOString();
        params.end_date = dateRange.end.toISOString();
      }

      // Fetch overview
      const overviewResponse = await apiClient.get('/api/v1/revenue/overview', {
        params,
        headers: { Authorization: `Bearer ${token}` },
      });
      setOverview(overviewResponse.data);

      // Fetch trends
      const trendsResponse = await apiClient.get('/api/v1/revenue/trends', {
        params: {
          period,
          lookback_days: dateRange ? 
            Math.ceil((dateRange.end.getTime() - dateRange.start.getTime()) / (1000 * 60 * 60 * 24)) : 
            30,
        },
        headers: { Authorization: `Bearer ${token}` },
      });
      setTrends(trendsResponse.data.trends);

      // Fetch forecast
      const forecastResponse = await apiClient.get('/api/v1/revenue/forecast', {
        params: { forecast_days: 7 },
        headers: { Authorization: `Bearer ${token}` },
      });
      setForecast(forecastResponse.data.forecast);

      // Fetch breakdown
      const breakdownResponse = await apiClient.get('/api/v1/revenue/breakdown', {
        params: { breakdown_by: breakdownBy },
        headers: { Authorization: `Bearer ${token}` },
      });
      setBreakdown(breakdownResponse.data.breakdown);

      // Fetch channel revenue if channelId is provided
      if (channelId) {
        const channelResponse = await apiClient.get(`/api/v1/revenue/channels/${channelId}`, {
          params,
          headers: { Authorization: `Bearer ${token}` },
        });
        setChannelRevenue([channelResponse.data]);
      } else if (overviewResponse.data.channel_breakdown) {
        // Use channel breakdown from overview
        setChannelRevenue(
          overviewResponse.data.channel_breakdown.map((ch: any) => ({
            channel_id: ch.channel_id,
            channel_name: ch.channel_name,
            total_revenue: ch.revenue,
            video_count: 0, // Would need additional API call for full details
            average_revenue_per_video: 0,
            total_views: 0,
          }))
        );
      }

    } catch (err: any) {
      console.error('Error fetching revenue data:', err);
      setError(err.response?.data?.detail || 'Failed to fetch revenue data');
    } finally {
      setLoading(false);
    }
  }, [token, channelId, dateRange, period, breakdownBy]);

  const exportData = useCallback(async (format: 'csv' | 'json') => {
    if (!token) {
      setError('Authentication required');
      return;
    }

    try {
      const params: any = { format };
      if (dateRange) {
        params.start_date = dateRange.start.toISOString();
        params.end_date = dateRange.end.toISOString();
      }

      const response = await apiClient.get('/api/v1/revenue/export', {
        params,
        headers: { Authorization: `Bearer ${token}` },
        responseType: format === 'csv' ? 'blob' : 'json',
      });

      // Create download link
      const blob = format === 'csv' 
        ? response.data 
        : new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
      
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `revenue_export_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error('Error exporting revenue data:', err);
      setError(err.response?.data?.detail || 'Failed to export revenue data');
    }
  }, [token, dateRange]);

  useEffect(() => {
    fetchRevenueData();
  }, [fetchRevenueData]);

  return {
    overview,
    trends,
    forecast,
    breakdown,
    channelRevenue,
    loading,
    error,
    refetch: fetchRevenueData,
    exportData,
  };
};