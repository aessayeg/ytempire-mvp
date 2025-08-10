# YTEMPIRE Analytics Engineer Documentation
## 7. REFERENCE

**Document Version**: 1.0  
**Date**: January 2025  
**Status**: CONSOLIDATED - PRODUCTION READY  
**Purpose**: Complete reference documentation for Analytics Engineering

---

## 7.1 API Documentation

### Frontend API Specifications

```typescript
// API Endpoints for Analytics Data Access

interface AnalyticsAPI {
  baseURL: 'https://api.ytempire.com/v1/analytics';
  
  endpoints: {
    // Metrics Endpoints
    metrics: {
      // Get channel metrics
      getChannelMetrics: {
        method: 'GET';
        path: '/metrics/channel/{channel_id}';
        params: {
          channel_id: string;
          start_date?: string;  // ISO 8601
          end_date?: string;    // ISO 8601
          metrics?: string[];   // ['views', 'revenue', 'engagement']
          granularity?: 'hour' | 'day' | 'week' | 'month';
        };
        response: {
          channel_id: string;
          metrics: {
            timestamp: string;
            views: number;
            revenue: number;
            engagement_rate: number;
            [key: string]: any;
          }[];
          period: {
            start: string;
            end: string;
          };
        };
      };
      
      // Get user metrics
      getUserMetrics: {
        method: 'GET';
        path: '/metrics/user/{user_id}';
        params: {
          user_id: string;
          include_channels?: boolean;
        };
        response: {
          user_id: string;
          current_month_revenue: number;
          goal_progress: number;
          lifetime_revenue: number;
          channels?: ChannelMetric[];
        };
      };
      
      // Execute custom query
      customQuery: {
        method: 'POST';
        path: '/analytics/custom_query';
        headers: {
          'Authorization': 'Bearer {token}';
        };
        body: {
          query: string;
          params?: Record<string, any>;
          timeout?: number;  // seconds
        };
        response: {
          query: string;
          results: any[];
          row_count: number;
          execution_time_ms: number;
        };
      };
    };
    
    // Dashboard Endpoints
    dashboards: {
      // List dashboards
      listDashboards: {
        method: 'GET';
        path: '/dashboards';
        response: {
          dashboards: {
            id: string;
            name: string;
            type: 'executive' | 'operational' | 'user';
            last_updated: string;
          }[];
        };
      };
      
      // Get dashboard data
      getDashboardData: {
        method: 'GET';
        path: '/dashboards/{dashboard_id}/data';
        params: {
          dashboard_id: string;
          refresh?: boolean;
        };
        response: {
          dashboard_id: string;
          data: Record<string, any>;
          cached: boolean;
          timestamp: string;
        };
      };
    };
    
    // Reports Endpoints
    reports: {
      // Generate report
      generateReport: {
        method: 'POST';
        path: '/reports/generate';
        body: {
          report_type: 'daily' | 'weekly' | 'monthly' | 'custom';
          parameters: {
            start_date: string;
            end_date: string;
            user_id?: string;
            channels?: string[];
            metrics?: string[];
          };
          format: 'json' | 'csv' | 'pdf';
        };
        response: {
          report_id: string;
          status: 'pending' | 'processing' | 'completed' | 'failed';
          download_url?: string;
        };
      };
      
      // Get report status
      getReportStatus: {
        method: 'GET';
        path: '/reports/{report_id}/status';
        response: {
          report_id: string;
          status: string;
          progress: number;
          download_url?: string;
          error?: string;
        };
      };
    };
  };
}
```

### Backend Database API

```python
# database_api.py
"""
Database API for Analytics Access
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, date
import psycopg2
from psycopg2.extras import RealDictCursor

class AnalyticsDB:
    """
    Analytics database interface
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._connection = None
        
    @property
    def connection(self):
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor
            )
        return self._connection
    
    def get_channel_metrics(
        self,
        channel_id: str,
        start_date: date,
        end_date: date,
        metrics: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get channel performance metrics
        
        Args:
            channel_id: YouTube channel ID
            start_date: Start date for metrics
            end_date: End date for metrics
            metrics: List of metrics to retrieve
            
        Returns:
            List of metric dictionaries
        """
        if metrics is None:
            metrics = ['views', 'revenue', 'engagement_rate']
        
        metric_columns = ', '.join([f"MAX({m}) as {m}" for m in metrics])
        
        query = f"""
            SELECT 
                DATE(timestamp) as date,
                {metric_columns}
            FROM analytics.fact_video_performance vp
            JOIN analytics.dim_channel c ON vp.channel_key = c.channel_key
            WHERE c.channel_id = %s
                AND timestamp >= %s
                AND timestamp < %s + INTERVAL '1 day'
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (channel_id, start_date, end_date))
            return cursor.fetchall()
    
    def get_user_revenue(
        self,
        user_id: str,
        period: str = 'current_month'
    ) -> Dict[str, Any]:
        """
        Get user revenue metrics
        
        Args:
            user_id: User UUID
            period: Time period for revenue calculation
            
        Returns:
            Revenue metrics dictionary
        """
        query = """
            SELECT 
                user_id,
                current_month_revenue,
                goal_progress_pct,
                revenue_30d,
                lifetime_revenue,
                days_to_10k_goal
            FROM analytics.v_user_revenue_tracking
            WHERE user_id = %s
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
    
    def execute_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Execute custom analytics query
        
        Args:
            query: SQL query to execute
            params: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query results as list of dictionaries
        """
        # Apply row-level security
        query = self._apply_security(query)
        
        with self.connection.cursor() as cursor:
            # Set statement timeout
            cursor.execute(f"SET statement_timeout = '{timeout}s'")
            
            # Execute query
            cursor.execute(query, params)
            
            # Reset timeout
            cursor.execute("RESET statement_timeout")
            
            return cursor.fetchall()
    
    def _apply_security(self, query: str) -> str:
        """Apply row-level security to query"""
        # Implementation depends on security requirements
        return query
```

---

## 7.2 Query Library

### Executive Queries

```sql
-- ============================================
-- EXECUTIVE QUERY LIBRARY
-- ============================================

-- 1. Company Overview Dashboard
WITH company_metrics AS (
    SELECT 
        COUNT(DISTINCT u.user_id) as total_users,
        COUNT(DISTINCT c.channel_key) as total_channels,
        COUNT(DISTINCT v.video_key) as total_videos,
        SUM(vp.views) as total_views,
        SUM(r.total_revenue_cents) / 100.0 as total_revenue,
        SUM(fc.total_cost_cents) / 100.0 as total_costs
    FROM analytics.dim_user u
    LEFT JOIN analytics.dim_channel c ON u.user_id = c.user_id
    LEFT JOIN analytics.dim_video v ON c.channel_id = v.channel_id
    LEFT JOIN analytics.fact_video_performance vp ON v.video_key = vp.video_key
    LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key
    LEFT JOIN analytics.fact_costs fc ON v.video_id = fc.video_id
    WHERE vp.timestamp >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT 
    total_users,
    total_channels,
    total_videos,
    total_views,
    total_revenue,
    total_costs,
    (total_revenue - total_costs) as profit,
    ROUND((total_revenue - total_costs) / NULLIF(total_costs, 0) * 100, 2) as roi_percentage,
    ROUND(total_revenue / NULLIF(total_views, 0) * 1000, 2) as rpm,
    ROUND(total_costs / NULLIF(total_videos, 0), 2) as cost_per_video
FROM company_metrics;

-- 2. Top Performing Channels
SELECT 
    c.channel_name,
    c.niche,
    u.email as owner_email,
    COUNT(DISTINCT v.video_key) as video_count,
    SUM(vp.views) as total_views,
    AVG(vp.engagement_rate) as avg_engagement,
    SUM(r.total_revenue_cents) / 100.0 as revenue,
    SUM(vp.cost_cents) / 100.0 as costs,
    (SUM(r.total_revenue_cents) - SUM(vp.cost_cents)) / 100.0 as profit,
    RANK() OVER (ORDER BY SUM(r.total_revenue_cents) - SUM(vp.cost_cents) DESC) as profit_rank
FROM analytics.dim_channel c
JOIN analytics.dim_user u ON c.user_id = u.user_id
JOIN analytics.dim_video v ON c.channel_id = v.channel_id
JOIN analytics.fact_video_performance vp ON v.video_key = vp.video_key
JOIN analytics.fact_revenue r ON v.video_key = r.video_key
WHERE vp.timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY c.channel_name, c.niche, u.email
ORDER BY profit DESC
LIMIT 20;

-- 3. Revenue Trend Analysis
WITH daily_revenue AS (
    SELECT 
        d.full_date,
        SUM(r.total_revenue_cents) / 100.0 as revenue,
        SUM(fc.total_cost_cents) / 100.0 as costs,
        COUNT(DISTINCT v.video_key) as videos_published
    FROM analytics.dim_date d
    LEFT JOIN analytics.fact_revenue r ON d.date_key = r.date_key
    LEFT JOIN analytics.fact_costs fc ON d.date_key = fc.date_key
    LEFT JOIN analytics.dim_video v ON DATE(v.published_at) = d.full_date
    WHERE d.full_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY d.full_date
)
SELECT 
    full_date,
    revenue,
    costs,
    videos_published,
    revenue - costs as profit,
    AVG(revenue) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as revenue_7d_ma,
    AVG(revenue) OVER (ORDER BY full_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as revenue_30d_ma,
    SUM(revenue) OVER (ORDER BY full_date) as cumulative_revenue
FROM daily_revenue
ORDER BY full_date DESC;
```

### Operational Queries

```sql
-- ============================================
-- OPERATIONAL QUERY LIBRARY
-- ============================================

-- 1. Real-time Performance Monitor
SELECT 
    DATE_TRUNC('hour', vp.timestamp) as hour,
    COUNT(DISTINCT v.video_key) as videos_published,
    SUM(vp.views) as hourly_views,
    AVG(vp.engagement_rate) as avg_engagement,
    SUM(vp.actual_revenue_cents) / 100.0 as hourly_revenue,
    COUNT(DISTINCT c.channel_key) as active_channels
FROM analytics.fact_video_performance vp
JOIN analytics.dim_video v ON vp.video_key = v.video_key
JOIN analytics.dim_channel c ON v.channel_id = c.channel_id
WHERE vp.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', vp.timestamp)
ORDER BY hour DESC;

-- 2. Cost Monitoring
SELECT 
    fc.cost_category,
    DATE_TRUNC('hour', fc.cost_timestamp) as hour,
    COUNT(DISTINCT fc.video_id) as videos,
    SUM(fc.total_cost_cents) / 100.0 as total_cost,
    AVG(fc.total_cost_cents) / 100.0 as avg_cost,
    MAX(fc.total_cost_cents) / 100.0 as max_cost,
    CASE 
        WHEN AVG(fc.total_cost_cents) > 300 THEN 'OVER_BUDGET'
        WHEN AVG(fc.total_cost_cents) > 250 THEN 'WARNING'
        ELSE 'OK'
    END as budget_status
FROM analytics.fact_costs fc
WHERE fc.cost_timestamp >= NOW() - INTERVAL '6 hours'
GROUP BY fc.cost_category, DATE_TRUNC('hour', fc.cost_timestamp)
ORDER BY hour DESC, total_cost DESC;

-- 3. API Quota Usage
SELECT 
    api_method,
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as api_calls,
    SUM(quota_cost) as quota_used,
    AVG(response_time_ms) as avg_response_time,
    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors,
    10000 - SUM(quota_cost) OVER (PARTITION BY DATE(timestamp)) as daily_quota_remaining
FROM analytics.youtube_api_usage
WHERE timestamp >= CURRENT_DATE
GROUP BY api_method, DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC, quota_used DESC;

-- 4. Video Pipeline Status
WITH pipeline_status AS (
    SELECT 
        v.video_key,
        v.video_title,
        c.channel_name,
        v.published_at,
        COALESCE(vp.views, 0) as views,
        COALESCE(vp.engagement_rate, 0) as engagement,
        COALESCE(r.total_revenue_cents, 0) / 100.0 as revenue,
        CASE 
            WHEN vp.timestamp IS NULL THEN 'PENDING_METRICS'
            WHEN r.total_revenue_cents IS NULL THEN 'PENDING_REVENUE'
            WHEN vp.views < 100 THEN 'LOW_VIEWS'
            WHEN vp.engagement_rate < 0.02 THEN 'LOW_ENGAGEMENT'
            ELSE 'HEALTHY'
        END as status
    FROM analytics.dim_video v
    LEFT JOIN analytics.dim_channel c ON v.channel_id = c.channel_id
    LEFT JOIN analytics.fact_video_performance vp ON v.video_key = vp.video_key
    LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key
    WHERE v.published_at >= CURRENT_DATE - INTERVAL '24 hours'
)
SELECT 
    status,
    COUNT(*) as video_count,
    STRING_AGG(video_title, ', ' ORDER BY published_at DESC) as videos
FROM pipeline_status
GROUP BY status
ORDER BY 
    CASE status
        WHEN 'PENDING_METRICS' THEN 1
        WHEN 'PENDING_REVENUE' THEN 2
        WHEN 'LOW_VIEWS' THEN 3
        WHEN 'LOW_ENGAGEMENT' THEN 4
        ELSE 5
    END;
```

### User Queries

```sql
-- ============================================
-- USER-SPECIFIC QUERY LIBRARY
-- ============================================

-- 1. User Dashboard Summary
SELECT 
    u.user_id,
    u.email,
    COUNT(DISTINCT c.channel_key) as channel_count,
    COUNT(DISTINCT v.video_key) as total_videos,
    SUM(CASE WHEN DATE(v.published_at) = CURRENT_DATE THEN 1 ELSE 0 END) as videos_today,
    SUM(CASE WHEN r.date_key = (SELECT date_key FROM analytics.dim_date WHERE full_date = CURRENT_DATE) 
        THEN r.total_revenue_cents ELSE 0 END) / 100.0 as revenue_today,
    SUM(CASE WHEN EXTRACT(MONTH FROM d.full_date) = EXTRACT(MONTH FROM CURRENT_DATE) 
        AND EXTRACT(YEAR FROM d.full_date) = EXTRACT(YEAR FROM CURRENT_DATE)
        THEN r.total_revenue_cents ELSE 0 END) / 100.0 as revenue_this_month,
    10000 - SUM(CASE WHEN EXTRACT(MONTH FROM d.full_date) = EXTRACT(MONTH FROM CURRENT_DATE) 
        AND EXTRACT(YEAR FROM d.full_date) = EXTRACT(YEAR FROM CURRENT_DATE)
        THEN r.total_revenue_cents ELSE 0 END) / 100.0 as revenue_to_goal
FROM analytics.dim_user u
LEFT JOIN analytics.dim_channel c ON u.user_id = c.user_id
LEFT JOIN analytics.dim_video v ON c.channel_id = v.channel_id
LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key
LEFT JOIN analytics.dim_date d ON r.date_key = d.date_key
WHERE u.user_id = %(user_id)s
GROUP BY u.user_id, u.email;

-- 2. Channel Performance by User
SELECT 
    c.channel_name,
    c.niche,
    COUNT(DISTINCT v.video_key) as video_count,
    SUM(vp.views) as total_views,
    AVG(vp.engagement_rate) as avg_engagement,
    SUM(r.total_revenue_cents) / 100.0 as revenue,
    SUM(fc.total_cost_cents) / 100.0 as costs,
    (SUM(r.total_revenue_cents) - SUM(fc.total_cost_cents)) / 100.0 as profit,
    CASE 
        WHEN SUM(fc.total_cost_cents) > 0 
        THEN ((SUM(r.total_revenue_cents) - SUM(fc.total_cost_cents))::FLOAT / SUM(fc.total_cost_cents)) * 100
        ELSE 0 
    END as roi_percentage
FROM analytics.dim_channel c
JOIN analytics.dim_video v ON c.channel_id = v.channel_id
LEFT JOIN analytics.fact_video_performance vp ON v.video_key = vp.video_key
LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key
LEFT JOIN analytics.fact_costs fc ON v.video_id = fc.video_id
WHERE c.user_id = %(user_id)s
    AND vp.timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY c.channel_name, c.niche
ORDER BY profit DESC;

-- 3. Daily Revenue Tracking
SELECT 
    d.full_date,
    COUNT(DISTINCT v.video_key) as videos_published,
    SUM(vp.views) as daily_views,
    SUM(r.total_revenue_cents) / 100.0 as daily_revenue,
    SUM(fc.total_cost_cents) / 100.0 as daily_costs,
    (SUM(r.total_revenue_cents) - SUM(fc.total_cost_cents)) / 100.0 as daily_profit,
    SUM(SUM(r.total_revenue_cents)) OVER (
        ORDER BY d.full_date 
        ROWS UNBOUNDED PRECEDING
    ) / 100.0 as cumulative_revenue
FROM analytics.dim_date d
LEFT JOIN analytics.dim_video v ON DATE(v.published_at) = d.full_date
LEFT JOIN analytics.dim_channel c ON v.channel_id = c.channel_id
LEFT JOIN analytics.fact_video_performance vp ON v.video_key = vp.video_key AND d.date_key = vp.date_key
LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key AND d.date_key = r.date_key
LEFT JOIN analytics.fact_costs fc ON v.video_id = fc.video_id AND d.date_key = fc.date_key
WHERE c.user_id = %(user_id)s
    AND d.full_date >= DATE_TRUNC('month', CURRENT_DATE)
    AND d.full_date <= CURRENT_DATE
GROUP BY d.full_date
ORDER BY d.full_date;
```

---

## 7.3 Configuration Files

### Database Configuration

```yaml
# config/database.yml
development:
  adapter: postgresql
  encoding: unicode
  database: ytempire_dev
  pool: 5
  username: analytics_eng
  password: <%= ENV['DB_PASSWORD'] %>
  host: localhost
  port: 5432
  schema_search_path: "analytics,public"
  
test:
  adapter: postgresql
  encoding: unicode
  database: ytempire_test
  pool: 5
  username: analytics_eng
  password: <%= ENV['DB_PASSWORD'] %>
  host: localhost
  port: 5432
  
production:
  adapter: postgresql
  encoding: unicode
  database: ytempire
  pool: <%= ENV['DB_POOL'] || 25 %>
  username: <%= ENV['DB_USERNAME'] %>
  password: <%= ENV['DB_PASSWORD'] %>
  host: <%= ENV['DB_HOST'] %>
  port: <%= ENV['DB_PORT'] || 5432 %>
  statement_timeout: 30000  # 30 seconds
  connect_timeout: 10
  checkout_timeout: 10
  reaping_frequency: 10
  schema_search_path: "analytics,public"
  prepared_statements: false
```

### Grafana Configuration

```json
{
  "datasources": [
    {
      "name": "PostgreSQL-Analytics",
      "type": "postgres",
      "access": "proxy",
      "url": "localhost:5432",
      "database": "ytempire",
      "user": "grafana_reader",
      "jsonData": {
        "sslmode": "require",
        "maxOpenConns": 10,
        "maxIdleConns": 2,
        "connMaxLifetime": 14400,
        "postgresVersion": 1500,
        "timescaledb": true
      }
    },
    {
      "name": "Redis-Cache",
      "type": "redis-datasource",
      "access": "proxy",
      "url": "redis://localhost:6379",
      "jsonData": {
        "client": "standalone"
      }
    }
  ],
  "dashboards": {
    "default": {
      "refresh": "5m",
      "timezone": "browser",
      "theme": "dark"
    },
    "folders": [
      {
        "name": "Executive",
        "uid": "exec"
      },
      {
        "name": "Operational",
        "uid": "ops"
      },
      {
        "name": "User",
        "uid": "user"
      }
    ]
  }
}
```

### Alert Configuration

```yaml
# config/alerts.yml
alerts:
  data_freshness:
    name: "Data Freshness Check"
    interval: 5m
    conditions:
      - metric: "data_age_minutes"
        operator: ">"
        threshold: 60
        severity: "warning"
      - metric: "data_age_minutes"
        operator: ">"
        threshold: 120
        severity: "critical"
    notifications:
      - channel: "slack"
        template: "Data is {{.Value}} minutes old (threshold: {{.Threshold}})"
      
  cost_overrun:
    name: "Cost Per Video Alert"
    interval: 1h
    conditions:
      - metric: "avg_cost_per_video"
        operator: ">"
        threshold: 3.00
        severity: "warning"
      - metric: "avg_cost_per_video"
        operator: ">"
        threshold: 3.50
        severity: "critical"
    notifications:
      - channel: "email"
        recipients: ["analytics@ytempire.com"]
        template: "Cost per video: ${{.Value}} (threshold: ${{.Threshold}})"
      
  api_quota:
    name: "YouTube API Quota"
    interval: 15m
    conditions:
      - metric: "quota_usage_percentage"
        operator: ">"
        threshold: 75
        severity: "warning"
      - metric: "quota_usage_percentage"
        operator: ">"
        threshold: 90
        severity: "critical"
    notifications:
      - channel: "pagerduty"
        integration_key: "${PAGERDUTY_KEY}"
```

---

## 7.4 Troubleshooting FAQ

### Common Issues

```markdown
# Frequently Asked Questions

## Query Performance Issues

### Q: My dashboard query is taking too long (>5 seconds)
**A:** Follow these steps:
1. Run EXPLAIN ANALYZE on the query
2. Check for missing indexes
3. Consider using materialized views
4. Verify statistics are up to date
5. Check for table bloat

**Quick Fix:**
```sql
-- Update statistics
ANALYZE analytics.fact_video_performance;

-- Check for missing indexes
SELECT * FROM analytics.tune_performance();

-- Refresh materialized views
SELECT analytics.refresh_materialized_views();
```

### Q: Getting "statement timeout" errors
**A:** The query is exceeding the 30-second timeout. Options:
1. Optimize the query
2. Add appropriate indexes
3. Use materialized views
4. Break into smaller queries
5. Request timeout increase (last resort)

## Data Quality Issues

### Q: Revenue numbers don't match YouTube Analytics
**A:** Check these common causes:
1. Data freshness - may be delayed
2. Time zone differences
3. Currency conversion
4. Include/exclude filters
5. Attribution window differences

**Verification Query:**
```sql
SELECT * FROM analytics.check_data_accuracy();
```

### Q: Missing data for recent videos
**A:** Typical causes:
1. ETL pipeline delay (check logs)
2. YouTube API quota exceeded
3. Video still processing
4. Network/connectivity issues

**Check Pipeline Status:**
```sql
SELECT * FROM analytics.v_pipeline_status
WHERE status != 'HEALTHY';
```

## Dashboard Issues

### Q: Grafana dashboard showing "No Data"
**A:** Troubleshooting steps:
1. Test data source connection
2. Check query in SQL client
3. Verify time range selection
4. Check dashboard variables
5. Review query syntax

### Q: Dashboard panels loading slowly
**A:** Optimization options:
1. Reduce refresh frequency
2. Optimize underlying queries
3. Use query caching
4. Limit data range
5. Simplify visualizations

## Access Issues

### Q: Can't connect to database
**A:** Check:
1. VPN connection (if required)
2. Database credentials
3. Network connectivity
4. Database server status
5. User permissions

**Test Connection:**
```bash
psql -h localhost -U analytics_eng -d ytempire -c "SELECT 1;"
```

### Q: Permission denied errors
**A:** Verify grants:
```sql
-- Check your permissions
SELECT * FROM information_schema.table_privileges
WHERE grantee = CURRENT_USER;
```

## Cost Management

### Q: How to identify high-cost videos?
**A:** Use this query:
```sql
SELECT 
    video_id,
    video_title,
    total_cost,
    revenue,
    profit,
    roi_percentage
FROM analytics.v_cost_analysis
WHERE total_cost > 3.00
ORDER BY total_cost DESC
LIMIT 20;
```

### Q: How to track cost trends?
**A:** Monitor daily costs:
```sql
SELECT * FROM analytics.v_daily_cost_summary
WHERE date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY date DESC;
```

## Maintenance

### Q: When should I vacuum tables?
**A:** Run weekly or when:
- Dead tuple percentage >20%
- Query performance degrades
- After large DELETE operations

### Q: How to identify bloated tables?
**A:** Check bloat:
```sql
SELECT * FROM analytics.tune_performance()
WHERE issue_type = 'TABLE_BLOAT';
```

## Emergency Procedures

### Q: Data pipeline is down
**A:** Immediate actions:
1. Check ETL logs
2. Verify source systems
3. Check API quotas
4. Restart failed jobs
5. Escalate if needed

### Q: Critical dashboard is broken
**A:** Quick recovery:
1. Switch to backup dashboard
2. Check recent changes
3. Revert if necessary
4. Test in development
5. Deploy fix

## Best Practices

### Q: How often should I refresh materialized views?
**A:** Depends on use case:
- Executive dashboards: Every hour
- Operational metrics: Every 15 minutes
- Historical analysis: Daily
- Real-time: Don't use materialized views

### Q: What's the best way to handle time zones?
**A:** Standards:
- Store all timestamps in UTC
- Convert for display only
- Be explicit about time zones
- Document assumptions

### Q: How to ensure data quality?
**A:** Regular checks:
1. Run daily quality checks
2. Monitor data freshness
3. Validate against source
4. Set up alerts
5. Document discrepancies
```

## Contact Information

```yaml
# support_contacts.yml
technical_support:
  data_team:
    slack: "#data-team"
    email: "data-team@ytempire.com"
    oncall: "data-oncall@ytempire.com"
    
  platform_ops:
    slack: "#platform-ops"
    email: "platform@ytempire.com"
    oncall: "ops-oncall@ytempire.com"
    
  emergency:
    pagerduty: "ytempire.pagerduty.com"
    hotline: "+1-XXX-XXX-XXXX"
    
documentation:
  internal_wiki: "wiki.ytempire.com"
  github: "github.com/ytempire"
  confluence: "ytempire.atlassian.net"
  
tools:
  grafana: "grafana.ytempire.com"
  airflow: "airflow.ytempire.com"
  jupyter: "notebooks.ytempire.com"
```