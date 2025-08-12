# Analytics Dashboard Access Guide

## How to Access the Dashboards

The analytics dashboards are now fully integrated into the YTEmpire MVP application and can be accessed through multiple routes:

### 1. Direct Navigation
Navigate to `/analytics/dashboard` in your browser after logging in.

### 2. Sidebar Menu
1. Look for the **Analytics** section in the sidebar menu
2. Click to expand the Analytics submenu
3. Select **Metrics Dashboard** (marked with a "NEW" badge)

### 3. Dashboard Components Available

The unified analytics dashboard includes 4 main sections accessible via tabs:

#### **Revenue Tracking Tab**
- Real-time revenue metrics
- CPM/RPM tracking
- Channel-wise revenue breakdown
- Revenue forecasting with ML models
- Historical trends and comparisons
- Export functionality for reports

#### **User Behavior Tab**
- Event tracking and analytics
- User session analysis
- Conversion funnels
- Cohort retention analysis
- Heatmap visualizations
- User journey mapping

#### **Performance Tab**
- Real-time system performance metrics
- Request/response time monitoring
- Database query performance
- Resource utilization (CPU, Memory, Disk)
- Slow endpoint identification
- Error rate tracking and alerts

#### **A/B Testing Tab**
- Create and manage experiments
- Statistical significance testing
- Variant performance comparison
- Real-time experiment results
- Confidence intervals and p-values
- Winner determination algorithms

## API Endpoints

All dashboards are powered by the following backend APIs:

### Revenue APIs
- `GET /api/v1/revenue/overview` - Revenue overview and metrics
- `GET /api/v1/revenue/forecast` - ML-powered revenue forecasting
- `GET /api/v1/revenue/channels/{channel_id}` - Channel-specific metrics
- `GET /api/v1/revenue/export` - Export revenue data

### User Behavior APIs
- `POST /api/v1/user-behavior/track` - Track user events
- `GET /api/v1/user-behavior/funnels` - Conversion funnel analysis
- `GET /api/v1/user-behavior/cohorts` - Cohort retention data
- `GET /api/v1/user-behavior/heatmaps` - Heatmap data

### Performance APIs
- `GET /api/v1/performance/overview` - Performance overview
- `GET /api/v1/performance/alerts` - Active performance alerts
- `GET /api/v1/performance/endpoint-metrics` - Endpoint-specific metrics
- `GET /api/v1/performance/database` - Database performance metrics

### A/B Testing APIs
- `POST /api/v1/experiments/` - Create new experiment
- `GET /api/v1/experiments/{id}/results` - Get experiment results
- `POST /api/v1/experiments/{id}/start` - Start experiment
- `POST /api/v1/experiments/{id}/conclude` - Conclude experiment

## Current Status

✅ **Fully Implemented:**
- Backend services for all 4 dashboard components
- API endpoints with authentication
- Frontend dashboard components with Material-UI
- Real-time data visualization with Recharts
- WebSocket support for live updates
- Statistical analysis for A/B testing
- Performance monitoring with alerts
- User behavior tracking with auto-instrumentation

## Testing the Dashboards

### Starting the Application

1. **Start the backend:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start the frontend:**
```bash
cd frontend
npm install
npm run dev
```

3. **Access the application:**
- Open http://localhost:3000 in your browser
- Login with your credentials
- Navigate to Analytics > Metrics Dashboard

### Sample Data
The services currently generate sample data for demonstration purposes. In production, these will connect to:
- Prometheus for performance metrics
- PostgreSQL for business data
- Redis for caching and real-time data
- YouTube API for actual channel metrics

## Features Highlights

### Revenue Dashboard
- Interactive charts showing revenue trends
- Forecast visualization with confidence intervals
- Channel performance comparison
- Export functionality for reports

### User Behavior Analytics
- Automatic event tracking on frontend
- Session recording and playback capabilities
- Funnel visualization with drop-off rates
- Cohort retention matrices

### Performance Monitoring
- Real-time performance metrics
- Alert system for critical issues
- Resource utilization gauges
- Slow query identification

### A/B Testing Platform
- Visual experiment creation wizard
- Real-time result updates
- Statistical significance calculations
- Automatic winner determination

## Next Steps for Production

1. **Connect to Real Data Sources:**
   - Integrate with YouTube Analytics API
   - Connect to production PostgreSQL
   - Set up Prometheus metrics collection
   - Configure Redis for production caching

2. **Enable Authentication:**
   - Ensure JWT tokens are properly validated
   - Implement role-based access control
   - Add audit logging for sensitive operations

3. **Performance Optimization:**
   - Implement data aggregation for large datasets
   - Add pagination for table views
   - Optimize chart rendering for large time ranges
   - Enable server-side caching

4. **Monitoring & Alerting:**
   - Set up alert thresholds in environment config
   - Configure email/Slack notifications
   - Implement escalation policies
   - Add custom alert rules

## Troubleshooting

If dashboards are not loading:

1. **Check Backend Status:**
   - Verify backend is running on port 8000
   - Check `/health` endpoint: http://localhost:8000/health
   - Review backend logs for errors

2. **Check Frontend Connection:**
   - Verify VITE_API_URL is set correctly
   - Check browser console for errors
   - Ensure authentication token is valid

3. **Check Database Connection:**
   - Verify PostgreSQL is running
   - Check database migrations are applied
   - Ensure Redis is accessible

4. **Clear Cache:**
   - Clear browser cache
   - Restart Redis if using caching
   - Clear any localStorage data

## Support

For issues or questions:
- Check logs in `backend/logs/`
- Review API documentation at http://localhost:8000/docs
- Check frontend console for errors
- Verify all required environment variables are set