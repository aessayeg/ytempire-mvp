# YTEMPIRE Reference Guide

## 7.1 API Documentation (Gap - Needs Backend Team Input)

### Proposed API Structure

```yaml
base_url: https://api.ytempire.com/v1
authentication: Bearer JWT Token
rate_limiting: 1000 requests/hour per user
```

### Authentication Endpoints

```typescript
// POST /auth/register
interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
}

interface RegisterResponse {
  user: User;
  token: string;
  refreshToken: string;
}

// POST /auth/login
interface LoginRequest {
  email: string;
  password: string;
}

interface LoginResponse {
  user: User;
  token: string;
  refreshToken: string;
}

// POST /auth/refresh
interface RefreshRequest {
  refreshToken: string;
}

interface RefreshResponse {
  token: string;
  refreshToken: string;
}

// POST /auth/logout
interface LogoutRequest {
  refreshToken: string;
}
```

### Channel Endpoints

```typescript
// GET /channels
interface GetChannelsResponse {
  channels: Channel[];
  total: number;
}

// GET /channels/:id
interface GetChannelResponse {
  channel: Channel;
}

// POST /channels
interface CreateChannelRequest {
  name: string;
  niche: string;
  description?: string;
  uploadSchedule: 'daily' | 'frequent' | 'moderate' | 'light';
  contentPreferences: {
    videoLength: 'short' | 'medium' | 'long';
    style: 'educational' | 'entertainment' | 'news' | 'review';
  };
}

interface CreateChannelResponse {
  channel: Channel;
}

// PUT /channels/:id
interface UpdateChannelRequest {
  name?: string;
  description?: string;
  uploadSchedule?: string;
  status?: 'active' | 'paused';
}

// DELETE /channels/:id
// No request body, returns 204 No Content

// POST /channels/:id/pause
// POST /channels/:id/resume
// Returns updated Channel object
```

### Video Endpoints

```typescript
// GET /videos
interface GetVideosRequest {
  channelId?: string;
  status?: 'queued' | 'processing' | 'completed' | 'failed';
  limit?: number;
  offset?: number;
}

interface GetVideosResponse {
  videos: Video[];
  total: number;
  hasMore: boolean;
}

// POST /videos/generate
interface GenerateVideoRequest {
  channelId: string;
  topic?: string; // Optional, AI will choose if not provided
  priority: 'high' | 'normal' | 'low';
  scheduledFor?: string; // ISO date
}

interface GenerateVideoResponse {
  video: Video;
  estimatedCost: number;
  estimatedCompletionTime: string;
}

// GET /videos/:id
// GET /videos/:id/status
// DELETE /videos/:id
```

### Analytics Endpoints

```typescript
// GET /analytics/dashboard
interface DashboardResponse {
  revenue: {
    total: number;
    today: number;
    week: number;
    month: number;
  };
  channels: {
    active: number;
    total: number;
  };
  videos: {
    today: number;
    week: number;
    total: number;
  };
  costs: {
    today: number;
    averagePerVideo: number;
  };
}

// GET /analytics/channels/:id
interface ChannelAnalyticsResponse {
  views: TimeSeriesData[];
  revenue: TimeSeriesData[];
  subscribers: TimeSeriesData[];
  topVideos: Video[];
}

// GET /analytics/costs
interface CostAnalyticsResponse {
  breakdown: {
    ai: number;
    voice: number;
    storage: number;
    api: number;
  };
  timeline: TimeSeriesData[];
  perChannel: ChannelCost[];
}
```

### WebSocket Events

```typescript
// WebSocket connection
// wss://api.ytempire.com/ws?token={jwt_token}

// Event Types (Only 3 for MVP)
interface WebSocketMessage {
  type: 'video.completed' | 'video.failed' | 'cost.alert';
  payload: any;
  timestamp: string;
}

// video.completed
interface VideoCompletedPayload {
  videoId: string;
  channelId: string;
  title: string;
  youtubeUrl: string;
  cost: number;
}

// video.failed
interface VideoFailedPayload {
  videoId: string;
  channelId: string;
  error: string;
  canRetry: boolean;
}

// cost.alert
interface CostAlertPayload {
  type: 'approaching_limit' | 'exceeded_limit';
  current: number;
  limit: number;
  message: string;
}
```

*Note: Complete API documentation to be provided by Backend Team*

## 7.2 Component Reference

### Component Catalog (35 Components)

#### Base Components (10)

```typescript
// 1. Button
import { Button } from '@mui/material';

<Button 
  variant="contained" // 'contained' | 'outlined' | 'text'
  color="primary"     // 'primary' | 'secondary' | 'error' | 'warning' | 'success'
  size="medium"       // 'small' | 'medium' | 'large'
  startIcon={<AddIcon />}
  onClick={handleClick}
  disabled={isLoading}
>
  Generate Video
</Button>

// 2. TextField
<TextField
  label="Channel Name"
  value={channelName}
  onChange={(e) => setChannelName(e.target.value)}
  error={!!error}
  helperText={error || "3-50 characters"}
  required
  fullWidth
/>

// 3. Select
<Select
  value={niche}
  onChange={handleNicheChange}
  displayEmpty
>
  <MenuItem value="">Select a niche</MenuItem>
  <MenuItem value="tech">Technology</MenuItem>
  <MenuItem value="gaming">Gaming</MenuItem>
  <MenuItem value="cooking">Cooking</MenuItem>
</Select>

// 4. Checkbox
<FormControlLabel
  control={
    <Checkbox 
      checked={autoGenerate}
      onChange={(e) => setAutoGenerate(e.target.checked)}
    />
  }
  label="Auto-generate videos daily"
/>

// 5. Radio
<RadioGroup value={videoLength} onChange={handleLengthChange}>
  <FormControlLabel value="short" control={<Radio />} label="Short (3-5 min)" />
  <FormControlLabel value="medium" control={<Radio />} label="Medium (8-10 min)" />
  <FormControlLabel value="long" control={<Radio />} label="Long (15+ min)" />
</RadioGroup>

// 6. Switch
<FormControlLabel
  control={<Switch checked={isPaused} onChange={handlePauseToggle} />}
  label="Pause channel"
/>

// 7. Chip
<Chip 
  label="Processing"
  color="info"
  size="small"
  icon={<CircularProgress size={16} />}
/>

// 8. Badge
<Badge badgeContent={videoCount} color="primary">
  <VideoLibraryIcon />
</Badge>

// 9. Avatar
<Avatar 
  alt={channel.name}
  src={channel.thumbnail}
  sx={{ width: 56, height: 56 }}
>
  {channel.name[0]}
</Avatar>

// 10. Tooltip
<Tooltip title="Cost includes AI, voice, and processing fees">
  <IconButton>
    <InfoIcon />
  </IconButton>
</Tooltip>
```

#### Layout Components (5)

```typescript
// 11. Card
<Card sx={{ minWidth: 275 }}>
  <CardContent>
    <Typography variant="h5">{channel.name}</Typography>
    <Typography color="text.secondary">
      {channel.videoCount} videos • ${channel.revenue}
    </Typography>
  </CardContent>
  <CardActions>
    <Button size="small">Generate Video</Button>
    <Button size="small">View Analytics</Button>
  </CardActions>
</Card>

// 12. Container
<Container maxWidth="lg">
  {/* Page content */}
</Container>

// 13. Grid
<Grid container spacing={3}>
  <Grid item xs={12} md={6} lg={3}>
    <MetricCard />
  </Grid>
</Grid>

// 14. Stack
<Stack direction="row" spacing={2} alignItems="center">
  <Avatar />
  <Typography>Channel Name</Typography>
  <Chip label="Active" />
</Stack>

// 15. Divider
<Divider sx={{ my: 2 }} />
```

#### Chart Components (5)

```typescript
// 16. LineChart - Revenue Trend
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

<ResponsiveContainer width="100%" height={300}>
  <LineChart data={revenueData}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="date" />
    <YAxis />
    <Tooltip formatter={(value) => `$${value}`} />
    <Line type="monotone" dataKey="revenue" stroke="#2196F3" />
  </LineChart>
</ResponsiveContainer>

// 17. BarChart - Channel Comparison
<BarChart data={channelData}>
  <CartesianGrid strokeDasharray="3 3" />
  <XAxis dataKey="channel" />
  <YAxis />
  <Tooltip />
  <Bar dataKey="videos" fill="#2196F3" />
  <Bar dataKey="revenue" fill="#FF9800" />
</BarChart>

// 18. PieChart - Cost Breakdown
<PieChart>
  <Pie
    data={costData}
    cx="50%"
    cy="50%"
    outerRadius={80}
    fill="#8884d8"
    dataKey="value"
    label
  >
    {costData.map((entry, index) => (
      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
    ))}
  </Pie>
  <Tooltip />
</PieChart>

// 19. AreaChart - Video Performance
<AreaChart data={performanceData}>
  <CartesianGrid strokeDasharray="3 3" />
  <XAxis dataKey="time" />
  <YAxis />
  <Tooltip />
  <Area type="monotone" dataKey="views" stroke="#2196F3" fill="#2196F3" fillOpacity={0.3} />
</AreaChart>

// 20. Custom Gauge - Cost per Video
const CostGauge = ({ value, max = 3 }) => {
  const percentage = (value / max) * 100;
  const color = value < 2.5 ? '#4CAF50' : value < 2.9 ? '#FF9800' : '#F44336';
  
  return (
    <Box sx={{ position: 'relative', width: 200, height: 100 }}>
      <CircularProgress
        variant="determinate"
        value={percentage}
        size={180}
        thickness={8}
        sx={{ color }}
      />
      <Typography variant="h4" sx={{ position: 'absolute', top: '40%', left: '50%', transform: 'translate(-50%, -50%)' }}>
        ${value}
      </Typography>
    </Box>
  );
};