import React from 'react';
import { 
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap
  Tooltip
} from 'recharts';
import { 
  Box,
  Paper,
  Typography,
  useTheme
} from '@mui/material';
import { format } from 'date-fns';

// Color palette
const COLORS = ['#667eea', '#764ba2', '#f093fb', '#fda085', '#84fab0', '#8fd3f4'];

interface ChartProps {
  title?: string;
  data: any[];
  height?: number;
}

export const ViewsLineChart: React.FC<ChartProps> = ({ title = 'Views Over Time', data, height = 300 }) => {
  const theme = useTheme();
  
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            dataKey="date" 
            tickFormatter={(date) => format(new Date(date), 'MMM dd')}
            stroke={theme.palette.text.secondary}
          />
          <YAxis stroke={theme.palette.text.secondary} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="views" 
            stroke="#667eea" 
            strokeWidth={2}
            dot={{ fill: '#667eea', r: 4 }}
            activeDot={{ r: 6 }}
          />
          <Line 
            type="monotone" 
            dataKey="projectedViews" 
            stroke="#764ba2" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export const RevenueAreaChart: React.FC<ChartProps> = ({ title = 'Revenue Trend', data, height = 300 }) => {
  const theme = useTheme();
  
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#667eea" stopOpacity={0.1}/>
            </linearGradient>
            <linearGradient id="colorCost" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f093fb" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#f093fb" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            dataKey="date" 
            tickFormatter={(date) => format(new Date(date), 'MMM dd')}
            stroke={theme.palette.text.secondary}
          />
          <YAxis stroke={theme.palette.text.secondary} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`
            }}
            formatter={(value: number) => `$${value.toFixed(2)}`}
          />
          <Legend />
          <Area type="monotone" dataKey="revenue" stroke="#667eea" fillOpacity={1} fill="url(#colorRevenue)" />
          <Area type="monotone" dataKey="cost" stroke="#f093fb" fillOpacity={1} fill="url(#colorCost)" />
        </AreaChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export const VideoPerformanceBar: React.FC<ChartProps> = ({ title = 'Video Performance', data, height = 300 }) => {
  const theme = useTheme();
  
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            dataKey="title" 
            angle={-45}
            textAnchor="end"
            height={100}
            stroke={theme.palette.text.secondary}
          />
          <YAxis stroke={theme.palette.text.secondary} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`
            }}
          />
          <Legend />
          <Bar dataKey="views" fill="#667eea" />
          <Bar dataKey="likes" fill="#764ba2" />
          <Bar dataKey="comments" fill="#f093fb" />
        </BarChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export const ChannelDistributionPie: React.FC<ChartProps> = ({ title = 'Channel Distribution', data, height = 300 }) => {
  const theme = useTheme();
  
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={(entry) => `${entry.name}: ${entry.value}`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip 
            contentStyle={{ 
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export const MetricsRadar: React.FC<ChartProps> = ({ title = 'Performance Metrics', data, height = 300 }) => {
  const theme = useTheme();
  
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <ResponsiveContainer width="100%" height={height}>
        <RadarChart data={data}>
          <PolarGrid stroke={theme.palette.divider} />
          <PolarAngleAxis dataKey="metric" stroke={theme.palette.text.secondary} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} stroke={theme.palette.text.secondary} />
          <Radar name="Current" dataKey="current" stroke="#667eea" fill="#667eea" fillOpacity={0.6} />
          <Radar name="Target" dataKey="target" stroke="#764ba2" fill="#764ba2" fillOpacity={0.3} />
          <Legend />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export const ContentTreemap: React.FC<ChartProps> = ({ title = 'Content Categories', data, height = 300 }) => {
  const theme = useTheme();
  
  const CustomContent = (props: any) => {
    const { root, depth, x, y, width, height, index, colors, name, value } = props;
    
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          style={{
            fill: COLORS[index % COLORS.length],
            stroke: '#fff',
            strokeWidth: 2 / (depth + 1e-10),
            strokeOpacity: 1 / (depth + 1e-10)
          }}
        />
        {depth === 1 && width > 50 && height > 30 && (
          <>
            <text
              x={x + width / 2}
              y={y + height / 2 - 7}
              textAnchor="middle"
              fill="#fff"
              fontSize={14}
            >
              {name}
            </text>
      <text
              x={x + width / 2}
              y={y + height / 2 + 7}
              textAnchor="middle"
              fill="#fff"
              fontSize={12}
            >
              {value}
            </text>
          </>
        )}
      </g>
    );
  };
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <ResponsiveContainer width="100%" height={height}>
        <Treemap
          data={data}
          dataKey="size"
          aspectRatio={4 / 3}
          stroke="#fff"
          fill="#8884d8"
          content={<CustomContent />}
        />
      </ResponsiveContainer>
    </Paper>
  );
};
export const RealTimeMetrics: React.FC<{ data: any[] }> = ({ data }) => {
  const theme = useTheme();
  
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>Real-Time Metrics</Typography>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            dataKey="time" 
            tickFormatter={(time) => format(new Date(time), 'HH:mm:ss')}
            stroke={theme.palette.text.secondary}
          />
          <YAxis stroke={theme.palette.text.secondary} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`
            }}
          />
          <Line 
            type="monotone" 
            dataKey="value" 
            stroke="#667eea" 
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
};
// Composite dashboard component
export const DashboardCharts: React.FC<{ metrics: any }> = ({ metrics }) => {
  return (
    <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))' }}>
      <ViewsLineChart data={metrics.viewsData} />
      <RevenueAreaChart data={metrics.revenueData} />
      <VideoPerformanceBar data={metrics.videoPerformance} />
      <ChannelDistributionPie data={metrics.channelDistribution} />
      <MetricsRadar data={metrics.performanceMetrics} />
      <ContentTreemap data={metrics.contentCategories} />
    </Box>
  );
};
// Export the new Channel Performance Charts component
export { default as ChannelPerformanceCharts } from './ChannelPerformanceCharts';
