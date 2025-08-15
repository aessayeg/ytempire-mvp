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
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  Brush,
  RadialBarChart,
  RadialBar
} from 'recharts';
import { format } from 'date-fns';

interface ChartProps {
  data: unknown[];
  height?: number;
  width?: string | number;
}

const COLORS = {
  primary: '#3b82f6',
  secondary: '#10b981',
  tertiary: '#f59e0b',
  quaternary: '#ef4444',
  quinary: '#8b5cf6',
  senary: '#ec4899',
  septenary: '#06b6d4',
  octonary: '#64748b'
};

const CHART_COLORS = Object.values(COLORS);

export const RevenueChart: React.FC<ChartProps> = ({ data, height = 300 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.8}/>
            <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis 
          dataKey="date" 
          tickFormatter={(value) => format(new Date(value), 'MMM dd')}
          stroke="#6b7280"
        />
        <YAxis 
          stroke="#6b7280"
          tickFormatter={(value) => `$${value}`}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
          formatter={(value: unknown) => [`$${value}`, 'Revenue']}
          labelFormatter={(label) => format(new Date(label), 'MMM dd, yyyy')}
        />
        <Area 
          type="monotone" 
          dataKey="revenue" 
          stroke={COLORS.primary} 
          fillOpacity={1} 
          fill="url(#colorRevenue)" 
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export const ViewsChart: React.FC<ChartProps> = ({ data, height = 300 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis 
          dataKey="date" 
          tickFormatter={(value) => format(new Date(value), 'MMM dd')}
          stroke="#6b7280"
        />
        <YAxis stroke="#6b7280" />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
          labelFormatter={(label) => format(new Date(label), 'MMM dd, yyyy')}
        />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="views" 
          stroke={COLORS.secondary} 
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 8 }}
        />
        <Line 
          type="monotone" 
          dataKey="uniqueViewers" 
          stroke={COLORS.tertiary} 
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 8 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export const ChannelPerformanceChart: React.FC<ChartProps> = ({ data, height = 300 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="channel" stroke="#6b7280" />
        <YAxis stroke="#6b7280" />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
        />
        <Legend />
        <Bar dataKey="videos" fill={COLORS.primary} radius={[8, 8, 0, 0]} />
        <Bar dataKey="views" fill={COLORS.secondary} radius={[8, 8, 0, 0]} />
        <Bar dataKey="revenue" fill={COLORS.tertiary} radius={[8, 8, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export const ContentDistributionPie: React.FC<ChartProps> = ({ data, height = 300 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={(entry) => `${entry.name}: ${entry.value}%`}
          outerRadius={80}
          fill="#8884d8"
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
};

export const EngagementRadar: React.FC<ChartProps> = ({ data, height = 300 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data}>
        <PolarGrid stroke="#e5e7eb" />
        <PolarAngleAxis dataKey="metric" stroke="#6b7280" />
        <PolarRadiusAxis stroke="#6b7280" />
        <Radar 
          name="Current" 
          dataKey="current" 
          stroke={COLORS.primary} 
          fill={COLORS.primary} 
          fillOpacity={0.6} 
        />
        <Radar 
          name="Target" 
          dataKey="target" 
          stroke={COLORS.secondary} 
          fill={COLORS.secondary} 
          fillOpacity={0.6} 
        />
        <Legend />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
};

export const VideoGenerationTimeline: React.FC<ChartProps> = ({ data, height = 300 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid stroke="#e5e7eb" />
        <XAxis 
          dataKey="time" 
          tickFormatter={(value) => format(new Date(value), 'HH:mm')}
          stroke="#6b7280"
        />
        <YAxis yAxisId="left" orientation="left" stroke="#6b7280" />
        <YAxis yAxisId="right" orientation="right" stroke="#6b7280" />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
          labelFormatter={(label) => format(new Date(label), 'HH:mm:ss')}
        />
        <Legend />
        <Bar yAxisId="left" dataKey="completed" fill={COLORS.secondary} />
        <Bar yAxisId="left" dataKey="failed" fill={COLORS.quaternary} />
        <Line 
          yAxisId="right" 
          type="monotone" 
          dataKey="queueSize" 
          stroke={COLORS.primary} 
          strokeWidth={2}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export const CostBreakdownRadial: React.FC<ChartProps> = ({ data, height = 300 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="80%" data={data}>
        <RadialBar
          minAngle={15}
          label={{ position: 'insideStart', fill: '#fff' }}
          background
          clockWise
          dataKey="cost"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </RadialBar>
        <Legend 
          iconSize={10}
          layout="vertical"
          verticalAlign="middle"
          align="right"
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
          formatter={(value: unknown) => [`$${value}`, 'Cost']}
        />
      </RadialBarChart>
    </ResponsiveContainer>
  );
};

export const TrendAnalysisChart: React.FC<ChartProps & { showBrush?: boolean }> = ({ 
  data, 
  height = 300,
  showBrush = true 
}) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis 
          dataKey="date" 
          tickFormatter={(value) => format(new Date(value), 'MMM dd')}
          stroke="#6b7280"
        />
        <YAxis stroke="#6b7280" />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
          labelFormatter={(label) => format(new Date(label), 'MMM dd, yyyy')}
        />
        <Legend />
        <ReferenceLine y={0} stroke="#6b7280" />
        <Line 
          type="monotone" 
          dataKey="trendScore" 
          stroke={COLORS.primary} 
          strokeWidth={2}
          dot={false}
        />
        <Line 
          type="monotone" 
          dataKey="prediction" 
          stroke={COLORS.secondary} 
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={false}
        />
        {showBrush && (
          <Brush 
            dataKey="date" 
            height={30} 
            stroke={COLORS.primary}
            tickFormatter={(value) => format(new Date(value), 'MM/dd')}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
};

export const RealTimeMetricsChart: React.FC<ChartProps> = ({ data, height = 200 }) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="colorMetric" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.septenary} stopOpacity={0.8}/>
            <stop offset="95%" stopColor={COLORS.septenary} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis 
          dataKey="timestamp" 
          tickFormatter={(value) => format(new Date(value), 'HH:mm:ss')}
          stroke="#6b7280"
        />
        <YAxis stroke="#6b7280" />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6'
          }}
          labelFormatter={(label) => format(new Date(label), 'HH:mm:ss')}
        />
        <Area 
          type="monotone" 
          dataKey="value" 
          stroke={COLORS.septenary} 
          fillOpacity={1} 
          fill="url(#colorMetric)"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};