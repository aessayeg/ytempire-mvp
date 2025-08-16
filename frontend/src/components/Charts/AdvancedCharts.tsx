import React, { useMemo } from 'react';
import { 
  Box,
  Card,
  CardContent,
  Typography,
  useTheme,
  Paper,
  Chip,
  IconButton,
  Menu,
  MenuItem
 } from '@mui/material';
import { 
  ResponsiveContainer,
  Treemap,
  Tooltip as RechartsTooltip,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  FunnelChart,
  Funnel,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Legend
 } from 'recharts';
import {  HeatMapGrid  } from 'react-grid-heatmap';
import {  MoreVert, Download, Refresh  } from '@mui/icons-material';

interface ChartProps {
  
title: string;
subtitle?: string;
data: Record<string, unknown>[][];
height?: number;
onRefresh?: () => void;
onExport?: () => void;
onFullscreen?: () => void;


}

// Heatmap Chart Component
export const HeatmapChart: React.FC<ChartProps & {
  xLabels: string[];,

  yLabels: string[];
  cellRenderCallback?: (x: number, y: number, value: number) => void}> = ({ title, subtitle, data, xLabels, yLabels, height = 400, cellRenderCallback, onRefresh, onExport }) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  return (
    <>
      <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Box>
            <Typography variant="h6" fontWeight="bold">
              {title}
            </Typography>
            {subtitle && (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
      <IconButton size="small" onClick={(e) => setAnchorEl(e.currentTarget}>
            <MoreVert />
          </IconButton>
        </Box>
        
        <Box sx={{ height, overflow: 'auto' }}>
          <HeatMapGrid
            data={data}
            xLabels={xLabels}
            yLabels={yLabels}
            cellRender={(x, y, value) => (
              <div title={`${xLabels[x]}, ${yLabels[y]}: ${value}`}>
                {value}
              </div>
            )}
            xLabelsStyle={() => ({
              fontSize: '0.75rem',
              color: theme.palette.text.secondary
            })}
            yLabelsStyle={() => ({
              fontSize: '0.75rem',
              color: theme.palette.text.secondary
            })}
            cellStyle={(x, y, ratio) => ({
              background: `rgba(33, 150, 243, ${ratio})`,
              color: ratio > 0.5 ? '#fff' : theme.palette.text.primary;
              fontSize: '0.7rem';
              border: '1px solid rgba(0,0,0,0.1)'})}
            cellHeight="30px"
            square
          />
        </Box>
        
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={() => setAnchorEl(null}
        >
          <MenuItem onClick={onRefresh}>
            <Refresh fontSize="small" sx={{ mr: 1 }} /> Refresh
          </MenuItem>
          <MenuItem onClick={onExport}>
            <Download fontSize="small" sx={{ mr: 1 }} /> Export
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  </>
  )};
// Funnel Chart Component
export const FunnelVisualization: React.FC<ChartProps> = ({ title, subtitle, data, height = 400, onRefresh }) => {
  const theme = useTheme();
  const COLORS = [ theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.success.main ];

  return (
    <>
      <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Box>
            <Typography variant="h6" fontWeight="bold">
              {title}
            </Typography>
            {subtitle && (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
      <Tooltip title="Refresh">
            <IconButton size="small" onClick={onRefresh}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
        
        <ResponsiveContainer width="100%" height={height}>
          <FunnelChart>
            <RechartsTooltip 
              contentStyle={{
                backgroundColor: theme.palette.background.paper,
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 4}}
            />
            <Funnel
              dataKey="value"
              data={data}
              isAnimationActive
              labelLine={false}
              label={(entry) => `${entry.name}: ${entry.value} (${entry.percent}%)`}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Funnel>
          </FunnelChart>
        </ResponsiveContainer>
        
        {/* Legend */}
        <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {data.map((item, index) => (
            <Chip
              key={item.name}
              label={`${item.name}: ${item.conversion}%`}
              size="small"
              sx={ {
                backgroundColor: COLORS[index % COLORS.length],
                color: '#fff' }}
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  </>
  )
};
// Treemap Chart Component
export const TreemapVisualization: React.FC<ChartProps> = ({ title, subtitle, data, height = 400 }) => {
  const theme = useTheme();
  const COLORS = [ '#8889DD',
    '#9597E4',
    '#8 DC77 B',
    '#A5 D297',
    '#E2 CF45',
    '#F8 C12 D',
    '#F2 A93 B',
    '#E68B3 C' ];

  const CustomizedContent = (props: unknown) => {
    const { root, depth, x, y, width, height, index, colors, name, value } = props;

    return (
    <>
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          style={ {
            fill: depth < 2 ? colors[Math.floor((index / root.children.length) * colors.length)] : 'none',
            stroke: '#fff',
            strokeWidth: 2 / (depth + 1 e-10),
            strokeOpacity: 1 / (depth + 1 e-10) }}
        />
        {depth === 1 && width > 50 && height > 30 && (
          <text
            x={x + width / 2}
            y={y + height / 2}
            textAnchor="middle"
            fill="#fff"
            fontSize={12}
          >
            {name}
          </text>
        )}
        {depth === 1 && width > 50 && height > 40 && (
          <text
            x={x + width / 2}
            y={y + height / 2 + 15}
            textAnchor="middle"
            fill="#fff"
            fontSize={10}
          >
            {value}
          </text>
        )}
      </g>
    </>
  )
};
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
            {subtitle}
          </Typography>
        )}
        <ResponsiveContainer width="100%" height={height}>
          <Treemap
            data={data}
            dataKey="size"
            ratio={4 / 3}
            stroke="#fff"
            fill="#8884d8"
            content={<CustomizedContent colors={COLORS} />}
          />
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
};
// Radar Chart Component
export const RadarVisualization: React.FC<ChartProps & {
  metrics: string[]}> = ({ title, subtitle, data, metrics, height = 400 }) => {
  const theme = useTheme();

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
            {subtitle}
          </Typography>
        )}
        <ResponsiveContainer width="100%" height={height}>
          <RadarChart data={data}>
            <PolarGrid 
              gridType="polygon"
              stroke={theme.palette.divider}
            />
            <PolarAngleAxis 
              dataKey="metric"
              tick={{ fontSize: 12 }}
              stroke={theme.palette.text.secondary}
            />
            <PolarRadiusAxis 
              angle={90}
              domain={[ 0, 100 ]
              tick={{ fontSize: 10 }}
              stroke={theme.palette.text.secondary}
            />
            {metrics.map((metric, index) => (
              <Radar
                key={metric}
                name={metric}
                dataKey={metric}
                stroke={index === 0 ? theme.palette.primary.main : theme.palette.secondary.main}
                fill={index === 0 ? theme.palette.primary.main : theme.palette.secondary.main}
                fillOpacity={0.3}
              />
            ))}
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )};
// Scatter Plot with Bubble sizes
export const BubbleChart: React.FC<ChartProps & {
  xKey: string;,

  yKey: string;
  zKey: string}> = ({ title, subtitle, data, xKey, yKey, zKey, height = 400 }) => {
  const theme = useTheme();

  const renderTooltip = (props: unknown) => {
    const { active, payload } = props;
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
    <Paper sx={{ p: 1, backgroundColor: 'background.paper' }}>
          <Typography variant="caption" display="block">
            {xKey}: {data[ xKey ]
          </Typography>
      <Typography variant="caption" display="block">
            {yKey}: {data[ yKey ]
          </Typography>
          <Typography variant="caption" display="block">
            {zKey}: {data[ zKey ]
          </Typography>
        </Paper>
      )}
    return null};
  return (
    <>
      <Card>
      <CardContent>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
            {subtitle}
          </Typography>
        )}
        <ResponsiveContainer width="100%" height={height}>
          <ScatterChart
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis 
              type="number" 
              dataKey={xKey} 
              name={xKey}
              tick={{ fontSize: 12 }}
              stroke={theme.palette.text.secondary}
            />
            <YAxis 
              type="number" 
              dataKey={yKey} 
              name={yKey}
              tick={{ fontSize: 12 }}
              stroke={theme.palette.text.secondary}
            />
            <ZAxis 
              type="number" 
              dataKey={zKey} 
              range={[ 60, 400 ]
              name={zKey}
            />
            <RechartsTooltip 
              cursor={{ strokeDasharray: '3 3' }}
              content={renderTooltip}
            />
            <Scatter 
              name="Data" 
              data={data} 
              fill={theme.palette.primary.main}
              fillOpacity={0.6}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  </>
  )};
// Cohort Retention Chart
export const CohortChart: React.FC<{
  title: string;,

  data: { cohort: string; week: number; retention: number }[];
  height?: number}> = ({ title, data, height = 400 }) => {
  const theme = useTheme();
  
  // Transform data for heatmap
  const cohorts = [...new Set(data.map(d => d.cohort))];
  const weeks = [...new Set(data.map(d => d.week))].sort((a, b) => a - b);
  
  const heatmapData = weeks.map(week =>
    cohorts.map(cohort => {
      const item = data.find(d => d.cohort === cohort && d.week === week);
      return item ? item.retention : 0})
  );

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          {title}
        </Typography>
      <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          User retention by cohort over time
        </Typography>
        
        <Box sx={{ height, overflow: 'auto' }}>
          <HeatMapGrid
            data={heatmapData}
            xLabels={cohorts}
            yLabels={weeks.map(w => `Week ${w}`)
            cellRender={(x, y, value) => `${value}%`}
            xLabelsStyle={() => ({
              fontSize: '0.75rem',
              color: theme.palette.text.secondary
            })}
            yLabelsStyle={() => ({
              fontSize: '0.75rem',
              color: theme.palette.text.secondary
            })}
            cellStyle={ (x, y, ratio) => ({
              background: ratio > 0.7 
                ? theme.palette.success.main
                : ratio > 0.4
                ? theme.palette.warning.main
                : theme.palette.error.main,
              color: '#fff',
              fontSize: '0.7rem',
              border: '1px solid rgba(255,255,255,0.2)' })}
            cellHeight="35px"
            square
          />
        </Box>
        
        {/* Legend */}
        <Box sx={{ mt: 2, display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Chip 
            label="High (>70%)" 
            size="small" 
            sx={{ backgroundColor: theme.palette.success.main, color: '#fff' }}
          />
          <Chip 
            label="Medium (40-70%)" 
            size="small" 
            sx={{ backgroundColor: theme.palette.warning.main, color: '#fff' }}
          />
          <Chip 
            label="Low (<40%)" 
            size="small" 
            sx={{ backgroundColor: theme.palette.error.main, color: '#fff' }}
          />
        </Box>
      </CardContent>
    </Card>
  )
}
