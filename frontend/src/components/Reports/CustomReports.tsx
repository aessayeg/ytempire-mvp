/**
 * Custom Reporting Component
 * Allows users to create, customize, and export detailed reports
 */

import React, { useState, useCallback, useMemo } from 'react';
import { 
  Box,
  Paper,
  Typography,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  LinearProgress,
  Card,
  CardContent,
  Stack,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormGroup,
  Checkbox,
  Tabs,
  Tab,
  Chip,
  IconButton
} from '@mui/material';
import { 
  Download as DownloadIcon,
  Schedule as ScheduleIcon,
  Share as ShareIcon,
  Save as SaveIcon,
  Add as AddIcon,
  Description as CsvIcon,
  Email as EmailIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  PictureAsPdf as PdfIcon,
  TableChart as ExcelIcon
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers';
import { 
  LineChart, 
  Line, 
  PieChart, 
  Pie, 
  Cell, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as ChartTooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { format, subDays, startOfWeek, endOfWeek, startOfMonth, endOfMonth } from 'date-fns';

// Types
interface ReportMetric {
  id: string;
  name: string;
  category: string;
  selected: boolean;
  aggregation: 'sum' | 'avg' | 'min' | 'max' | 'count';
}

interface ReportFilter {
  field: string;
  operator: 'equals' | 'contains' | 'greater' | 'less' | 'between';
  value: unknown;
}

interface ReportSchedule {
  frequency: 'daily' | 'weekly' | 'monthly';
  time: string;
  recipients: string[];
  format: 'pdf' | 'excel' | 'csv';
}

interface SavedReport {
  id: string;
  name: string;
  description: string;
  metrics: string[];
  filters: ReportFilter[];
  dateRange: [Date, Date];
  createdAt: Date;
  lastRun: Date;
  schedule?: ReportSchedule;
}

interface ReportData {
  metrics: Record<string, number>;
  charts: {
    type: 'line' | 'bar' | 'pie';
    data: any[];
  }[];
  tables: {
    headers: string[];
    rows: unknown[][];
  }[];
}

const AVAILABLE_METRICS: ReportMetric[] = [
  // Video Metrics
  { id: 'total_videos', name: 'Total Videos', category: 'Videos', selected: true, aggregation: 'sum' },
  { id: 'published_videos', name: 'Published Videos', category: 'Videos', selected: true, aggregation: 'sum' },
  { id: 'avg_video_length', name: 'Average Video Length', category: 'Videos', selected: false, aggregation: 'avg' },
  { id: 'avg_generation_time', name: 'Avg Generation Time', category: 'Videos', selected: false, aggregation: 'avg' },
  
  // Performance Metrics
  { id: 'total_views', name: 'Total Views', category: 'Performance', selected: true, aggregation: 'sum' },
  { id: 'total_watch_time', name: 'Total Watch Time', category: 'Performance', selected: true, aggregation: 'sum' },
  { id: 'avg_view_duration', name: 'Avg View Duration', category: 'Performance', selected: false, aggregation: 'avg' },
  { id: 'click_through_rate', name: 'Click-Through Rate', category: 'Performance', selected: false, aggregation: 'avg' },
  { id: 'engagement_rate', name: 'Engagement Rate', category: 'Performance', selected: false, aggregation: 'avg' },
  
  // Revenue Metrics
  { id: 'total_revenue', name: 'Total Revenue', category: 'Revenue', selected: true, aggregation: 'sum' },
  { id: 'ad_revenue', name: 'Ad Revenue', category: 'Revenue', selected: false, aggregation: 'sum' },
  { id: 'rpm', name: 'RPM', category: 'Revenue', selected: false, aggregation: 'avg' },
  { id: 'cpm', name: 'CPM', category: 'Revenue', selected: false, aggregation: 'avg' },
  
  // Cost Metrics
  { id: 'total_cost', name: 'Total Cost', category: 'Costs', selected: true, aggregation: 'sum' },
  { id: 'cost_per_video', name: 'Cost per Video', category: 'Costs', selected: false, aggregation: 'avg' },
  { id: 'ai_costs', name: 'AI Service Costs', category: 'Costs', selected: false, aggregation: 'sum' },
  { id: 'profit_margin', name: 'Profit Margin', category: 'Costs', selected: false, aggregation: 'avg' },
  
  // Channel Metrics
  { id: 'subscriber_count', name: 'Subscribers', category: 'Channel', selected: false, aggregation: 'sum' },
  { id: 'subscriber_growth', name: 'Subscriber Growth', category: 'Channel', selected: false, aggregation: 'sum' },
  { id: 'channel_health_score', name: 'Channel Health Score', category: 'Channel', selected: false, aggregation: 'avg' }
];

const PRESET_DATE_RANGES = [
  { label: 'Last 7 Days', getValue: () => [subDays(new Date(), 7), new Date()] },
  { label: 'Last 30 Days', getValue: () => [subDays(new Date(), 30), new Date()] },
  { label: 'This Week', getValue: () => [startOfWeek(new Date()), endOfWeek(new Date())] },
  { label: 'This Month', getValue: () => [startOfMonth(new Date()), endOfMonth(new Date())] },
  { label: 'Last Month', getValue: () => {
    const lastMonth = subDays(startOfMonth(new Date()), 1);
    return [startOfMonth(lastMonth), endOfMonth(lastMonth)];
  }}
];

export const CustomReports: React.FC = () => {
  // State
  const [selectedMetrics, setSelectedMetrics] = useState<ReportMetric[]>(
    AVAILABLE_METRICS.filter(m => m.selected)
  );
  const [dateRange, setDateRange] = useState<[Date, Date]>([
    subDays(new Date(), 30),
    new Date()
  ]);
  const [filters, setFilters] = useState<ReportFilter[]>([]);
  const [savedReports, setSavedReports] = useState<SavedReport[]>([]);
  const [currentTab, setCurrentTab] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [scheduleDialog, setScheduleDialog] = useState(false);
  const [saveDialog, setSaveDialog] = useState(false);
  const [reportName, setReportName] = useState('');
  const [reportDescription, setReportDescription] = useState('');
  const [schedule, setSchedule] = useState<ReportSchedule>({
    frequency: 'weekly',
    time: '09:00',
    recipients: [],
    format: 'pdf'
  });
  const [startDate, setStartDate] = useState<Date | null>(dateRange[0]);
  const [endDate, setEndDate] = useState<Date | null>(dateRange[1]);

  // Generate mock report data
  const generateReportData = useCallback(async () => {
    setIsGenerating(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Generate mock data based on selected metrics
    const metrics: Record<string, number> = {};
    selectedMetrics.forEach(metric => {
      const baseValue = Math.random() * 10000;
      metrics[metric.id] = metric.aggregation === 'avg' 
        ? parseFloat((Math.random() * 100).toFixed(2))
        : Math.floor(baseValue);
    });
    
    // Generate chart data
    const chartData = Array.from({ length: 30 }, (_, i) => ({
      date: format(subDays(new Date(), 30 - i), 'MMM dd'),
      views: Math.floor(Math.random() * 5000),
      revenue: parseFloat((Math.random() * 500).toFixed(2)),
      videos: Math.floor(Math.random() * 10)
    }));
    
    // Generate table data
    const tableData = Array.from({ length: 10 }, (_, i) => [
      `Video ${i + 1}`,
      Math.floor(Math.random() * 100000),
      parseFloat((Math.random() * 100).toFixed(2)),
      format(subDays(new Date(), i), 'yyyy-MM-dd')
    ]);
    
    setReportData({
      metrics,
      charts: [
        { type: 'line', data: chartData },
        { 
          type: 'pie',
          data: [
            { name: 'Ad Revenue', value: 4500 },
            { name: 'Sponsorships', value: 2200 },
            { name: 'Affiliates', value: 1800 },
            { name: 'Other', value: 500 }
          ]
        }
      ],
      tables: [
        {
          headers: ['Video Title', 'Views', 'Revenue ($)', 'Published Date'],
          rows: tableData
        }
      ]
    });
    
    setIsGenerating(false);
  }, [selectedMetrics]);

  // Export functions
  const exportToPDF = useCallback(() => {
    // In production, use jsPDF or similar
    console.log('Exporting to PDF...');
    alert('PDF export initiated. Report will be downloaded shortly.');
  }, [reportData]);

  const exportToExcel = useCallback(() => {
    // In production, use xlsx or similar
    console.log('Exporting to Excel...');
    alert('Excel export initiated. Report will be downloaded shortly.');
  }, [reportData]);

  const exportToCSV = useCallback(() => {
    if (!reportData) return;
    
    // Convert metrics to CSV
    let csv = 'Metric,Value\n';
    Object.entries(reportData.metrics).forEach(([key, value]) => {
      const metric = AVAILABLE_METRICS.find(m => m.id === key);
      csv += `"${metric?.name || key}",${value}\n`;
    });
    
    // Add table data
    if (reportData.tables.length > 0) {
      csv += '\n\nDetailed Data\n';
      csv += reportData.tables[0].headers.join(',') + '\n';
      reportData.tables[0].rows.forEach(row => {
        csv += row.join(',') + '\n';
      });
    }

    // Download CSV
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report_${format(new Date(), 'yyyy-MM-dd')}.csv`;
    a.click();
  }, [reportData]);

  // Save report
  const saveReport = useCallback(() => {
    const newReport: SavedReport = {
      id: Date.now().toString(),
      name: reportName,
      description: reportDescription,
      metrics: selectedMetrics.map(m => m.id),
      filters,
      dateRange,
      createdAt: new Date(),
      lastRun: new Date()
    };
    setSavedReports([...savedReports, newReport]);
    setSaveDialog(false);
    setReportName('');
    setReportDescription('');
  }, [reportName, reportDescription, selectedMetrics, filters, dateRange, savedReports]);

  // Load saved report
  const loadSavedReport = useCallback((report: SavedReport) => {
    const metrics = AVAILABLE_METRICS.map(m => ({
      ...m,
      selected: report.metrics.includes(m.id)
    }));
    setSelectedMetrics(metrics.filter(m => m.selected));
    setFilters(report.filters);
    setDateRange(report.dateRange);
    generateReportData();
  }, [generateReportData]);

  // Metric selection
  const toggleMetric = useCallback((metricId: string) => {
    setSelectedMetrics(prev => {
      const metric = AVAILABLE_METRICS.find(m => m.id === metricId);
      if (!metric) return prev;
      
      const exists = prev.find(m => m.id === metricId);
      if (exists) {
        return prev.filter(m => m.id !== metricId);
      } else {
        return [...prev, metric];
      }
    });
  }, []);

  // Update date range when individual dates change
  React.useEffect(() => {
    if (startDate && endDate) {
      setDateRange([startDate, endDate]);
    }
  }, [startDate, endDate]);

  // Render metric categories
  const renderMetricCategories = () => {
    const categories = [...new Set(AVAILABLE_METRICS.map(m => m.category))];
    
    return categories.map(category => (
      <Box key={category} sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
          {category}
        </Typography>
        <FormGroup row>
          {AVAILABLE_METRICS
            .filter(m => m.category === category)
            .map(metric => (
              <FormControlLabel
                key={metric.id}
                control={
                  <Checkbox
                    checked={selectedMetrics.some(m => m.id === metric.id)}
                    onChange={() => toggleMetric(metric.id)}
                    size="small"
                  />
                }
                label={
                  <Typography variant="body2">{metric.name}</Typography>
                }
                sx={{ mr: 2 }}
              />
            ))}
        </FormGroup>
      </Box>
    ));
  };

  // Render charts
  const renderCharts = () => {
    if (!reportData) return null;
    
    return (
      <Grid container spacing={3}>
        {/* Line Chart */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Trend Analysis
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={reportData.charts[0].data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <ChartTooltip />
                <Legend />
                <Line type="monotone" dataKey="views" stroke="#8884d8" />
                <Line type="monotone" dataKey="revenue" stroke="#82ca9d" />
                <Line type="monotone" dataKey="videos" stroke="#ffc658" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Pie Chart */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Revenue Breakdown
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={reportData.charts[1].data}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label
                >
                  {reportData.charts[1].data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={['#0088FE', '#00C49F', '#FFBB28', '#FF8042'][index % 4]} />
                  ))}
                </Pie>
                <ChartTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Custom Reports
      </Typography>
      
      <Tabs value={currentTab} onChange={(_, v) => setCurrentTab(v)} sx={{ mb: 3 }}>
        <Tab label="Create Report" icon={<AddIcon />} />
        <Tab label="Saved Reports" icon={<SaveIcon />} />
        <Tab label="Scheduled Reports" icon={<ScheduleIcon />} />
      </Tabs>
      
      {currentTab === 0 && (
        <>
          {/* Report Configuration */}
          <Grid container spacing={3}>
            {/* Date Range Selection */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Date Range
                </Typography>
                <Stack direction="row" spacing={2} alignItems="center">
                  {PRESET_DATE_RANGES.map(preset => (
                    <Button
                      key={preset.label}
                      variant="outlined"
                      size="small"
                      onClick={() => {
                        const range = preset.getValue() as [Date, Date];
                        setDateRange(range);
                        setStartDate(range[0]);
                        setEndDate(range[1]);
                      }}
                    >
                      {preset.label}
                    </Button>
                  ))}
                  <DatePicker
                    label="Start Date"
                    value={startDate}
                    onChange={(newValue) => setStartDate(newValue)}
                    renderInput={(params: any) => <TextField {...params} size="small" />}
                  />
                  <Box sx={{ mx: 2 }}> to </Box>
                  <DatePicker
                    label="End Date"
                    value={endDate}
                    onChange={(newValue) => setEndDate(newValue)}
                    renderInput={(params: any) => <TextField {...params} size="small" />}
                  />
                </Stack>
              </Paper>
            </Grid>
            
            {/* Metric Selection */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Select Metrics
                </Typography>
                {renderMetricCategories()}
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={`${selectedMetrics.length} metrics selected`}
                    color="primary"
                    sx={{ mr: 1 }}
                  />
                </Box>
              </Paper>
            </Grid>
            
            {/* Generate Button */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<RefreshIcon />}
                  onClick={generateReportData}
                  disabled={isGenerating || selectedMetrics.length === 0}
                >
                  Generate Report
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<SaveIcon />}
                  onClick={() => setSaveDialog(true)}
                  disabled={!reportData}
                >
                  Save Report
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ScheduleIcon />}
                  onClick={() => setScheduleDialog(true)}
                  disabled={!reportData}
                >
                  Schedule Report
                </Button>
              </Box>
            </Grid>
          </Grid>
          
          {/* Loading State */}
          {isGenerating && (
            <Box sx={{ mt: 3 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Generating report...
              </Typography>
            </Box>
          )}

          {/* Report Results */}
          {reportData && !isGenerating && (
            <Box sx={{ mt: 3 }}>
              {/* Export Buttons */}
              <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PdfIcon />}
                  onClick={exportToPDF}
                  color="error"
                >
                  Export PDF
                </Button>
                <Button
                  variant="contained"
                  startIcon={<ExcelIcon />}
                  onClick={exportToExcel}
                  color="success"
                >
                  Export Excel
                </Button>
                <Button
                  variant="contained"
                  startIcon={<CsvIcon />}
                  onClick={exportToCSV}
                >
                  Export CSV
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<EmailIcon />}
                >
                  Email Report
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ShareIcon />}
                >
                  Share Report
                </Button>
              </Box>
              
              {/* Metrics Summary */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                {Object.entries(reportData.metrics).map(([key, value]) => {
                  const metric = AVAILABLE_METRICS.find(m => m.id === key);
                  return (
                    <Grid item xs={12} sm={6} md={3} key={key}>
                      <Card>
                        <CardContent>
                          <Typography color="textSecondary" gutterBottom variant="body2">
                            {metric?.name || key}
                          </Typography>
                          <Typography variant="h5">
                            {typeof value === 'number' && value > 1000 
                              ? value.toLocaleString()
                              : value}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
              
              {/* Charts */}
              {renderCharts()}

              {/* Data Table */}
              {reportData.tables.length > 0 && (
                <Paper sx={{ mt: 3, p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Detailed Data
                  </Typography>
                  <Box sx={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {reportData.tables[0].headers.map((header, i) => (
                            <th key={i} style={{ 
                              padding: '8px', 
                              borderBottom: '2px solid #ddd',
                              textAlign: 'left'
                            }}>
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {reportData.tables[0].rows.map((row, i) => (
                          <tr key={i}>
                            {row.map((cell, j) => (
                              <td key={j} style={{ 
                                padding: '8px', 
                                borderBottom: '1px solid #eee'
                              }}>
                                {String(cell)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Box>
                </Paper>
              )}
            </Box>
          )}
        </>
      )}

      {currentTab === 1 && (
        <Grid container spacing={2}>
          {savedReports.map(report => (
            <Grid item xs={12} md={6} lg={4} key={report.id}>
              <Card>
                <CardContent>
                  <Typography variant="h6">{report.name}</Typography>
                  <Typography variant="body2" color="textSecondary" paragraph>
                    {report.description}
                  </Typography>
                  <Typography variant="caption" display="block">
                    Created: {format(report.createdAt, 'MMM dd, yyyy')}
                  </Typography>
                  <Typography variant="caption" display="block">
                    Last Run: {format(report.lastRun, 'MMM dd, yyyy HH:mm')}
                  </Typography>
                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Button 
                      size="small" 
                      variant="contained"
                      onClick={() => loadSavedReport(report)}
                    >
                      Load
                    </Button>
                    <Button size="small" variant="outlined">
                      Edit
                    </Button>
                    <IconButton size="small" color="error">
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Save Report Dialog */}
      <Dialog open={saveDialog} onClose={() => setSaveDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Save Report Configuration</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Report Name"
            value={reportName}
            onChange={(e) => setReportName(e.target.value)}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Description"
            value={reportDescription}
            onChange={(e) => setReportDescription(e.target.value)}
            multiline
            rows={3}
            margin="normal"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialog(false)}>Cancel</Button>
          <Button onClick={saveReport} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
      
      {/* Schedule Report Dialog */}
      <Dialog open={scheduleDialog} onClose={() => setScheduleDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Schedule Report</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Frequency</InputLabel>
            <Select
              value={schedule.frequency}
              onChange={(e) => setSchedule({ ...schedule, frequency: e.target.value as any })}
            >
              <MenuItem value="daily">Daily</MenuItem>
              <MenuItem value="weekly">Weekly</MenuItem>
              <MenuItem value="monthly">Monthly</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            type="time"
            label="Time"
            value={schedule.time}
            onChange={(e) => setSchedule({ ...schedule, time: e.target.value })}
            margin="normal"
            InputLabelProps={{ shrink: true }}
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Format</InputLabel>
            <Select
              value={schedule.format}
              onChange={(e) => setSchedule({ ...schedule, format: e.target.value as any })}
            >
              <MenuItem value="pdf">PDF</MenuItem>
              <MenuItem value="excel">Excel</MenuItem>
              <MenuItem value="csv">CSV</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setScheduleDialog(false)}>Cancel</Button>
          <Button variant="contained">Schedule</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};