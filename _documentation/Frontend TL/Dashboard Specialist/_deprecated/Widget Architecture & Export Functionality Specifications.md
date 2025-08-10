# Widget Architecture & Export Functionality Specifications

**Version**: 1.0  
**Date**: January 2025  
**For**: Dashboard Specialist  
**From**: Frontend Team Lead  
**Focus**: Modular widgets and data export capabilities

---

## 1. Widget Architecture Design

### 1.1 Core Widget System

```typescript
// Base widget interface for all dashboard components
interface BaseWidget {
  id: string;
  type: WidgetType;
  title: string;
  description?: string;
  
  // Layout properties
  gridPosition: {
    x: number;
    y: number;
    w: number;  // Width in grid units (1-12)
    h: number;  // Height in grid units
  };
  
  // Data configuration
  dataSource: {
    endpoint: string;
    refreshInterval: number;  // milliseconds
    params?: Record<string, any>;
  };
  
  // Display configuration
  displayConfig: {
    showHeader: boolean;
    showActions: boolean;
    showExport: boolean;
    showRefresh: boolean;
  };
  
  // State
  state: {
    isLoading: boolean;
    hasError: boolean;
    lastUpdated: Date;
    data: any;
  };
}

// Widget types available in MVP
enum WidgetType {
  METRIC_CARD = 'metric_card',
  LINE_CHART = 'line_chart',
  BAR_CHART = 'bar_chart',
  PIE_CHART = 'pie_chart',
  AREA_CHART = 'area_chart',
  DATA_TABLE = 'data_table',
  VIDEO_QUEUE = 'video_queue',
  ALERT_BANNER = 'alert_banner'
}
```

### 1.2 Widget Factory Pattern

```typescript
// Widget factory for creating different widget types
class WidgetFactory {
  private static widgetRegistry = new Map<WidgetType, React.ComponentType<any>>();
  
  // Register widget components
  static register(type: WidgetType, component: React.ComponentType<any>) {
    this.widgetRegistry.set(type, component);
  }
  
  // Create widget instance
  static create(config: WidgetConfig): React.ReactElement {
    const Component = this.widgetRegistry.get(config.type);
    
    if (!Component) {
      throw new Error(`Widget type ${config.type} not registered`);
    }
    
    return <Component {...config} />;
  }
  
  // Initialize all widgets
  static initialize() {
    this.register(WidgetType.METRIC_CARD, MetricCardWidget);
    this.register(WidgetType.LINE_CHART, LineChartWidget);
    this.register(WidgetType.BAR_CHART, BarChartWidget);
    this.register(WidgetType.PIE_CHART, PieChartWidget);
    this.register(WidgetType.AREA_CHART, AreaChartWidget);
    this.register(WidgetType.DATA_TABLE, DataTableWidget);
    this.register(WidgetType.VIDEO_QUEUE, VideoQueueWidget);
    this.register(WidgetType.ALERT_BANNER, AlertBannerWidget);
  }
}
```

### 1.3 Widget Container Component

```tsx
interface WidgetContainerProps {
  widget: BaseWidget;
  onRemove?: (id: string) => void;
  onRefresh?: (id: string) => void;
  onExport?: (id: string, format: ExportFormat) => void;
  onSettings?: (id: string) => void;
}

const WidgetContainer: React.FC<WidgetContainerProps> = ({
  widget,
  onRemove,
  onRefresh,
  onExport,
  onSettings
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  
  // Auto-refresh logic
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch(widget.dataSource.endpoint, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getAuthToken()}`
          }
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const newData = await response.json();
        setData(newData);
        setLastUpdated(new Date());
        
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    
    // Initial fetch
    fetchData();
    
    // Set up polling
    const interval = setInterval(fetchData, widget.dataSource.refreshInterval);
    
    return () => clearInterval(interval);
  }, [widget.dataSource]);
  
  // Handle manual refresh
  const handleRefresh = () => {
    setLastUpdated(new Date());
    onRefresh?.(widget.id);
  };
  
  // Handle export menu
  const [exportAnchor, setExportAnchor] = useState<null | HTMLElement>(null);
  
  const handleExportClick = (event: React.MouseEvent<HTMLElement>) => {
    setExportAnchor(event.currentTarget);
  };
  
  const handleExportClose = () => {
    setExportAnchor(null);
  };
  
  const handleExportFormat = (format: ExportFormat) => {
    onExport?.(widget.id, format);
    handleExportClose();
  };
  
  return (
    <Card 
      sx={{ 
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative'
      }}
    >
      {widget.displayConfig.showHeader && (
        <CardHeader
          title={widget.title}
          subheader={`Updated ${formatRelativeTime(lastUpdated)}`}
          action={
            <Box>
              {widget.displayConfig.showRefresh && (
                <IconButton 
                  size="small" 
                  onClick={handleRefresh}
                  disabled={isLoading}
                >
                  <RefreshIcon />
                </IconButton>
              )}
              
              {widget.displayConfig.showExport && (
                <>
                  <IconButton size="small" onClick={handleExportClick}>
                    <DownloadIcon />
                  </IconButton>
                  <Menu
                    anchorEl={exportAnchor}
                    open={Boolean(exportAnchor)}
                    onClose={handleExportClose}
                  >
                    <MenuItem onClick={() => handleExportFormat('csv')}>
                      Export as CSV
                    </MenuItem>
                    <MenuItem onClick={() => handleExportFormat('json')}>
                      Export as JSON
                    </MenuItem>
                    <MenuItem onClick={() => handleExportFormat('png')}>
                      Export as Image
                    </MenuItem>
                  </Menu>
                </>
              )}
              
              {onSettings && (
                <IconButton size="small" onClick={() => onSettings(widget.id)}>
                  <SettingsIcon />
                </IconButton>
              )}
              
              {onRemove && (
                <IconButton size="small" onClick={() => onRemove(widget.id)}>
                  <CloseIcon />
                </IconButton>
              )}
            </Box>
          }
        />
      )}
      
      <CardContent sx={{ flex: 1, position: 'relative', overflow: 'auto' }}>
        {isLoading && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(255, 255, 255, 0.8)',
              zIndex: 1
            }}
          >
            <CircularProgress />
          </Box>
        )}
        
        {error ? (
          <Alert severity="error">
            <AlertTitle>Error</AlertTitle>
            {error}
          </Alert>
        ) : (
          <WidgetContent widget={widget} data={data} />
        )}
      </CardContent>
    </Card>
  );
};
```

### 1.4 Widget Types Implementation

#### Metric Card Widget

```tsx
const MetricCardWidget: React.FC<WidgetProps> = ({ config, data }) => {
  const { value, change, trend } = data || {};
  
  return (
    <Box sx={{ textAlign: 'center', py: 2 }}>
      <Typography variant="h3" color="primary">
        {formatValue(value, config.format)}
      </Typography>
      
      {change !== undefined && (
        <Box display="flex" alignItems="center" justifyContent="center" mt={1}>
          {change > 0 ? <TrendingUpIcon color="success" /> : <TrendingDownIcon color="error" />}
          <Typography 
            variant="body1" 
            color={change > 0 ? 'success.main' : 'error.main'}
            ml={1}
          >
            {change > 0 ? '+' : ''}{change.toFixed(1)}%
          </Typography>
        </Box>
      )}
      
      {trend && (
        <Box mt={2}>
          <SparklineChart data={trend} height={50} />
        </Box>
      )}
    </Box>
  );
};
```

#### Data Table Widget

```tsx
const DataTableWidget: React.FC<WidgetProps> = ({ config, data }) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [orderBy, setOrderBy] = useState(config.defaultSort?.field || '');
  const [order, setOrder] = useState<'asc' | 'desc'>(config.defaultSort?.order || 'asc');
  
  const handleSort = (field: string) => {
    const isAsc = orderBy === field && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(field);
  };
  
  const sortedData = useMemo(() => {
    if (!data || !orderBy) return data;
    
    return [...data].sort((a, b) => {
      const aVal = a[orderBy];
      const bVal = b[orderBy];
      
      if (order === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });
  }, [data, orderBy, order]);
  
  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            {config.columns.map((column) => (
              <TableCell
                key={column.field}
                sortDirection={orderBy === column.field ? order : false}
              >
                {column.sortable ? (
                  <TableSortLabel
                    active={orderBy === column.field}
                    direction={orderBy === column.field ? order : 'asc'}
                    onClick={() => handleSort(column.field)}
                  >
                    {column.label}
                  </TableSortLabel>
                ) : (
                  column.label
                )}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedData
            ?.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
            .map((row, index) => (
              <TableRow key={row.id || index}>
                {config.columns.map((column) => (
                  <TableCell key={column.field}>
                    {column.formatter 
                      ? column.formatter(row[column.field], row)
                      : row[column.field]
                    }
                  </TableCell>
                ))}
              </TableRow>
            ))}
        </TableBody>
      </Table>
      <TablePagination
        component="div"
        count={data?.length || 0}
        page={page}
        onPageChange={(e, newPage) => setPage(newPage)}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={(e) => {
          setRowsPerPage(parseInt(e.target.value, 10));
          setPage(0);
        }}
      />
    </TableContainer>
  );
};
```

#### Video Queue Widget

```tsx
const VideoQueueWidget: React.FC<WidgetProps> = ({ config, data }) => {
  const { queue = [], processing = [] } = data || {};
  
  return (
    <Box>
      {/* Currently Processing */}
      {processing.length > 0 && (
        <Box mb={2}>
          <Typography variant="subtitle2" gutterBottom>
            Currently Processing ({processing.length})
          </Typography>
          {processing.map((video) => (
            <Box key={video.id} mb={1}>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="body2">{video.title}</Typography>
                <Chip 
                  label={video.stage} 
                  size="small" 
                  color="primary"
                />
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={video.progress} 
                sx={{ mt: 0.5 }}
              />
            </Box>
          ))}
        </Box>
      )}
      
      {/* Queued Videos */}
      {queue.length > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            In Queue ({queue.length})
          </Typography>
          <List dense>
            {queue.slice(0, 5).map((video, index) => (
              <ListItem key={video.id}>
                <ListItemText
                  primary={video.title}
                  secondary={`Position: ${index + 1} • Channel: ${video.channelName}`}
                />
              </ListItem>
            ))}
            {queue.length > 5 && (
              <ListItem>
                <ListItemText 
                  secondary={`+${queue.length - 5} more in queue`}
                />
              </ListItem>
            )}
          </List>
        </Box>
      )}
      
      {processing.length === 0 && queue.length === 0 && (
        <Typography variant="body2" color="textSecondary">
          No videos in queue
        </Typography>
      )}
    </Box>
  );
};
```

### 1.5 Widget State Management

```typescript
// Zustand store for widget management
interface WidgetStore {
  widgets: Map<string, BaseWidget>;
  layouts: Map<string, WidgetLayout>;
  activeLayout: string;
  
  // Actions
  addWidget: (widget: BaseWidget) => void;
  removeWidget: (widgetId: string) => void;
  updateWidget: (widgetId: string, updates: Partial<BaseWidget>) => void;
  moveWidget: (widgetId: string, newPosition: GridPosition) => void;
  
  // Layout management
  saveLayout: (name: string) => void;
  loadLayout: (name: string) => void;
  resetLayout: () => void;
  
  // Data management
  refreshWidget: (widgetId: string) => void;
  refreshAllWidgets: () => void;
}

export const useWidgetStore = create<WidgetStore>((set, get) => ({
  widgets: new Map(),
  layouts: new Map(),
  activeLayout: 'default',
  
  addWidget: (widget) => {
    set((state) => {
      const newWidgets = new Map(state.widgets);
      newWidgets.set(widget.id, widget);
      return { widgets: newWidgets };
    });
  },
  
  removeWidget: (widgetId) => {
    set((state) => {
      const newWidgets = new Map(state.widgets);
      newWidgets.delete(widgetId);
      return { widgets: newWidgets };
    });
  },
  
  updateWidget: (widgetId, updates) => {
    set((state) => {
      const newWidgets = new Map(state.widgets);
      const widget = newWidgets.get(widgetId);
      if (widget) {
        newWidgets.set(widgetId, { ...widget, ...updates });
      }
      return { widgets: newWidgets };
    });
  },
  
  moveWidget: (widgetId, newPosition) => {
    const widget = get().widgets.get(widgetId);
    if (widget) {
      get().updateWidget(widgetId, { gridPosition: newPosition });
    }
  },
  
  saveLayout: (name) => {
    const currentWidgets = Array.from(get().widgets.values());
    const layout = {
      name,
      widgets: currentWidgets.map(w => ({
        id: w.id,
        type: w.type,
        gridPosition: w.gridPosition,
        dataSource: w.dataSource,
        displayConfig: w.displayConfig
      }))
    };
    
    set((state) => {
      const newLayouts = new Map(state.layouts);
      newLayouts.set(name, layout);
      return { layouts: newLayouts };
    });
    
    // Persist to localStorage
    localStorage.setItem(`dashboard_layout_${name}`, JSON.stringify(layout));
  },
  
  loadLayout: (name) => {
    const layout = get().layouts.get(name);
    if (layout) {
      // Clear current widgets
      set({ widgets: new Map() });
      
      // Load widgets from layout
      layout.widgets.forEach(widgetConfig => {
        get().addWidget(widgetConfig as BaseWidget);
      });
      
      set({ activeLayout: name });
    }
  },
  
  resetLayout: () => {
    get().loadLayout('default');
  },
  
  refreshWidget: (widgetId) => {
    const widget = get().widgets.get(widgetId);
    if (widget) {
      // Trigger refresh by updating lastUpdated
      get().updateWidget(widgetId, {
        state: {
          ...widget.state,
          lastUpdated: new Date()
        }
      });
    }
  },
  
  refreshAllWidgets: () => {
    get().widgets.forEach((widget) => {
      get().refreshWidget(widget.id);
    });
  }
}));
```

---

## 2. Export Functionality Specifications

### 2.1 Export Service Architecture

```typescript
enum ExportFormat {
  CSV = 'csv',
  JSON = 'json',
  EXCEL = 'excel',
  PNG = 'png',
  PDF = 'pdf'
}

class ExportService {
  // Export data based on format
  async export(
    data: any,
    format: ExportFormat,
    filename: string,
    options?: ExportOptions
  ): Promise<void> {
    switch (format) {
      case ExportFormat.CSV:
        await this.exportCSV(data, filename, options);
        break;
      case ExportFormat.JSON:
        await this.exportJSON(data, filename, options);
        break;
      case ExportFormat.EXCEL:
        await this.exportExcel(data, filename, options);
        break;
      case ExportFormat.PNG:
        await this.exportImage(data, filename, options);
        break;
      case ExportFormat.PDF:
        await this.exportPDF(data, filename, options);
        break;
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }
  
  // CSV Export
  private async exportCSV(data: any[], filename: string, options?: any): Promise<void> {
    const csv = this.convertToCSV(data, options);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    this.downloadFile(blob, `${filename}.csv`);
  }
  
  private convertToCSV(data: any[], options?: any): string {
    if (!data || data.length === 0) return '';
    
    // Get headers
    const headers = options?.headers || Object.keys(data[0]);
    
    // Create CSV content
    const csvContent = [
      headers.join(','),
      ...data.map(row => 
        headers.map(header => {
          const value = row[header];
          // Escape quotes and handle commas
          if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
            return `"${value.replace(/"/g, '""')}"`;
          }
          return value ?? '';
        }).join(',')
      )
    ].join('\n');
    
    return csvContent;
  }
  
  // JSON Export
  private async exportJSON(data: any, filename: string, options?: any): Promise<void> {
    const json = JSON.stringify(data, null, options?.indent || 2);
    const blob = new Blob([json], { type: 'application/json' });
    this.downloadFile(blob, `${filename}.json`);
  }
  
  // Excel Export (using SheetJS)
  private async exportExcel(data: any[], filename: string, options?: any): Promise<void> {
    // Note: Requires xlsx library
    const XLSX = await import('xlsx');
    
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, options?.sheetName || 'Data');
    
    // Apply styles if provided
    if (options?.styles) {
      this.applyExcelStyles(worksheet, options.styles);
    }
    
    // Generate buffer
    const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
    const blob = new Blob([excelBuffer], { 
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
    });
    
    this.downloadFile(blob, `${filename}.xlsx`);
  }
  
  // Image Export (for charts)
  private async exportImage(element: HTMLElement, filename: string, options?: any): Promise<void> {
    // Note: Requires html2canvas library
    const html2canvas = await import('html2canvas');
    
    const canvas = await html2canvas.default(element, {
      backgroundColor: options?.backgroundColor || '#ffffff',
      scale: options?.scale || 2
    });
    
    canvas.toBlob((blob) => {
      if (blob) {
        this.downloadFile(blob, `${filename}.png`);
      }
    }, 'image/png');
  }
  
  // PDF Export
  private async exportPDF(data: any, filename: string, options?: any): Promise<void> {
    // Note: Requires jspdf library
    const jsPDF = (await import('jspdf')).default;
    const doc = new jsPDF(options?.orientation || 'portrait');
    
    // Add title
    if (options?.title) {
      doc.setFontSize(16);
      doc.text(options.title, 20, 20);
    }
    
    // Add content based on type
    if (Array.isArray(data)) {
      // Table data
      this.addTableToPDF(doc, data, options);
    } else if (typeof data === 'string') {
      // Text content
      doc.setFontSize(12);
      doc.text(data, 20, 40);
    }
    
    doc.save(`${filename}.pdf`);
  }
  
  // Helper: Add table to PDF
  private addTableToPDF(doc: any, data: any[], options?: any): void {
    // Note: Requires jspdf-autotable
    const headers = options?.headers || Object.keys(data[0]);
    const rows = data.map(item => headers.map(header => item[header]));
    
    doc.autoTable({
      head: [headers],
      body: rows,
      startY: options?.startY || 40,
      theme: options?.theme || 'striped'
    });
  }
  
  // Helper: Download file
  private downloadFile(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}