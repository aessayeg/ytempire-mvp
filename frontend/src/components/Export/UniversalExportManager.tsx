/**
 * Universal Export Manager
 * Provides comprehensive export functionality for all data throughout the application
 */

import React, { useState, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormGroup,
  FormControlLabel,
  Checkbox,
  TextField,
  Box,
  Typography,
  Alert,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Chip,
  Paper,
  Grid,
  Divider,
  RadioGroup,
  Radio,
  Tooltip
} from '@mui/material';
import {
  Download as DownloadIcon,
  PictureAsPdf as PdfIcon,
  TableChart as ExcelIcon,
  Description as CsvIcon,
  Code as JsonIcon,
  Print as PrintIcon,
  Email as EmailIcon,
  CloudDownload as CloudIcon,
  DriveFileRenameOutline as RenameIcon,
  FilterList as FilterIcon,
  Schedule as ScheduleIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import { format } from 'date-fns';
import * as XLSX from 'xlsx';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import { saveAs } from 'file-saver';

// Export types
export type ExportFormat = 'csv' | 'excel' | 'pdf' | 'json' | 'xml' | 'print';

export interface ExportConfig {
  format: ExportFormat;
  filename: string;
  includeHeaders: boolean;
  includeMetadata: boolean;
  dateRange?: [Date, Date];
  filters?: Record<string, any>;
  columns?: string[];
  customTemplate?: string;
  compression?: boolean;
  encryption?: boolean;
  password?: string;
}

export interface ExportData {
  title: string;
  data: any[];
  columns?: { key: string; label: string; type?: string }[];
  metadata?: Record<string, any>;
  charts?: { type: string; data: any }[];
  summary?: Record<string, any>;
}

interface ExportManagerProps {
  open: boolean;
  onClose: () => void;
  data: ExportData;
  onExport?: (config: ExportConfig) => void;
  allowedFormats?: ExportFormat[];
}

export const UniversalExportManager: React.FC<ExportManagerProps> = ({
  open,
  onClose,
  data,
  onExport,
  allowedFormats = ['csv', 'excel', 'pdf', 'json']
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [exportConfig, setExportConfig] = useState<ExportConfig>({
    format: 'excel',
    filename: `${data.title.toLowerCase().replace(/\s+/g, '_')}_${format(new Date(), 'yyyy-MM-dd')}`,
    includeHeaders: true,
    includeMetadata: true,
    columns: data.columns?.map(c => c.key) || []
  });
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportSuccess, setExportSuccess] = useState(false);

  const steps = ['Select Format', 'Configure Options', 'Preview & Export'];

  // Format configurations
  const formatConfigs = {
    csv: {
      icon: <CsvIcon />,
      label: 'CSV',
      description: 'Comma-separated values, compatible with all spreadsheet applications',
      color: 'success'
    },
    excel: {
      icon: <ExcelIcon />,
      label: 'Excel',
      description: 'Microsoft Excel format with formatting and multiple sheets support',
      color: 'primary'
    },
    pdf: {
      icon: <PdfIcon />,
      label: 'PDF',
      description: 'Portable document format with charts and formatting',
      color: 'error'
    },
    json: {
      icon: <JsonIcon />,
      label: 'JSON',
      description: 'JavaScript Object Notation for developers and APIs',
      color: 'info'
    },
    xml: {
      icon: <JsonIcon />,
      label: 'XML',
      description: 'Extensible Markup Language for data interchange',
      color: 'warning'
    },
    print: {
      icon: <PrintIcon />,
      label: 'Print',
      description: 'Send directly to printer with print-friendly formatting',
      color: 'default'
    }
  };

  // Export to CSV
  const exportToCSV = useCallback(() => {
    const headers = exportConfig.includeHeaders && data.columns
      ? data.columns.filter(c => exportConfig.columns?.includes(c.key)).map(c => c.label).join(',')
      : '';
    
    const rows = data.data.map(row => {
      return exportConfig.columns?.map(col => {
        const value = row[col];
        // Escape commas and quotes
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      }).join(',');
    }).join('\n');
    
    let csvContent = headers ? `${headers}\n${rows}` : rows;
    
    if (exportConfig.includeMetadata && data.metadata) {
      const metadataRows = Object.entries(data.metadata)
        .map(([key, value]) => `"${key}","${value}"`)
        .join('\n');
      csvContent = `Metadata\n${metadataRows}\n\nData\n${csvContent}`;
    }
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, `${exportConfig.filename}.csv`);
  }, [data, exportConfig]);

  // Export to Excel
  const exportToExcel = useCallback(() => {
    const workbook = XLSX.utils.book_new();
    
    // Main data sheet
    const mainData = data.data.map(row => {
      const exportRow: any = {};
      exportConfig.columns?.forEach(col => {
        const column = data.columns?.find(c => c.key === col);
        exportRow[column?.label || col] = row[col];
      });
      return exportRow;
    });
    
    const mainSheet = XLSX.utils.json_to_sheet(mainData);
    XLSX.utils.book_append_sheet(workbook, mainSheet, 'Data');
    
    // Metadata sheet
    if (exportConfig.includeMetadata && data.metadata) {
      const metadataArray = Object.entries(data.metadata).map(([key, value]) => ({
        Property: key,
        Value: value
      }));
      const metadataSheet = XLSX.utils.json_to_sheet(metadataArray);
      XLSX.utils.book_append_sheet(workbook, metadataSheet, 'Metadata');
    }
    
    // Summary sheet
    if (data.summary) {
      const summaryArray = Object.entries(data.summary).map(([key, value]) => ({
        Metric: key,
        Value: value
      }));
      const summarySheet = XLSX.utils.json_to_sheet(summaryArray);
      XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary');
    }
    
    // Generate and save file
    const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
    const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    saveAs(blob, `${exportConfig.filename}.xlsx`);
  }, [data, exportConfig]);

  // Export to PDF
  const exportToPDF = useCallback(() => {
    const doc = new jsPDF();
    
    // Add title
    doc.setFontSize(16);
    doc.text(data.title, 14, 15);
    
    // Add metadata
    if (exportConfig.includeMetadata && data.metadata) {
      doc.setFontSize(10);
      let yPosition = 30;
      Object.entries(data.metadata).forEach(([key, value]) => {
        doc.text(`${key}: ${value}`, 14, yPosition);
        yPosition += 5;
      });
      yPosition += 5;
    }
    
    // Add table
    const tableColumns = data.columns
      ?.filter(c => exportConfig.columns?.includes(c.key))
      .map(c => c.label) || [];
    
    const tableRows = data.data.map(row => {
      return exportConfig.columns?.map(col => row[col] || '');
    });
    
    (doc as any).autoTable({
      head: [tableColumns],
      body: tableRows,
      startY: exportConfig.includeMetadata ? 60 : 30,
      theme: 'grid',
      styles: { fontSize: 8 }
    });
    
    // Add summary
    if (data.summary) {
      const finalY = (doc as any).lastAutoTable.finalY + 10;
      doc.setFontSize(12);
      doc.text('Summary', 14, finalY);
      doc.setFontSize(10);
      let summaryY = finalY + 5;
      Object.entries(data.summary).forEach(([key, value]) => {
        doc.text(`${key}: ${value}`, 14, summaryY);
        summaryY += 5;
      });
    }
    
    doc.save(`${exportConfig.filename}.pdf`);
  }, [data, exportConfig]);

  // Export to JSON
  const exportToJSON = useCallback(() => {
    const exportData: any = {
      title: data.title,
      exportDate: new Date().toISOString(),
      data: data.data.map(row => {
        const exportRow: any = {};
        exportConfig.columns?.forEach(col => {
          exportRow[col] = row[col];
        });
        return exportRow;
      })
    };
    
    if (exportConfig.includeMetadata && data.metadata) {
      exportData.metadata = data.metadata;
    }
    
    if (data.summary) {
      exportData.summary = data.summary;
    }
    
    const jsonString = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    saveAs(blob, `${exportConfig.filename}.json`);
  }, [data, exportConfig]);

  // Export to XML
  const exportToXML = useCallback(() => {
    const jsonToXml = (obj: any, rootName: string = 'root'): string => {
      let xml = `<?xml version="1.0" encoding="UTF-8"?>\n<${rootName}>`;
      
      const convertToXml = (data: any, indent: string = '  '): string => {
        let result = '';
        
        if (Array.isArray(data)) {
          data.forEach(item => {
            result += `\n${indent}<item>${convertToXml(item, indent + '  ')}\n${indent}</item>`;
          });
        } else if (typeof data === 'object' && data !== null) {
          Object.entries(data).forEach(([key, value]) => {
            const safeKey = key.replace(/[^a-zA-Z0-9_]/g, '_');
            if (typeof value === 'object') {
              result += `\n${indent}<${safeKey}>${convertToXml(value, indent + '  ')}\n${indent}</${safeKey}>`;
            } else {
              result += `\n${indent}<${safeKey}>${value}</${safeKey}>`;
            }
          });
        } else {
          result = String(data);
        }
        
        return result;
      };
      
      xml += convertToXml(obj);
      xml += `\n</${rootName}>`;
      
      return xml;
    };
    
    const exportData: any = {
      title: data.title,
      exportDate: new Date().toISOString(),
      data: data.data.map(row => {
        const exportRow: any = {};
        exportConfig.columns?.forEach(col => {
          exportRow[col] = row[col];
        });
        return exportRow;
      })
    };
    
    if (exportConfig.includeMetadata && data.metadata) {
      exportData.metadata = data.metadata;
    }
    
    const xmlString = jsonToXml(exportData, 'export');
    const blob = new Blob([xmlString], { type: 'application/xml' });
    saveAs(blob, `${exportConfig.filename}.xml`);
  }, [data, exportConfig]);

  // Handle export
  const handleExport = async () => {
    setIsExporting(true);
    setExportError(null);
    setExportProgress(0);
    
    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setExportProgress(prev => Math.min(prev + 20, 90));
      }, 200);
      
      switch (exportConfig.format) {
        case 'csv':
          exportToCSV();
          break;
        case 'excel':
          exportToExcel();
          break;
        case 'pdf':
          exportToPDF();
          break;
        case 'json':
          exportToJSON();
          break;
        case 'xml':
          exportToXML();
          break;
        case 'print':
          window.print();
          break;
      }
      
      clearInterval(progressInterval);
      setExportProgress(100);
      setExportSuccess(true);
      
      // Call custom export handler if provided
      onExport?.(exportConfig);
      
      // Close dialog after success
      setTimeout(() => {
        onClose();
        setActiveStep(0);
        setExportSuccess(false);
        setExportProgress(0);
      }, 1500);
      
    } catch (error) {
      setExportError(error instanceof Error ? error.message : 'Export failed');
    } finally {
      setIsExporting(false);
    }
  };

  // Handle column selection
  const handleColumnToggle = (column: string) => {
    setExportConfig(prev => ({
      ...prev,
      columns: prev.columns?.includes(column)
        ? prev.columns.filter(c => c !== column)
        : [...(prev.columns || []), column]
    }));
  };

  // Get preview data
  const getPreviewData = () => {
    return data.data.slice(0, 5).map(row => {
      const previewRow: any = {};
      exportConfig.columns?.forEach(col => {
        const column = data.columns?.find(c => c.key === col);
        previewRow[column?.label || col] = row[col];
      });
      return previewRow;
    });
  };

  const handleNext = () => {
    setActiveStep(prev => prev + 1);
  };

  const handleBack = () => {
    setActiveStep(prev => prev - 1);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: 500 }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DownloadIcon />
          Export {data.title}
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
          {steps.map(label => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        {activeStep === 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select Export Format
            </Typography>
            <RadioGroup
              value={exportConfig.format}
              onChange={(e) => setExportConfig({ ...exportConfig, format: e.target.value as ExportFormat })}
            >
              <Grid container spacing={2}>
                {allowedFormats.map(format => {
                  const config = formatConfigs[format];
                  return (
                    <Grid item xs={12} sm={6} key={format}>
                      <Paper
                        sx={{
                          p: 2,
                          cursor: 'pointer',
                          border: 2,
                          borderColor: exportConfig.format === format ? 'primary.main' : 'transparent',
                          '&:hover': {
                            borderColor: 'primary.light'
                          }
                        }}
                        onClick={() => setExportConfig({ ...exportConfig, format })}
                      >
                        <FormControlLabel
                          value={format}
                          control={<Radio />}
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Box sx={{ color: `${config.color}.main` }}>
                                {config.icon}
                              </Box>
                              <Box>
                                <Typography variant="subtitle1">
                                  {config.label}
                                </Typography>
                                <Typography variant="caption" color="textSecondary">
                                  {config.description}
                                </Typography>
                              </Box>
                            </Box>
                          }
                        />
                      </Paper>
                    </Grid>
                  );
                })}
              </Grid>
            </RadioGroup>
          </Box>
        )}
        
        {activeStep === 1 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configure Export Options
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                label="Filename"
                value={exportConfig.filename}
                onChange={(e) => setExportConfig({ ...exportConfig, filename: e.target.value })}
                helperText={`File will be saved as ${exportConfig.filename}.${exportConfig.format}`}
                margin="normal"
              />
            </Box>
            
            <Box sx={{ mb: 3 }}>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={exportConfig.includeHeaders}
                      onChange={(e) => setExportConfig({ ...exportConfig, includeHeaders: e.target.checked })}
                    />
                  }
                  label="Include column headers"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={exportConfig.includeMetadata}
                      onChange={(e) => setExportConfig({ ...exportConfig, includeMetadata: e.target.checked })}
                    />
                  }
                  label="Include metadata"
                />
                {exportConfig.format === 'excel' && (
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={exportConfig.compression || false}
                        onChange={(e) => setExportConfig({ ...exportConfig, compression: e.target.checked })}
                      />
                    }
                    label="Compress file"
                  />
                )}
              </FormGroup>
            </Box>
            
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Select Columns to Export
              </Typography>
              <Paper variant="outlined" sx={{ p: 2, maxHeight: 200, overflow: 'auto' }}>
                <FormGroup>
                  {data.columns?.map(column => (
                    <FormControlLabel
                      key={column.key}
                      control={
                        <Checkbox
                          checked={exportConfig.columns?.includes(column.key) || false}
                          onChange={() => handleColumnToggle(column.key)}
                        />
                      }
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {column.label}
                          {column.type && (
                            <Chip label={column.type} size="small" variant="outlined" />
                          )}
                        </Box>
                      }
                    />
                  ))}
                </FormGroup>
              </Paper>
            </Box>
          </Box>
        )}
        
        {activeStep === 2 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Preview & Export
            </Typography>
            
            {exportSuccess ? (
              <Alert severity="success" icon={<CheckIcon />}>
                Export completed successfully!
              </Alert>
            ) : exportError ? (
              <Alert severity="error" icon={<WarningIcon />}>
                {exportError}
              </Alert>
            ) : (
              <>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Export Summary</Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="Format" 
                        secondary={formatConfigs[exportConfig.format].label}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Filename" 
                        secondary={`${exportConfig.filename}.${exportConfig.format}`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Records" 
                        secondary={data.data.length}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Columns" 
                        secondary={exportConfig.columns?.length}
                      />
                    </ListItem>
                  </List>
                </Alert>
                
                <Typography variant="subtitle2" gutterBottom>
                  Data Preview (First 5 rows)
                </Typography>
                <Paper variant="outlined" sx={{ p: 1, overflow: 'auto', maxHeight: 200 }}>
                  <pre style={{ margin: 0, fontSize: '0.75rem' }}>
                    {JSON.stringify(getPreviewData(), null, 2)}
                  </pre>
                </Paper>
              </>
            )}
            
            {isExporting && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress variant="determinate" value={exportProgress} />
                <Typography variant="caption" color="textSecondary" sx={{ mt: 1 }}>
                  Exporting... {exportProgress}%
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        {activeStep > 0 && (
          <Button onClick={handleBack}>Back</Button>
        )}
        {activeStep < steps.length - 1 ? (
          <Button 
            onClick={handleNext} 
            variant="contained"
            disabled={activeStep === 1 && exportConfig.columns?.length === 0}
          >
            Next
          </Button>
        ) : (
          <Button
            onClick={handleExport}
            variant="contained"
            startIcon={<DownloadIcon />}
            disabled={isExporting || exportSuccess}
          >
            Export
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

// Export hook for easy integration
export const useExport = (data: ExportData) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const openExportDialog = useCallback(() => {
    setIsOpen(true);
  }, []);
  
  const closeExportDialog = useCallback(() => {
    setIsOpen(false);
  }, []);
  
  const ExportComponent = useCallback(() => (
    <UniversalExportManager
      open={isOpen}
      onClose={closeExportDialog}
      data={data}
    />
  ), [isOpen, data, closeExportDialog]);
  
  return {
    openExportDialog,
    closeExportDialog,
    ExportComponent
  };
};

export default UniversalExportManager;