/**
 * Custom Heatmap Grid Component
 * React 19 compatible replacement for react-grid-heatmap
 */
import React from 'react';
import { Box } from '@mui/material';

interface HeatmapGridProps {
  data: number[][];
  xLabels: string[];
  yLabels: string[];
  cellRender?: (x: number, y: number, value: number) => React.ReactNode;
  xLabelsStyle?: () => React.CSSProperties;
  yLabelsStyle?: () => React.CSSProperties;
  cellStyle?: (x: number, y: number, ratio: number) => React.CSSProperties;
  cellHeight?: string;
  square?: boolean;
}

export const HeatMapGrid: React.FC<HeatmapGridProps> = ({
  data,
  xLabels,
  yLabels,
  cellRender,
  xLabelsStyle = () => ({}),
  yLabelsStyle = () => ({}),
  cellStyle = () => ({}),
  cellHeight = '30px',
  square = false
}) => {
  // Calculate max value for ratio
  const maxValue = Math.max(...data.flat());
  const minValue = Math.min(...data.flat());
  const range = maxValue - minValue || 1;

  const getCellContent = (x: number, y: number, value: number) => {
    if (cellRender) {
      return cellRender(x, y, value);
    }
    return value;
  };

  const getCellStyles = (x: number, y: number, value: number): React.CSSProperties => {
    const ratio = (value - minValue) / range;
    const defaultStyles: React.CSSProperties = {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: cellHeight,
      width: square ? cellHeight : 'auto',
      minWidth: square ? cellHeight : '60px',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      fontSize: '0.75rem',
      fontWeight: 500
    };
    
    const customStyles = cellStyle(x, y, ratio);
    return { ...defaultStyles, ...customStyles };
  };

  return (
    <Box sx={{ overflowX: 'auto', overflowY: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr>
            <th style={{ 
              position: 'sticky', 
              left: 0, 
              background: 'white',
              zIndex: 2,
              padding: '8px',
              ...yLabelsStyle()
            }}></th>
            {xLabels.map((label, index) => (
              <th 
                key={index} 
                style={{ 
                  padding: '8px',
                  textAlign: 'center',
                  fontWeight: 'normal',
                  ...xLabelsStyle()
                }}
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {yLabels.map((yLabel, yIndex) => (
            <tr key={yIndex}>
              <td style={{ 
                position: 'sticky', 
                left: 0, 
                background: 'white',
                padding: '8px',
                fontWeight: 'normal',
                ...yLabelsStyle()
              }}>
                {yLabel}
              </td>
              {xLabels.map((_, xIndex) => {
                const value = data[yIndex]?.[xIndex] ?? 0;
                return (
                  <td 
                    key={xIndex} 
                    style={{ padding: '2px' }}
                  >
                    <div style={getCellStyles(xIndex, yIndex, value)}>
                      {getCellContent(xIndex, yIndex, value)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Box>
  );
};

export default HeatMapGrid;