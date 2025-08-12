import React from 'react';
import { Tooltip, IconButton, Box, Typography } from '@mui/material';
import { HelpOutline, Info } from '@mui/icons-material';

interface HelpTooltipProps {
  title: string;
  description?: string;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  size?: 'small' | 'medium';
  variant?: 'help' | 'info';
  children?: React.ReactNode;
  interactive?: boolean;
}

export const HelpTooltip: React.FC<HelpTooltipProps> = ({
  title,
  description,
  placement = 'top',
  size = 'small',
  variant = 'help',
  children,
  interactive = false,
}) => {
  const Icon = variant === 'help' ? HelpOutline : Info;
  
  const tooltipContent = (
    <Box>
      <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
        {title}
      </Typography>
      {description && (
        <Typography variant="caption" sx={{ mt: 0.5, display: 'block', opacity: 0.9 }}>
          {description}
        </Typography>
      )}
    </Box>
  );

  if (children) {
    return (
      <Tooltip 
        title={tooltipContent} 
        placement={placement}
        arrow
        enterDelay={200}
        leaveDelay={interactive ? 200 : 0}
        interactive={interactive}
      >
        <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
          {children}
        </Box>
      </Tooltip>
    );
  }

  return (
    <Tooltip 
      title={tooltipContent} 
      placement={placement}
      arrow
      enterDelay={200}
      leaveDelay={interactive ? 200 : 0}
      interactive={interactive}
    >
      <IconButton size={size} sx={{ ml: 0.5, opacity: 0.6, '&:hover': { opacity: 1 } }}>
        <Icon fontSize={size} />
      </IconButton>
    </Tooltip>
  );
};