import React from 'react';
import { 
  Button,
  ButtonProps,
  CircularProgress
 } from '@mui/material';
import {  Check, Error  } from '@mui/icons-material';

interface LoadingButtonProps extends ButtonProps {
  loading?: boolean;
  success?: boolean;
  error?: boolean;
  loadingPosition?: 'start' | 'end' | 'center';
  loadingIndicator?: React.ReactNode;
  children: React.ReactNode}

export const LoadingButton: React.FC<LoadingButtonProps> = ({
  loading = false, success = false, error = false, loadingPosition = 'center', loadingIndicator, children, disabled, startIcon, endIcon, ...props
}) => {
  const getLoadingIndicator = () => {
    if (loadingIndicator) return loadingIndicator;
    return <CircularProgress size={20} color="inherit" />
  };
  const getContent = () => {
    if (success) {
      return (
        <>
          <Check sx={{ mr: 1 }} />
          {children}
        </>
      )}

    if (error) {
      return (
        <>
          <Error sx={{ mr: 1 }} />
          {children}
        </>
      )}

    if (loading) {
      switch (loadingPosition) {
        case 'start':
          return (
            <>
              {getLoadingIndicator()}
              <span style={{ marginLeft: 8 }}>{children}</span>
            </>
          );
        case 'end':
          return (
            <>
              <span style={{ marginRight: 8 }}>{children}</span>
              {getLoadingIndicator()}
            </>
          );
        case 'center':
          return getLoadingIndicator()}
    }

    return children
  };
  return (
    <Button
      disabled={disabled || loading}
      startIcon={!loading && !success && !error ? startIcon : undefined}
      endIcon={!loading && !success && !error ? endIcon : undefined}
      {...props}
    >
      {getContent()}
    </Button>
  )
};
// Button group with loading states
export const LoadingButtonGroup: React.FC<{
  buttons: Array<{

    key: string;
  label: string;,

    onClick: () => void;
    loading?: boolean;
    disabled?: boolean;
    variant?: 'text' | 'outlined' | 'contained';
    color?: 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success'}>;
  orientation?: 'horizontal' | 'vertical'}> = ({ buttons, orientation = 'horizontal' }) => { return (
    <div
      style={{
        display: 'flex',
        flexDirection: orientation === 'vertical' ? 'column' : 'row',
        gap: 8 }}
    >
      {buttons.map((button) => (
        <LoadingButton
          key={button.key}
          onClick={button.onClick}
          loading={button.loading}
          disabled={button.disabled}
          variant={button.variant || 'contained'}
          color={button.color || 'primary'}
        >
          {button.label}
        </LoadingButton>
      ))}
    </div>
  )
};
