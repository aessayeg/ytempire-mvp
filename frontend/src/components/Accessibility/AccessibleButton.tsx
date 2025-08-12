import React, { forwardRef } from 'react';
import { Button, Tooltip } from '@mui/material';
import type { ButtonProps } from '@mui/material';
import { getAriaProps } from '../../utils/accessibility';

interface AccessibleButtonProps extends ButtonProps {
  ariaLabel?: string;
  ariaDescribedBy?: string;
  ariaExpanded?: boolean;
  ariaPressed?: boolean;
  tooltip?: string;
  keyboardShortcut?: string;
  role?: string;
}

export const AccessibleButton = forwardRef<HTMLButtonElement, AccessibleButtonProps>(
  (
    {
      ariaLabel,
      ariaDescribedBy,
      ariaExpanded,
      ariaPressed,
      tooltip,
      keyboardShortcut,
      role,
      disabled,
      children,
      onClick,
      ...props
    },
    ref
  ) => {
    const ariaProps = getAriaProps({
      label: ariaLabel || (typeof children === 'string' ? children : undefined),
      describedBy: ariaDescribedBy,
      expanded: ariaExpanded,
      disabled,
      role,
    });

    // Handle keyboard shortcuts
    React.useEffect(() => {
      if (!keyboardShortcut || disabled) return;

      const handleKeyDown = (e: KeyboardEvent) => {
        const keys = keyboardShortcut.toLowerCase().split('+');
        const isMatch = keys.every((key) => {
          switch (key) {
            case 'ctrl':
              return e.ctrlKey;
            case 'alt':
              return e.altKey;
            case 'shift':
              return e.shiftKey;
            case 'meta':
              return e.metaKey;
            default:
              return e.key.toLowerCase() === key;
          }
        });

        if (isMatch) {
          e.preventDefault();
          onClick?.(e as any);
        }
      };

      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }, [keyboardShortcut, disabled, onClick]);

    const button = (
      <Button
        ref={ref}
        disabled={disabled}
        onClick={onClick}
        {...ariaProps}
        {...props}
        aria-pressed={ariaPressed}
      >
        {children}
      </Button>
    );

    if (tooltip) {
      const tooltipTitle = keyboardShortcut
        ? `${tooltip} (${keyboardShortcut})`
        : tooltip;

      return (
        <Tooltip title={tooltipTitle} arrow>
          <span>{button}</span>
        </Tooltip>
      );
    }

    return button;
  }
);

AccessibleButton.displayName = 'AccessibleButton';