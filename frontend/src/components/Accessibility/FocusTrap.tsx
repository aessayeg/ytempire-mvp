import React, { useEffect, useRef } from 'react';
import { getFocusableElements } from '../../utils/accessibility';

interface FocusTrapProps {
  children: React.ReactNode;
  active?: boolean;
  returnFocus?: boolean;
  initialFocus?: string;
  finalFocus?: string;
  allowEscape?: boolean;
}

export const FocusTrap: React.FC<FocusTrapProps> = ({
  children,
  active = true,
  returnFocus = true,
  initialFocus,
  finalFocus,
  allowEscape = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!active || !containerRef.current) return;

    // Store previous focus
    if (returnFocus) {
      previousFocusRef.current = document.activeElement as HTMLElement;
    }

    const container = containerRef.current;
    const focusableElements = getFocusableElements(container);
    
    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Set initial focus
    if (initialFocus) {
      const initialElement = container.querySelector(initialFocus) as HTMLElement;
      initialElement?.focus();
    } else {
      firstElement?.focus();
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && allowEscape) {
        previousFocusRef.current?.focus();
        return;
      }

      if (e.key !== 'Tab') return;

      // Refresh focusable elements (they might have changed)
      const currentFocusableElements = getFocusableElements(container);
      const currentFirst = currentFocusableElements[0];
      const currentLast = currentFocusableElements[currentFocusableElements.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === currentFirst) {
          e.preventDefault();
          currentLast?.focus();
        }
      } else {
        if (document.activeElement === currentLast) {
          e.preventDefault();
          currentFirst?.focus();
        }
      }
    };

    // Add event listener
    container.addEventListener('keydown', handleKeyDown);

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
      
      // Return focus
      if (returnFocus && previousFocusRef.current) {
        if (finalFocus) {
          const finalElement = document.querySelector(finalFocus) as HTMLElement;
          finalElement?.focus();
        } else {
          previousFocusRef.current.focus();
        }
      }
    };
  }, [active, returnFocus, initialFocus, finalFocus, allowEscape]);

  return (
    <div ref={containerRef} data-focus-trap={active}>
      {children}
    </div>
  );
};

// Hook for managing focus
export const useFocusManagement = () => {
  const focusHistory = useRef<HTMLElement[]>([]);

  const pushFocus = (element?: HTMLElement) => {
    const current = element || (document.activeElement as HTMLElement);
    focusHistory.current.push(current);
  };

  const popFocus = () => {
    const previous = focusHistory.current.pop();
    if (previous && document.body.contains(previous)) {
      previous.focus();
    }
  };

  const clearFocusHistory = () => {
    focusHistory.current = [];
  };

  const focusElement = (selector: string) => {
    const element = document.querySelector(selector) as HTMLElement;
    if (element) {
      pushFocus();
      element.focus();
    }
  };

  const focusFirst = (container?: HTMLElement) => {
    const root = container || document.body;
    const focusableElements = getFocusableElements(root);
    if (focusableElements[0]) {
      pushFocus();
      focusableElements[0].focus();
    }
  };

  const focusLast = (container?: HTMLElement) => {
    const root = container || document.body;
    const focusableElements = getFocusableElements(root);
    const lastElement = focusableElements[focusableElements.length - 1];
    if (lastElement) {
      pushFocus();
      lastElement.focus();
    }
  };

  return {
    pushFocus,
    popFocus,
    clearFocusHistory,
    focusElement,
    focusFirst,
    focusLast,
  };
};