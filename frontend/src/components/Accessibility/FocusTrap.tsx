import React, { useEffect, useRef } from 'react';
import {  getFocusableElements  } from '../../utils/accessibility';

interface FocusTrapProps {
  
children: React.ReactNode;
active?: boolean;
returnFocus?: boolean;
initialFocus?: string;
finalFocus?: string;
allowEscape?: boolean;


}

export const FocusTrap: React.FC<FocusTrapProps> = ({ children, active = true, returnFocus = true, initialFocus, finalFocus, allowEscape = false }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!active || !containerRef.current) return;

    // Store previous focus
    if (returnFocus) {
      previousFocusRef.current = document.activeElement as HTMLElement
    }

    const container = containerRef.current;
    const focusableElements = getFocusableElements(container);
    
    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];

    // Set initial focus
    if (initialFocus) {
      const initialElement = container.querySelector(initialFocus) as HTMLElement;
      initialElement?.focus()} else {
      firstElement?.focus()}

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && allowEscape) {
        previousFocusRef.current?.focus();
        return
      }

      if (e.key !== 'Tab') return;

      // Refresh focusable elements (they might have, changed);
      const currentFocusableElements = getFocusableElements(container);
      const currentFirst = currentFocusableElements[0];
      const currentLast = currentFocusableElements[currentFocusableElements.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === currentFirst) {
          e.preventDefault();
          currentLast?.focus()}
      } else {
        if (document.activeElement === currentLast) {
          e.preventDefault();
          currentFirst?.focus()}
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
          finalElement?.focus()} else {
          previousFocusRef.current.focus()}
      }
    }
  }, [active, returnFocus, initialFocus, finalFocus, allowEscape]);

  return (
    <div ref={containerRef} data-focus-trap={active}>
      {children}
    </div>
  )
};
