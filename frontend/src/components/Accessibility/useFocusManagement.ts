import {  useRef  } from 'react';
import {  getFocusableElements  } from '../../utils/accessibility';

// Hook for managing focus
export const useFocusManagement = () => {
  const focusHistory = useRef<HTMLElement[]>([]);

  const pushFocus = (element?: HTMLElement) => {
    const current = element || (document.activeElement as HTMLElement);
    focusHistory.current.push(current)};

  const popFocus = () => {
    const previous = focusHistory.current.pop();
    if (previous && document.body.contains(previous)) {
      previous.focus()}
  };

  const clearFocusHistory = () => {
    focusHistory.current = [];
  };

  const focusElement = (selector: string) => {
    const element = document.querySelector(selector) as HTMLElement;
    if (element) {
      pushFocus();
      element.focus()}
  };

  const focusFirst = (container?: HTMLElement) => {
    const root = container || document.body;
    const focusableElements = getFocusableElements(root);
    if (focusableElements[0]) {
      pushFocus();
      focusableElements[0].focus()}
  };

  const focusLast = (container?: HTMLElement) => {
    const root = container || document.body;
    const focusableElements = getFocusableElements(root);
    const lastElement = focusableElements[focusableElements.length - 1];
    if (lastElement) {
      pushFocus();
      lastElement.focus()}
  };

  return {
    pushFocus,
    popFocus,
    clearFocusHistory,
    focusElement,
    focusFirst,
    focusLast
  };
};