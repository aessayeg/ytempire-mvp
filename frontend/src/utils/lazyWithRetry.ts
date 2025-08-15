/**
 * Enhanced lazy loading with retry logic for code splitting
 * Handles network failures and chunk loading errors gracefully
 */

import { lazy } from 'react';
import type { ComponentType } from 'react';

const MAX_RETRY_COUNT = 3;
const RETRY_DELAY = 1000;

interface ImportCallbackModule {
  default: ComponentType<unknown>;
}

type ImportCallback = () => Promise<ImportCallbackModule>;

interface RetryState {
  count: number;
  lastError?: Error;
}

const retryStates = new Map<string, RetryState>();

export function lazyWithRetry(
  importCallback: ImportCallback,
  componentName?: string
): React.LazyExoticComponent<ComponentType<unknown>> {
  const key = componentName || importCallback.toString();
  
  return lazy(async () => {
    const state = retryStates.get(key) || { count: 0 };
    
    const retry = async (error: Error): Promise<ImportCallbackModule> => {
      state.count += 1;
      state.lastError = error;
      retryStates.set(key, state);
      
      if (state.count > MAX_RETRY_COUNT) {
        // If we've exceeded max retries, try to reload the page once
        if (!sessionStorage.getItem(`reload_attempted_${key}`)) {
          sessionStorage.setItem(`reload_attempted_${key}`, 'true');
          window.location.reload();
          // This won't be reached, but TypeScript needs it
          return Promise.reject(error);
        }
        
        console.error(`Failed to load component after ${MAX_RETRY_COUNT} retries:`, error);
        throw new Error(`Unable to load component. Please refresh the page or contact support.`);
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * state.count));
      
      try {
        const module = await importCallback();
        // Reset state on success
        retryStates.delete(key);
        sessionStorage.removeItem(`reload_attempted_${key}`);
        return module;
      } catch (_retryError) {
        return retry(retryError as Error);
      }
    };
    
    try {
      const module = await importCallback();
      // Reset state on success
      retryStates.delete(key);
      sessionStorage.removeItem(`reload_attempted_${key}`);
      return module;
    } catch (_error) {
      console.warn(`Failed to load component, attempting retry:`, error);
      return retry(error as Error);
    }
  });
}

// Preload function for critical routes
export function preloadComponent(
  importCallback: ImportCallback
): Promise<void> {
  return importCallback().then(() => undefined).catch(() => undefined);
}

// Intersection Observer for predictive preloading
export function setupPredictivePreloading(
  linkSelector: string = 'a[href^="/"]',
  preloadMap: Map<string, ImportCallback>
): void {
  if (!('IntersectionObserver' in window)) return;
  
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const link = entry.target as HTMLAnchorElement;
          const path = link.getAttribute('href');
          if (path && preloadMap.has(path)) {
            const importCallback = preloadMap.get(path);
            if (importCallback) {
              preloadComponent(importCallback);
            }
          }
        }
      });
    },
    {
      rootMargin: '50px'
    }
  );
  
  // Observe all internal links
  document.querySelectorAll(linkSelector).forEach((link) => {
    observer.observe(link);
  });
  
  // Handle dynamically added links
  const mutationObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) {
          const element = node as Element;
          if (element.matches(linkSelector)) {
            observer.observe(element);
          }
          element.querySelectorAll(linkSelector).forEach((link) => {
            observer.observe(link);
          });
        }
      });
    });
  });
  
  mutationObserver.observe(document.body, {
    childList: true,
    subtree: true
  });
}