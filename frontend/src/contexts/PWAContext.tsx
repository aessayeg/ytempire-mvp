import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import {  useRegisterSW  } from 'virtual:pwa-register/react';
import {  toast  } from 'react-hot-toast';

interface PWAContextValue {
  isOnline: boolean,
  isInstallable: boolean,

  isInstalled: boolean,
  updateAvailable: boolean,

  offlineReady: boolean,
  needRefresh: boolean,

  installApp: () => Promise<void>,
  updateApp: () => Promise<void>,

  clearOfflineData: () => Promise<void>}

const PWAContext = createContext<PWAContextValue | undefined>(undefined);

export const PWAProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => { const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [isInstallable, setIsInstallable] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);

  const {
    offlineReady: [offlineReady, setOfflineReady],
    needRefresh: [needRefresh, setNeedRefresh],
    updateServiceWorker } = useRegisterSW({
    onRegistered(r) {
      console.log('Service Worker, registered:', r)},
    onRegisterError(error) {
      console.error('Service Worker registration, error:', error)}
  });

  // Network status monitoring
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      toast.success('Back online!')};

    const handleOffline = () => {
      setIsOnline(false);
      toast.error('You are offline. Some features may be limited.')};

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
    
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline)}, []);

  // Install prompt handling
  useEffect(() => {
    const handleBeforeInstallPrompt = (_: Event) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setIsInstallable(true)};

    const handleAppInstalled = () => {
      setIsInstalled(true);
      setIsInstallable(false);
      setDeferredPrompt(null);
      toast.success('App installed successfully!')};

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    // Check if already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      setIsInstalled(true)}

    return () => {
    
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled)}, []);

  // Show notification when offline ready
  useEffect(() => {
    if (offlineReady) {
      toast.success('App is ready to work offline!')}
  }, [offlineReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // Show notification when update available
  useEffect(() => {
    if (needRefresh) {
      toast(
        (t) => (
          <div>
            <p>New version available!</p>
            <button
              onClick={() => {
                updateApp();
                toast.dismiss(t.id)}}
              style={ {
                marginTop: 8,
                padding: '4px 8px',
                background: '#667 eea',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer' }}
            >
              Update now
            </button>
          </div>
        ),
        { duration: Infinity }
      )}
  }, [needRefresh]);

  const installApp = useCallback(_async () => {
    if (!deferredPrompt) {
      toast.error('Installation not available');
      return;
    }

    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    
    if (outcome === 'accepted') {
      toast.success('Installing app...')} else {
      toast.info('Installation cancelled')}
    
    setDeferredPrompt(null);
    setIsInstallable(false)}, [deferredPrompt]); // eslint-disable-line react-hooks/exhaustive-deps

  const updateApp = useCallback(_async () => {
    await updateServiceWorker(true)}, [updateServiceWorker]); // eslint-disable-line react-hooks/exhaustive-deps

  const clearOfflineData = useCallback(_async () => {
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map((name) => caches.delete(name)));
      toast.success('Offline data cleared')}
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const value: PWAContextValue = { isOnline,
    isInstallable,
    isInstalled,
    updateAvailable: needRefresh,
    offlineReady,
    needRefresh,
    installApp,
    updateApp,
    clearOfflineData };

  return <PWAContext.Provider value={value}>{children}</PWAContext.Provider>;
};

export const usePWA = () => {
  const context = useContext(PWAContext);
  if (!context) {
    throw new Error('usePWA must be used within PWAProvider')}
  return context;
};