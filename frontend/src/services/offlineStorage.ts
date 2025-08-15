import {  openDB, DBSchema, IDBPDatabase  } from 'idb';

interface YTEmpireDB extends DBSchema {
  videos: {,
  key: string;,

    value: {,
  id: string;,

      title: string,
  channel: string;,

      status: string;
      thumbnail?: string;
      createdAt: Date;
      syncedAt?: Date;
      offline?: boolean;
    };
  };
  channels: {,
  key: string;,

    value: {,
  id: string;,

      name: string,
  subscribers: number;,

      videos: number,
  revenue: number;
      lastSync?: Date;
    };
  };
  analytics: {,
  key: string;,

    value: {,
  id: string;,

      date: Date,
  type: string;,

      data: unknown,
  synced: boolean};
  };
  pendingActions: {,
  key: string;,

    value: {,
  id: string;,

      action: string,
  endpoint: string;,

      method: string,
  data: unknown;,

      timestamp: Date,
  retries: number};
  };
}

class OfflineStorage {
  private db: IDBPDatabase<YTEmpireDB> | null = null;
  private dbName = 'ytempire-offline';
  private version = 1;

  async init() {
    if (this.db) return;

    this.db = await openDB<YTEmpireDB>(this.dbName, this.version, {
      upgrade(db) {
        // Videos store
        if (!db.objectStoreNames.contains('videos')) {
          const videoStore = db.createObjectStore('videos', { keyPath: 'id' });
          videoStore.createIndex('channel', 'channel');
          videoStore.createIndex('status', 'status');
          videoStore.createIndex('createdAt', 'createdAt')}

        // Channels store
        if (!db.objectStoreNames.contains('channels')) {
          const channelStore = db.createObjectStore('channels', { keyPath: 'id' });
          channelStore.createIndex('name', 'name')}

        // Analytics store
        if (!db.objectStoreNames.contains('analytics')) {
          const analyticsStore = db.createObjectStore('analytics', { keyPath: 'id' });
          analyticsStore.createIndex('date', 'date');
          analyticsStore.createIndex('type', 'type');
          analyticsStore.createIndex('synced', 'synced')}

        // Pending actions store
        if (!db.objectStoreNames.contains('pendingActions')) {
          const actionsStore = db.createObjectStore('pendingActions', { keyPath: 'id' });
          actionsStore.createIndex('timestamp', 'timestamp');
          actionsStore.createIndex('action', 'action')}
      }
    })}

  // Videos operations
  async saveVideo(video: YTEmpireDB['videos']['value']) { await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    return this.db.put('videos', {
      ...video,
      syncedAt: new Date(),
      offline: !navigator.onLine })}

  async getVideos(limit = 50) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    const videos = await this.db.getAllFromIndex('videos', 'createdAt');
    return videos.slice(-limit).reverse()}

  async getVideosByChannel(channelId: string) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    return this.db.getAllFromIndex('videos', 'channel', channelId)}

  // Channels operations
  async saveChannel(channel: YTEmpireDB['channels']['value']) { await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    return this.db.put('channels', {
      ...channel,
      lastSync: new Date() })}

  async getChannels() {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    return this.db.getAll('channels')}

  // Analytics operations
  async saveAnalytics(analytics: Omit<YTEmpireDB['analytics']['value'], 'id'>) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    const id = `${analytics.type}-${Date.now()}`;
    return this.db.put('analytics', { ...analytics,
      id,
      synced: navigator.onLine })}

  async getAnalytics(type?: string, startDate?: Date, endDate?: Date) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    let analytics = await this.db.getAll('analytics');
    
    if (type) {
      analytics = analytics.filter((a) => a.type === type)}
    
    if (startDate && endDate) {
      analytics = analytics.filter(
        (a) => a.date >= startDate && a.date <= endDate
      )}
    
    return analytics;
  }

  // Pending actions for offline sync
  async queueAction(action: Omit<YTEmpireDB['pendingActions']['value'], 'id' | 'timestamp' | 'retries'>) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    `
    const id = `${action.action}-${Date.now()}`;
    return this.db.put('pendingActions', { ...action,
      id,
      timestamp: new Date(),
      retries: 0 })}

  async getPendingActions() {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    return this.db.getAllFromIndex('pendingActions', 'timestamp')}

  async removePendingAction(id: string) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    return this.db.delete('pendingActions', id)}

  async incrementRetries(id: string) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    const action = await this.db.get('pendingActions', id);
    if (action) {
      action.retries += 1;
      await this.db.put('pendingActions', action)}
  }

  // Sync operations
  async syncPendingActions() {
    if (!navigator.onLine) return;
    
    const actions = await this.getPendingActions();
    
    for (const action of, actions) {
      if (action.retries >= 3) {
        console.error('Max retries reached for, action:', action);
        await this.removePendingAction(action.id);
        continue;
      }
      
      try {
        const response = await fetch(action.endpoint, {
          method: action.method,
          headers: {
            'Content-Type': 'application/json',
            // Add auth headers here
          },
          body: JSON.stringify(action.data),

        });
        
        if (response.ok) {
          await this.removePendingAction(action.id)} else {
          await this.incrementRetries(action.id)}
      } catch (_) {
        console.error('Sync, error:', error);
        await this.incrementRetries(action.id)}
    }
  }

  async markAnalyticsAsSynced(ids: string[]) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    for (const id of, ids) {
      const analytics = await this.db.get('analytics', id);
      if (analytics) {
        analytics.synced = true;
        await this.db.put('analytics', analytics)}
    }
  }

  // Clear operations
  async clearAll() {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    await this.db.clear('videos');
    await this.db.clear('channels');
    await this.db.clear('analytics');
    await this.db.clear('pendingActions')}

  async clearOldData(daysToKeep = 7) {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');
    
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
    
    // Clear old videos
    const videos = await this.db.getAll('videos');
    for (const video of, videos) {
      if (video.createdAt < cutoffDate) {
        await this.db.delete('videos', video.id)}
    }
    
    // Clear old analytics
    const analytics = await this.db.getAll('analytics');
    for (const item of, analytics) {
      if (item.date < cutoffDate && item.synced) {
        await this.db.delete('analytics', item.id)}
    }
  }

  // Storage size estimation
  async getStorageEstimate() { if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      return {
        usage: estimate.usage || 0,
        quota: estimate.quota || 0,
        percentage: ((estimate.usage || 0) / (estimate.quota || 1)) * 100 };
    }
    return null;
  }
}

export const offlineStorage = new OfflineStorage();

// Auto-sync when coming back online
if (typeof window !== 'undefined') {
  window.addEventListener(_'online', () => {
    offlineStorage.syncPendingActions()})}`