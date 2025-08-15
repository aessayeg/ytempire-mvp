interface Announcement {
  id: string;
  message: string;
  priority: 'polite' | 'assertive';
}

class AnnouncementManager {
  private listeners: ((announcement: Announcement) => void)[] = [];
  
  announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
    const announcement: Announcement = {
      id: Date.now().toString(),
      message,
      priority,
    };
    
    this.listeners.forEach((listener) => listener(announcement));
  }
  
  subscribe(listener: (announcement: Announcement) => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }
}

export const announcementManager = new AnnouncementManager();

// Hook for using announcements
export const useAnnounce = () => {
  return {
    announce: (message: string, priority: 'polite' | 'assertive' = 'polite') => {
      announcementManager.announce(message, priority);
    },
    announcePolite: (message: string) => {
      announcementManager.announce(message, 'polite');
    },
    announceAssertive: (message: string) => {
      announcementManager.announce(message, 'assertive');
    },
  };
};

export type { Announcement };