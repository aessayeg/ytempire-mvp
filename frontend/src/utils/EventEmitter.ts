/**
 * Browser-compatible EventEmitter
 * Simple implementation for WebSocket and other event-based systems
 */

type EventListener = (...args: unknown[]) => void;

export class EventEmitter {
  private events: Map<string, Set<EventListener>> = new Map();

  on(event: string, listener: EventListener): this {
    if (!this.events.has(event)) {
      this.events.set(event, new Set())
    }
    this.events.get(event)!.add(listener);
    return this
  }

  once(event: string, listener: EventListener): this {
    const onceWrapper = (...args: unknown[]) => {
      listener(...args);
      this.off(event, onceWrapper)
    };
    return this.on(event, onceWrapper)
  }

  off(event: string, listener: EventListener): this {
    const listeners = this.events.get(event);
    if (listeners) {
      listeners.delete(listener);
      if (listeners.size === 0) {
        this.events.delete(event)
      }
    }
    return this
  }

  emit(event: string, ...args: unknown[]): boolean {
    const listeners = this.events.get(event);
    if (!listeners || listeners.size === 0) {
      return false
    }
    listeners.forEach(listener => {
      try {
        listener(...args)
      } catch (error) {
        console.error(`Error in event listener for "${event}":`, error)
      }
    });
    return true
  }

  removeAllListeners(event?: string): this {
    if (event) {
      this.events.delete(event)
    } else {
      this.events.clear()
    }
    return this
  }

  listenerCount(event: string): number {
    const listeners = this.events.get(event);
    return listeners ? listeners.size : 0
  }

  eventNames(): string[] {
    return Array.from(this.events.keys())}}
