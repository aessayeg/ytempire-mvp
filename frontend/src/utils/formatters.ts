/**
 * Formatters - Data formatting utilities
 * Owner: Frontend Team Lead
 */

/**
 * Format number as currency
 */
export const formatCurrency = (
  amount: number,
  currency: string = 'USD',
  locale: string = 'en-US'
): string => {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(amount)
}

/**
 * Format number with thousand separators
 */
export const formatNumber = (
  num: number,
  locale: string = 'en-US',
  options?: Intl.NumberFormatOptions
): string => {
  return new Intl.NumberFormat(locale, options).format(num)
}

/**
 * Format number as percentage
 */
export const formatPercentage = (
  value: number,
  decimals: number = 1,
  locale: string = 'en-US'
): string => {
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(value / 100)
}

/**
 * Format large numbers with K, M, B suffixes
 */
export const formatCompactNumber = (
  num: number,
  locale: string = 'en-US'
): string => {
  return new Intl.NumberFormat(locale, {
    notation: 'compact',
    compactDisplay: 'short'
  }).format(num)
}

/**
 * Format date relative to now (e.g., "2 days ago")
 */
export const formatRelativeTime = (
  date: Date | string | number,
  locale: string = 'en-US'
): string => {
  const now = new Date()
  const targetDate = new Date(date)
  const diffInSeconds = Math.floor((now.getTime() - targetDate.getTime()) / 1000)

  if (Math.abs(diffInSeconds) < 60) {
    return 'just now'
  }

  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' })

  const intervals = [
    { unit: 'year' as const, seconds: 31536000 },
    { unit: 'month' as const, seconds: 2592000 },
    { unit: 'week' as const, seconds: 604800 },
    { unit: 'day' as const, seconds: 86400 },
    { unit: 'hour' as const, seconds: 3600 },
    { unit: 'minute' as const, seconds: 60 }
  ]

  for (const interval of intervals) {
    const count = Math.floor(Math.abs(diffInSeconds) / interval.seconds)
    if (count >= 1) {
      return rtf.format(diffInSeconds > 0 ? -count : count, interval.unit)
    }
  }

  return rtf.format(-Math.floor(Math.abs(diffInSeconds)), 'second')
}

/**
 * Format date
 */
export const formatDate = (
  date: Date | string | number,
  options?: Intl.DateTimeFormatOptions,
  locale: string = 'en-US'
): string => {
  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  }
  
  return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(new Date(date))
}

/**
 * Format time
 */
export const formatTime = (
  date: Date | string | number,
  options?: Intl.DateTimeFormatOptions,
  locale: string = 'en-US'
): string => {
  const defaultOptions: Intl.DateTimeFormatOptions = {
    hour: 'numeric',
    minute: 'numeric'
  }
  
  return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(new Date(date))
}

/**
 * Format datetime
 */
export const formatDateTime = (
  date: Date | string | number,
  options?: Intl.DateTimeFormatOptions,
  locale: string = 'en-US'
): string => {
  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric'
  }
  
  return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(new Date(date))
}

/**
 * Format duration in seconds to human readable format
 */
export const formatDuration = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`
  }
  
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) {
    const remainingSeconds = Math.round(seconds % 60)
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`
  }
  
  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  if (hours < 24) {
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`
  }
  
  const days = Math.floor(hours / 24)
  const remainingHours = hours % 24
  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`
}

/**
 * Format video duration (MM:SS or HH:MM:SS)
 */
export const formatVideoDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainingSeconds = Math.floor(seconds % 60)

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
  } else {
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }
}

/**
 * Format file size
 */
export const formatFileSize = (bytes: number, decimals: number = 2): string => {
  if (bytes === 0) return '0 B'
  
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

/**
 * Format phone number
 */
export const formatPhoneNumber = (phoneNumber: string): string => {
  // Remove all non-digits
  const cleaned = phoneNumber.replace(/\D/g, '')
  
  // Check if it's a US number
  if (cleaned.length === 10) {
    return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`
  } else if (cleaned.length === 11 && cleaned[0] === '1') {
    return `+1 (${cleaned.slice(1, 4)}) ${cleaned.slice(4, 7)}-${cleaned.slice(7)}`
  }
  
  // Return as-is if not a recognized format
  return phoneNumber
}

/**
 * Format credit card number
 */
export const formatCreditCard = (cardNumber: string): string => {
  const cleaned = cardNumber.replace(/\D/g, '')
  const groups = cleaned.match(/.{1,4}/g)
  return groups ? groups.join(' ') : cleaned
}

/**
 * Mask sensitive data (e.g., email, credit card)
 */
export const maskEmail = (email: string): string => {
  const [local, domain] = email.split('@')
  if (local.length <= 2) {
    return `${local[0]}*@${domain}`
  }
  return `${local[0]}${'*'.repeat(local.length - 2)}${local[local.length - 1]}@${domain}`
}

export const maskCreditCard = (cardNumber: string): string => {
  const cleaned = cardNumber.replace(/\D/g, '')
  if (cleaned.length < 4) return cardNumber
  return `**** **** **** ${cleaned.slice(-4)}`
}

export const maskPhone = (phoneNumber: string): string => {
  const cleaned = phoneNumber.replace(/\D/g, '')
  if (cleaned.length < 4) return phoneNumber
  return `***-***-${cleaned.slice(-4)}`
}

/**
 * Format URL for display (remove protocol, www)
 */
export const formatDisplayUrl = (url: string): string => {
  try {
    const urlObj = new URL(url)
    let hostname = urlObj.hostname
    
    // Remove www.
    if (hostname.startsWith('www.')) {
      hostname = hostname.slice(4)
    }
    
    return hostname + urlObj.pathname
  } catch {
    return url
  }
}

/**
 * Format social media handles
 */
export const formatHandle = (handle: string, platform: 'twitter' | 'instagram' | 'youtube'): string => {
  const cleaned = handle.replace(/^@/, '')
  
  switch (platform) {
    case 'twitter':
    case 'instagram':
      return `@${cleaned}`
    case 'youtube':
      return cleaned.startsWith('UC') ? cleaned : `@${cleaned}`
    default:
      return cleaned
  }
}

/**
 * Format YouTube video URL
 */
export const formatYouTubeUrl = (videoId: string): string => {
  return `https://www.youtube.com/watch?v=${videoId}`
}

/**
 * Extract YouTube video ID from URL
 */
export const extractYouTubeVideoId = (url: string): string | null => {
  const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/
  const match = url.match(regex)
  return match ? match[1] : null
}

/**
 * Format text with line breaks to HTML
 */
export const formatTextToHtml = (text: string): string => {
  return text
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^/, '<p>')
    .replace(/$/, '</p>')
}

/**
 * Format mentions and hashtags
 */
export const formatSocialText = (text: string): string => {
  return text
    .replace(/@(\w+)/g, '<a href="/users/$1" class="mention">@$1</a>')
    .replace(/#(\w+)/g, '<a href="/tags/$1" class="hashtag">#$1</a>')
}

/**
 * Format code blocks
 */
export const formatCodeBlock = (code: string, language?: string): string => {
  const escapedCode = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
  
  return `<pre><code${language ? ` class="language-${language}"` : ''}>${escapedCode}</code></pre>`
}

/**
 * Format search query highlights
 */
export const highlightSearchTerms = (text: string, query: string): string => {
  if (!query.trim()) return text
  
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  return text.replace(regex, '<mark>$1</mark>')
}

/**
 * Format initials from name
 */
export const formatInitials = (name: string, maxLength: number = 2): string => {
  return name
    .split(' ')
    .map(word => word.charAt(0))
    .join('')
    .toUpperCase()
    .slice(0, maxLength)
}

/**
 * Format address
 */
export const formatAddress = (address: {
  street?: string
  city?: string
  state?: string
  zipCode?: string
  country?: string
}): string => {
  const parts = [
    address.street,
    address.city,
    address.state && address.zipCode ? `${address.state} ${address.zipCode}` : address.state || address.zipCode,
    address.country
  ].filter(Boolean)
  
  return parts.join(', ')
}

/**
 * Format list with conjunctions
 */
export const formatList = (items: string[], conjunction: string = 'and'): string => {
  if (items.length === 0) return ''
  if (items.length === 1) return items[0]
  if (items.length === 2) return `${items[0]} ${conjunction} ${items[1]}`
  
  return `${items.slice(0, -1).join(', ')}, ${conjunction} ${items[items.length - 1]}`
}

/**
 * Format error messages
 */
export const formatErrorMessage = (error: any): string => {
  if (typeof error === 'string') return error
  if (error?.message) return error.message
  if (error?.detail) return error.detail
  if (error?.error) return formatErrorMessage(error.error)
  return 'An unexpected error occurred'
}

/**
 * Format validation errors
 */
export const formatValidationErrors = (errors: Record<string, string[]>): string[] => {
  return Object.entries(errors).flatMap(([field, fieldErrors]) => 
    fieldErrors.map(error => `${field}: ${error}`)
  )
}