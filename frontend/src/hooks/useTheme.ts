/**
 * Theme Hook
 * Owner: Dashboard Specialist
 */

import { useThemeContext } from '@/contexts/ThemeContext'

export const useTheme = () => {
  return useThemeContext()
}

export default useTheme