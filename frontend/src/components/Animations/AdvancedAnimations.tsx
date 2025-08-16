/**
 * Advanced Animation Components and Hooks
 * Provides sophisticated animation effects throughout the application
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {  motion, AnimatePresence, useInView, useScroll, useTransform  } from 'framer-motion';
import { 
  Box
 } from '@mui/material';
import {  useEnhancedTheme  } from '../../contexts/EnhancedThemeContext';

import {  fadeInUp, fadeInScale  } from './variants';


// Animated card component
export const AnimatedCard: React.FC<{
  children: React.ReactNode;
  delay?: number;
  className?: string
}> = ({ children, delay = 0, className }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, amount: 0.3 });
  const { themeConfig } = useEnhancedTheme();

  if (!themeConfig.animationsEnabled) {
    return <div className={className}>{children}</div>
  }

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 50 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
      transition={{
        duration: 0.6,
        delay,
        ease: [0.48, 0.15, 0.25, 0.96]
      }}
      whileHover={{
        scale: 1.02,
        boxShadow: '0 10px 30px rgba(0,0,0,0.2)'
      }}
      className={className}
    >
      {children}
    </motion.div>
  );

// Parallax scroll component
export const ParallaxSection: React.FC<{
  children: React.ReactNode;
  speed?: number}> = ({ children, speed = 0.5 }) => {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start']
  });
  
  const y = useTransform(scrollYProgress, [0, 1], [0, speed * 100]);
  const { themeConfig } = useEnhancedTheme();

  if (!themeConfig.animationsEnabled) {
    return <div>{children}</div>
  }

  return (
    <motion.div ref={ref} style={{ y }}>
      {children}
    </motion.div>
  );

// Morphing shape background
export const MorphingBackground: React.FC = () => {
  const { isDarkMode, themeConfig } = useEnhancedTheme();
  
  if (!themeConfig.animationsEnabled) {
    return null
  }

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        overflow: 'hidden',
        zIndex: -1

       }}
    >
      <svg
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%'

         }}
        viewBox="0 0 1440 800"
      >
        <motion.path
          d="M, 0,400 C, 0,400, 0,200, 0,200 C 114.35714285714286,156.53571428571428 228.71428571428572,113.07142857142857, 351,131 C 473.2857142857143,148.92857142857142 603.5,228.25, 710,245 C 816.5,261.75 899.2857142857142,216.92857142857142, 1029,196 C 1158.7142857142858,175.07142857142858 1335.357142857143,177.03571428571428, 1440,200 C, 1440,200, 1440,400, 1440,400 Z"
          fill={isDarkMode ? '#1e1e1e' : '#f0f0f0'}
          fillOpacity="0.3"
          animate={{
            d: [
              "M, 0,400 C, 0,400, 0,200, 0,200 C 114.35714285714286,156.53571428571428 228.71428571428572,113.07142857142857, 351,131 C 473.2857142857143,148.92857142857142 603.5,228.25, 710,245 C 816.5,261.75 899.2857142857142,216.92857142857142, 1029,196 C 1158.7142857142858,175.07142857142858 1335.357142857143,177.03571428571428, 1440,200 C, 1440,200, 1440,400, 1440,400 Z",
              "M, 0,400 C, 0,400, 0,200, 0,200 C 89.35714285714286,244.17857142857142 178.71428571428572,288.35714285714283, 306,276 C 433.2857142857143,263.6428571428571 598.5,194.75, 741,183 C 883.5,171.25 1003.2857142857142,216.64285714285714, 1123,234 C 1242.7142857142858,251.35714285714286 1362.357142857143,240.67857142857142, 1440,235 C, 1440,235, 1440,400, 1440,400 Z",
              "M, 0,400 C, 0,400, 0,200, 0,200 C 114.35714285714286,156.53571428571428 228.71428571428572,113.07142857142857, 351,131 C 473.2857142857143,148.92857142857142 603.5,228.25, 710,245 C 816.5,261.75 899.2857142857142,216.92857142857142, 1029,196 C 1158.7142857142858,175.07142857142858 1335.357142857143,177.03571428571428, 1440,200 C, 1440,200, 1440,400, 1440,400 Z"
            ]
          }}
          transition={{
            repeat: Infinity,
            repeatType: "reverse",
            duration: 10,
            ease: "easeInOut"

          }}
        />
      </svg>
    </Box>
  );

// Animated counter
export const AnimatedCounter: React.FC<{
  value: number;
  duration?: number;
  prefix?: string;
  suffix?: string}> = ({ value, duration = 2, prefix = '', suffix = '' }) => {
  const [displayValue, setDisplayValue] = useState(0);
  const { themeConfig } = useEnhancedTheme();

  useEffect(() => {
    if (!themeConfig.animationsEnabled) {
      setDisplayValue(value);
      return
    }

    const startTime = Date.now();

    const updateValue = () => {
      const now = Date.now();
      const progress = Math.min((now - startTime) / (duration * 1000), 1);
      
      // Easing function
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentValue = Math.floor(easeOutQuart * value);
      
      setDisplayValue(currentValue);
      
      if (progress < 1) {
        requestAnimationFrame(updateValue)}
    };
    requestAnimationFrame(updateValue)}, [value, duration, themeConfig.animationsEnabled]);

  return (
    <span>
      {prefix}{displayValue.toLocaleString()}{suffix}
    </span>
  );

// Typewriter effect
export const TypewriterText: React.FC<{
  text: string;
  delay?: number;
  speed?: number}> = ({ text, delay = 0, speed = 50 }) => {
  const [displayedText, setDisplayedText] = useState('');
  const { themeConfig } = useEnhancedTheme();

  useEffect(() => {
    if (!themeConfig.animationsEnabled) {
      setDisplayedText(text);
      return
    }

    let index = 0;
    const timeout = setTimeout(() => {
      const interval = setInterval(() => {
        if (index < text.length) {
          setDisplayedText(text.slice(0, index + 1));
          index++
        } else {
          clearInterval(interval)}
      }, speed);
      
      return () => clearInterval(interval)
    }, delay);
    
    return () => clearTimeout(timeout)
  }, [text, delay, speed, themeConfig.animationsEnabled]);

  return <span>{displayedText}</span>
};
// Ripple effect component
export const RippleButton: React.FC<{
  children: React.ReactNode;
  onClick?: () => void;
  className?: string
}> = ({ children, onClick, className }) => {
  const [ripples, setRipples] = useState<Array<{ x: number; y: number; id: number }>>([]);
  const { themeConfig } = useEnhancedTheme();

  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!themeConfig.animationsEnabled) {
      onClick?.();;
      return
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const id = Date.now();
    
    setRipples(prev => [...prev, { x, y, id }]);
    
    setTimeout(() => {
      setRipples(prev => prev.filter(r => r.id !== id))
    }, 600);
    
    onClick?.();
  }, [onClick, themeConfig.animationsEnabled]);

  return (
    <Box
      onClick={handleClick}
      className={className}
      sx={{
        position: 'relative',
        overflow: 'hidden',
        cursor: 'pointer'

       }}
    >
      {children}
      <AnimatePresence>
        {ripples.map(ripple => (
          <motion.div
            key={ripple.id}
            style={{
              position: 'absolute',
              left: ripple.x,
              top: ripple.y,
              transform: 'translate(-50%, -50%)',
              borderRadius: '50%',
              backgroundColor: 'rgba(255, 255, 255, 0.5)'
            }}
            initial={{ width: 0, height: 0, opacity: 1 }}
            animate={{ width: 200, height: 200, opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          />
        ))}
      </AnimatePresence>
    </Box>
  );

// Page transition wrapper
export const PageTransition: React.FC<{
  children: React.ReactNode;
  variant?: 'fade' | 'slide' | 'scale'}> = ({ children, variant = 'fade' }) => {
  const { themeConfig } = useEnhancedTheme();
  
  if (!themeConfig.animationsEnabled) {
    return <>{children}</>
  }

  const variants = {
    fade: fadeInUp,
    slide: slideInRight,
    scale: fadeInScale
  };
  return (
    <AnimatePresence mode="wait">
      <motion.div
        variants={variants[variant]}
        initial="initial"
        animate="animate"
        exit="exit"
        transition={{ duration: 0.3 }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  )
};
// Loading skeleton with animation
export const AnimatedSkeleton: React.FC<{
  width?: string | number;
  height?: string | number;
  variant?: 'text' | 'rectangular' | 'circular'}> = ({ width = '100%', height = 20, variant = 'text' }) => {
  const { isDarkMode } = useEnhancedTheme();
  
  const baseColor = isDarkMode ? '#2a2a2a' : '#e0e0e0';
  const highlightColor = isDarkMode ? '#3a3a3a' : '#f0f0f0';
  
  return (
    <Box
      sx={{
        width,
        height,
        backgroundColor: baseColor,
        borderRadius: variant === 'circular' ? '50%' : variant === 'text' ? 1 : 2,
        position: 'relative',
        overflow: 'hidden',
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: '-150%',
          width: '150%',
          height: '100%',
          background: `linear-gradient(90deg, transparent, ${highlightColor}, transparent)`,
          animation: `${shimmer} 2s infinite}}
    />
  );

// Floating action button with animation
export const FloatingActionButton: React.FC<{
  children: React.ReactNode;
  onClick?: () => void
}> = ({ children, onClick }) => {
  const { themeConfig } = useEnhancedTheme();
  
  if (!themeConfig.animationsEnabled) {
    return (
    <Box
        onClick={onClick}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          width: 56,
          height: 56,
          borderRadius: '50%',
          backgroundColor: 'primary.main',
          color: 'primary.contrastText',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          boxShadow: 3
}}
      >
        {children}
      </Box>
    )}

  return (
    <motion.div
      style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        width: 56,
        height: 56,
        borderRadius: '50%',
        backgroundColor: 'var(--primary-color)',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        boxShadow: '0 4px 10px rgba(0,0,0,0.3)'
      }}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );

// Export all animation utilities

export default {
  AnimatedCard,
  ParallaxSection,
  MorphingBackground,
  AnimatedCounter,
  TypewriterText,
  RippleButton,
  PageTransition,
  AnimatedSkeleton,
  FloatingActionButton
  };
