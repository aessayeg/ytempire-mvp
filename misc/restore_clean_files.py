#!/usr/bin/env python3
"""
Restore corrupted TypeScript files to a clean, working state
"""

import os
from pathlib import Path

def restore_styled_components():
    """Restore styledComponents.ts to a clean state"""
    content = '''/**
 * Animated Styled Components
 * Pre-built animated components using CSS keyframes
 */

import { 
  Box,
  styled,
  keyframes
} from '@mui/material';

// Keyframe animations
const pulse = keyframes`
  0% { transform: scale(1) }
  50% { transform: scale(1.05) }
  100% { transform: scale(1) }
`;

const shimmer = keyframes`
  0% { background-position: -1000px 0 }
  100% { background-position: 1000px 0 }
`;

const rotate = keyframes`
  from { transform: rotate(0deg) }
  to { transform: rotate(360deg) }
`;

const bounce = keyframes`
  0%, 100% { transform: translateY(0) }
  50% { transform: translateY(-20px) }
`;

const glow = keyframes`
  0% { box-shadow: 0 0 5px rgba(66, 153, 225, 0.5) }
  50% { box-shadow: 0 0 20px rgba(66, 153, 225, 0.8), 0 0 30px rgba(66, 153, 225, 0.6) }
  100% { box-shadow: 0 0 5px rgba(66, 153, 225, 0.5) }
`;

// Styled components with animations
export const PulsingBox = styled(Box)`
  animation: ${pulse} 2s ease-in-out infinite;
`;

export const ShimmerLoader = styled(Box)`
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0) 0%,
    rgba(255, 255, 255, 0.3) 50%,
    rgba(255, 255, 255, 0) 100%
  );
  background-size: 1000px 100%;
  animation: ${shimmer} 2s infinite;
`;

export const SpinningLoader = styled(Box)`
  animation: ${rotate} 1s linear infinite;
`;

export const BouncingBox = styled(Box)`
  animation: ${bounce} 1.5s ease-in-out infinite;
`;

export const GlowingBox = styled(Box)`
  animation: ${glow} 2s ease-in-out infinite;
`;

// Complex animated components
export const AnimatedCard = styled(Box)`
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  }
`;

export const AnimatedButton = styled(Box)`
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.5s, height 0.5s;
  }
  
  &:hover::before {
    width: 300px;
    height: 300px;
  }
`;

export const FloatingElement = styled(Box)`
  animation: ${keyframes`
    0%, 100% { transform: translateY(0) }
    25% { transform: translateY(-10px) rotate(5deg) }
    75% { transform: translateY(10px) rotate(-5deg) }
  `} 3s ease-in-out infinite;
`;

export const ProgressBar = styled(Box)`
  position: relative;
  height: 4px;
  background: rgba(0, 0, 0, 0.1);
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      90deg,
      transparent,
      #4299e1,
      transparent
    );
    animation: ${shimmer} 2s linear infinite;
  }
`;

export const TypewriterCursor = styled('span')`
  @keyframes blink {
    0%, 50% { opacity: 1 }
    51%, 100% { opacity: 0 }
  }
  
  &::after {
    content: '|';
    animation: blink 1s step-start infinite;
  }
`;
'''
    return content

def restore_variants():
    """Restore variants.ts to a clean state"""
    content = '''/**
 * Animation Variants
 * Reusable animation variants for consistent motion design
 */

import { Variants } from 'framer-motion';

// Basic transitions
export const fadeInUp: Variants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};

export const fadeInScale: Variants = {
  initial: { opacity: 0, scale: 0.9 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.9 }
};

export const slideInLeft: Variants = {
  initial: { opacity: 0, x: -50 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: 50 }
};

export const slideInRight: Variants = {
  initial: { opacity: 0, x: 50 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -50 }
};

export const staggerContainer: Variants = {
  animate: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2
    }
  }
};

export const staggerItem: Variants = {
  initial: { opacity: 0, y: 20 },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.4, 0, 0.2, 1]
    }
  }
};

// Complex animations
export const morphingCard: Variants = {
  initial: {
    borderRadius: '8px',
    scale: 1
  },
  hover: {
    borderRadius: '16px',
    scale: 1.02,
    transition: {
      duration: 0.3,
      ease: [0.4, 0, 0.2, 1]
    }
  }
};

export const floatingEffect: Variants = {
  initial: { y: 0 },
  animate: {
    y: [-5, 5, -5],
    transition: {
      duration: 3,
      repeat: Infinity,
      ease: 'easeInOut'
    }
  }
};

export const pulseEffect: Variants = {
  initial: { scale: 1, opacity: 1 },
  animate: {
    scale: [1, 1.1, 1],
    opacity: [1, 0.8, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: 'easeInOut'
    }
  }
};

export const typewriterContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.03,
      delayChildren: 0.5
    }
  }
};

export const typewriterChild: Variants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: {
      duration: 0.1
    }
  }
};
'''
    return content

def main():
    """Main function to restore files"""
    
    frontend_dir = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend")
    
    # Files to restore
    files_to_restore = {
        "src/components/Animations/styledComponents.ts": restore_styled_components(),
        "src/components/Animations/variants.ts": restore_variants(),
    }
    
    for file_path, content in files_to_restore.items():
        full_path = frontend_dir / file_path
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Restored: {file_path}")
    
    print("\nFiles restored successfully")

if __name__ == "__main__":
    main()