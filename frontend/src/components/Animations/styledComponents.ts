/**
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
