/**
 * Animated Styled Components
 * Pre-built animated components using CSS keyframes
 */

import { Box, styled, keyframes } from '@mui/material';

// Keyframe animations
const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const shimmer = keyframes`
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
`;

const rotate = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

const bounce = keyframes`
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-20px); }
`;

const glow = keyframes`
  0% { box-shadow: 0 0 5px rgba(66, 153, 225, 0.5); }
  50% { box-shadow: 0 0 20px rgba(66, 153, 225, 0.8), 0 0 30px rgba(66, 153, 225, 0.6); }
  100% { box-shadow: 0 0 5px rgba(66, 153, 225, 0.5); }
`;

// Styled components with animations
export const PulseBox = styled(Box)`
  animation: ${pulse} 2s ease-in-out infinite;
`;

export const ShimmerBox = styled(Box)`
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 1000px 100%;
  animation: ${shimmer} 2s infinite;
`;

export const RotateBox = styled(Box)`
  animation: ${rotate} 2s linear infinite;
`;

export const BounceBox = styled(Box)`
  animation: ${bounce} 1s ease-in-out infinite;
`;

export const GlowBox = styled(Box)`
  animation: ${glow} 2s ease-in-out infinite;
`;