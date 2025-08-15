import React, { useRef, useState, useEffect } from 'react';
import {
  Box,
  IconButton,
  Slider,
  Typography,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  VolumeUp,
  VolumeOff,
  Fullscreen,
  FullscreenExit,
  Speed,
  Settings,
  PictureInPicture,
} from '@mui/icons-material';

interface VideoPlayerProps {
  videoUrl: string;
  thumbnail?: string;
  autoPlay?: boolean;
  controls?: boolean;
  muted?: boolean;
  onPlay?: () => void;
  onPause?: () => void;
  onEnded?: () => void;
  onTimeUpdate?: (currentTime: number, duration: number) => void;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoUrl,
  thumbnail,
  autoPlay = false,
  controls = true,
  muted = false,
  onPlay,
  onPause,
  onEnded,
  onTimeUpdate,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(muted);
  const [volume, setVolume] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showControls, setShowControls] = useState(true);
  const controlsTimeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
      setIsLoading(false);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      onTimeUpdate?.(video.currentTime, video.duration);
    };

    const handlePlay = () => {
      setIsPlaying(true);
      onPlay?.();
    };

    const handlePause = () => {
      setIsPlaying(false);
      onPause?.();
    };

    const handleEnded = () => {
      setIsPlaying(false);
      onEnded?.();
    };

    const handleError = () => {
      setError('Failed to load video');
      setIsLoading(false);
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);
    video.addEventListener('error', handleError);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('error', handleError);
    };
  }, [onPlay, onPause, onEnded, onTimeUpdate]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    video.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (event: Event, newValue: number | number[]) => {
    const video = videoRef.current;
    if (!video) return;

    const volumeValue = newValue as number;
    video.volume = volumeValue;
    setVolume(volumeValue);
    if (volumeValue === 0) {
      setIsMuted(true);
    } else if (isMuted) {
      setIsMuted(false);
    }
  };

  const handleSeek = (event: Event, newValue: number | number[]) => {
    const video = videoRef.current;
    if (!video) return;

    const time = newValue as number;
    video.currentTime = time;
    setCurrentTime(time);
  };

  const toggleFullscreen = async () => {
    const container = containerRef.current;
    if (!container) return;

    if (!isFullscreen) {
      if (container.requestFullscreen) {
        await container.requestFullscreen();
      }
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        await document.exitFullscreen();
      }
      setIsFullscreen(false);
    }
  };

  const togglePictureInPicture = async () => {
    const video = videoRef.current;
    if (!video) return;

    try {
      if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
      } else {
        await video.requestPictureInPicture();
      }
    } catch (_error) {
      console.error('PiP not supported');
    }
  };

  const changePlaybackRate = () => {
    const video = videoRef.current;
    if (!video) return;

    const rates = [0.5, 0.75, 1, 1.25, 1.5, 2];
    const currentIndex = rates.indexOf(playbackRate);
    const nextIndex = (currentIndex + 1) % rates.length;
    const newRate = rates[nextIndex];
    
    video.playbackRate = newRate;
    setPlaybackRate(newRate);
  };

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const handleMouseMove = () => {
    setShowControls(true);
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }
    controlsTimeoutRef.current = setTimeout(() => {
      if (isPlaying) {
        setShowControls(false);
      }
    }, 3000);
  };

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Paper
      ref={containerRef}
      sx={{
        position: 'relative',
        width: '100%',
        backgroundColor: 'black',
        overflow: 'hidden',
      }}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => isPlaying && setShowControls(false)}
    >
      <Box position="relative" paddingTop="56.25%">
        <video
          ref={videoRef}
          src={videoUrl}
          poster={thumbnail}
          autoPlay={autoPlay}
          muted={isMuted}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain',
          }}
          onClick={togglePlay}
        />

        {isLoading && (
          <Box
            position="absolute"
            top="50%"
            left="50%"
            sx={{ transform: 'translate(-50%, -50%)' }}
          >
            <CircularProgress size={60} sx={{ color: 'white' }} />
          </Box>
        )}

        {controls && showControls && (
          <Box
            position="absolute"
            bottom={0}
            left={0}
            right={0}
            sx={{
              background: 'linear-gradient(to top, rgba(0,0,0,0.8), transparent)',
              transition: 'opacity 0.3s',
              opacity: showControls ? 1 : 0,
            }}
          >
            {/* Progress Bar */}
            <Box px={2} py={1}>
              <Slider
                value={currentTime}
                max={duration}
                onChange={handleSeek}
                sx={{
                  color: 'primary.main',
                  '& .MuiSlider-thumb': {
                    width: 12,
                    height: 12,
                  },
                }}
              />
            </Box>

            {/* Controls */}
            <Box
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              px={2}
              pb={1}
            >
              <Box display="flex" alignItems="center" gap={1}>
                <IconButton onClick={togglePlay} sx={{ color: 'white' }}>
                  {isPlaying ? <Pause /> : <PlayArrow />}
                </IconButton>

                <IconButton onClick={toggleMute} sx={{ color: 'white' }}>
                  {isMuted ? <VolumeOff /> : <VolumeUp />}
                </IconButton>

                <Box width={100}>
                  <Slider
                    value={isMuted ? 0 : volume}
                    max={1}
                    step={0.1}
                    onChange={handleVolumeChange}
                    sx={{
                      color: 'white',
                      '& .MuiSlider-thumb': {
                        width: 12,
                        height: 12,
                      },
                    }}
                  />
                </Box>

                <Typography variant="body2" sx={{ color: 'white', ml: 2 }}>
                  {formatTime(currentTime)} / {formatTime(duration)}
                </Typography>
              </Box>

              <Box display="flex" alignItems="center" gap={1}>
                <IconButton
                  onClick={changePlaybackRate}
                  sx={{ color: 'white' }}
                  size="small"
                >
                  <Typography variant="caption">{playbackRate}x</Typography>
                </IconButton>

                <IconButton
                  onClick={togglePictureInPicture}
                  sx={{ color: 'white' }}
                  size="small"
                >
                  <PictureInPicture />
                </IconButton>

                <IconButton
                  onClick={toggleFullscreen}
                  sx={{ color: 'white' }}
                  size="small"
                >
                  {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
                </IconButton>
              </Box>
            </Box>
          </Box>
        )}
      </Box>
    </Paper>
  );
};