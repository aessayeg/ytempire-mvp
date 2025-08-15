import React, { useEffect, useState } from 'react';
import { 
  Box
 } from '@mui/material';
import {  announcementManager, type Announcement  } from './announcementManager';

export const ScreenReaderAnnouncer: React.FC = () => {
  const [announcements, setAnnouncements] = useState<Announcement[]>([]);
  
  useEffect(() => {
    const unsubscribe = announcementManager.subscribe((announcement) => {
      setAnnouncements((prev) => [...prev, announcement]);
      
      // Remove announcement after 1 second
      setTimeout(() => {
        setAnnouncements((prev) => prev.filter((a) => a.id !== announcement.id))}, 1000)});
    
    return unsubscribe;
  }, []);
  
  return (
    <>
      <Box
        component="div"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        sx={{
          position: 'absolute',
          left: '-10000px',
          width: '1px',
          height: '1px',
          overflow: 'hidden',

        }}
      >
        {announcements
          .filter((a) => a.priority === 'polite')
          .map((a) => a.message)
          .join('. ')}
      </Box>
      
      <Box
        component="div"
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
        sx={{
          position: 'absolute',
          left: '-10000px',
          width: '1px',
          height: '1px',
          overflow: 'hidden',

        }}
      >
        {announcements
          .filter((a) => a.priority === 'assertive')
          .map((a) => a.message)
          .join('. ')}
      </Box>
    </>
  )};

