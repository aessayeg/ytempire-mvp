import React, { useState } from 'react';
import { 
  Box,
  Paper,
  Typography,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Link,
  Chip,
  Alert
 } from '@mui/material';
import { 
  Close,
  CheckCircle,
  Lightbulb,
  School,
  KeyboardArrowRight,
  KeyboardArrowDown,
  VideoLibrary,
  QuestionAnswer
 } from '@mui/icons-material';

interface HelpItem {
  id: string,
  title: string,

  content: string;
  type?: 'tip' | 'tutorial' | 'faq';
  link?: string;
  videoUrl?: string;
}

interface InlineHelpProps {
  context: string;
  items?: HelpItem[];
  onClose?: () => void;
  persistent?: boolean;
  variant?: 'compact' | 'expanded';
}
const defaultHelpItems: Record<string, HelpItem[]> = { 'video-creation': [ {
      id: '1',
      title: 'Quick Start',
      content: 'Start by selecting a trending topic or enter your own. Our AI will generate an optimized script.',
      type: 'tip' },
    { id: '2',
      title: 'Batch Processing',
      content: 'Create up to 10 videos at once by enabling batch mode for better efficiency.',
      type: 'tip' },
    { id: '3',
      title: 'Cost Optimization',
      content: 'Use GPT-3.5 for drafts and GPT-4 for final versions to reduce costs by 40%.',
      type: 'tip' } ],
  'channel-management': [ { id: '1',
      title: 'Channel Health',
      content: 'Keep your channel health above 80% for optimal YouTube algorithm performance.',
      type: 'tip' },
    { id: '2',
      title: 'Multi-Channel Strategy',
      content: 'Diversify content across channels to test different niches and audiences.',
      type: 'tutorial',
      link: '/docs/multi-channel' },
    { id: '3',
      title: 'Scheduling Best Practices',
      content: 'Schedule videos during peak hours (2-4 PM and 7-9, PM) for maximum engagement.',
      type: 'tip' } ],
  'analytics': [ { id: '1',
      title: 'Key Metrics',
      content: 'Focus on CTR (Click-Through, Rate) and AVD (Average View, Duration) for growth.',
      type: 'tip' },
    { id: '2',
      title: 'Revenue Tracking',
      content: 'Monitor RPM (Revenue Per, Mille) trends to optimize content strategy.',
      type: 'tutorial',
      videoUrl: '/tutorials/revenue-tracking' } ]
};

export const InlineHelp: React.FC<InlineHelpProps> = ({ context, items, onClose, persistent = false, variant = 'compact' }) => {
  const [expanded, setExpanded] = useState(variant === 'expanded');
  const [completedItems, setCompletedItems] = useState<string[]>([]);
  
  const helpItems = items || defaultHelpItems[context] || [];
  
  const handleItemComplete = (itemId: string) => {
    setCompletedItems(prev => [...prev, itemId])};
  
  const getIcon = (_type?: string) => {
    switch (type) {
      case 'tip':
        return <Lightbulb color="primary" />;
      case 'tutorial':
        return <School color="secondary" />;
      case 'faq':
        return <QuestionAnswer />;
        return <CheckCircle color="success" />}
  };
  
  if (helpItems.length === 0) return null;
  
  return (
    <>
      <Paper
      elevation={2}
      sx={ {
        p: 2,
        mb: 2,
        backgroundColor: 'background.paper',
        border: 1,
        borderColor: 'divider' }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton
            size="small"
            onClick={() => setExpanded(!expanded}
            sx={{ p: 0.5 }}
          >
            {expanded ? <KeyboardArrowDown /> </>: <KeyboardArrowRight />}
          </IconButton>
      <Typography variant="subtitle2" fontWeight="medium">
            Quick Help
          </Typography>
          <Chip
            label={`${completedItems.length}/${helpItems.length}`}
            size="small"
            color={completedItems.length === helpItems.length ? 'success' : 'default'}
          />
        </Box>
        {!persistent && onClose && (
          <IconButton size="small" onClick={onClose}>
            <Close fontSize="small" />
          </IconButton>
        )}
      </Box>
      
      <Collapse in={expanded}>
        <Box sx={{ mt: 2 }}>
          {variant === 'compact' ? (
            <List dense sx={{ p: 0 }}>
              {helpItems.map((item) => (
                <ListItem
                  key={item.id}
                  sx={ {
                    pl: 0,
                    opacity: completedItems.includes(item.id) ? 0.6 : 1,
                    textDecoration: completedItems.includes(item.id) ? 'line-through' : 'none' }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    {completedItems.includes(item.id) ? (
                      <CheckCircle color="success" fontSize="small" />
                    ) : (
                      getIcon(item.type)
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.title}
                    secondary={item.content}
                    primaryTypographyProps={{ variant: 'body2', fontWeight: 'medium' }}
                    secondaryTypographyProps={{ variant: 'caption' }}
                  />
                  {!completedItems.includes(item.id) && (
                    <IconButton
                      size="small"
                      onClick={() => handleItemComplete(item.id}
                      sx={{ ml: 1 }}
                    >
                      <CheckCircle fontSize="small" />
                    </IconButton>
                  )}
                </ListItem>
              ))}
            </List>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {helpItems.map((item) => (
                <Alert
                  key={item.id}
                  severity={item.type === 'tip' ? 'info' : 'success'}
                  icon={getIcon(item.type)}
                  action={
                    !completedItems.includes(item.id) && (
                      <IconButton
                        size="small"
                        onClick={() => handleItemComplete(item.id}
                      >
                        <CheckCircle />
                      </IconButton>
                    )}
                >
                  <Typography variant="body2" fontWeight="medium">
                    {item.title}
                  </Typography>
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    {item.content}
                  </Typography>
                  {item.link && (
                    <Link href={item.link} sx={{ fontSize: 12, mt: 1, display: 'inline-block' }}>
                      Learn more →
                    </Link>
                  )}
                  {item.videoUrl && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
                      <VideoLibrary fontSize="small" />
                      <Link href={item.videoUrl} sx={{ fontSize: 12 }}>
                        Watch tutorial
                      </Link>
                    </Box>
                  )}
                </Alert>
              ))}
            </Box>
          )}
          {completedItems.length === helpItems.length && (
            <Alert severity="success" sx={{ mt: 2 }}>
              <Typography variant="caption">
                Great job! You've completed all help items. 
              </Typography>
            </Alert>
          )}
        </Box>
      </Collapse>
    </Paper>
  </>
  )};