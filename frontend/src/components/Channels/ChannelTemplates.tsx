import React, { useState } from 'react';
import { 
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel
 } from '@mui/material';
import { 
  Add,
  Edit,
  Delete,
  ContentCopy,
  Save,
  Star,
  StarBorder,
  Category,
  Schedule,
  Visibility,
  MonetizationOn,
  Settings,
  VideoLibrary,
  Description,
  Public,
  CheckCircle,
  Upload
 } from '@mui/icons-material';

interface Template {
  id: string,
  name: string,

  description: string,
  type: 'channel' | 'video' | 'scheduling' | 'monetization',

  category: string;
  thumbnail?: string;
  isFavorite: boolean,
  isDefault: boolean,

  usage: number;
  lastUsed?: Date;
  config: {
    // Channel Template Config
    channelSettings?: {
      category: string,
  tags: string[],

      description: string,
  keywords: string[],

      country: string,
  language: string};
    // Video Template Config
    videoSettings?: {
      title: string,
  description: string,

      tags: string[],
  category: string,

      visibility: 'public' | 'unlisted' | 'private',
  thumbnail: string,

      endScreen: boolean,
  cards: boolean,

      comments: boolean,
  likes: boolean};
    // Scheduling Template Config
    schedulingSettings?: {
      publishTime: string,
  timezone: string,

      frequency: 'daily' | 'weekly' | 'custom',
  daysOfWeek: number[],

      maxPerDay: number,
  spreadOverDay: boolean};
    // Monetization Template Config
    monetizationSettings?: {
      enabled: boolean,
  midrollAds: boolean,

      productPlacement: boolean,
  paidPromotion: boolean,

      merchShelf: boolean,
  channelMemberships: boolean,

      superChat: boolean};
  };
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number,
  value: number}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <>
      <div hidden={value !== index}>
      {value === index && <Box sx={{ py: 2 }}>{children}</Box>}
    </div>
  </>
  )};

export const ChannelTemplates: React.FC = () => { const [templates, setTemplates] = useState<Template[]>([
    {
      id: '1',
      name: 'Tech Channel Starter',
      description: 'Perfect template for technology-focused channels',
      type: 'channel',
      category: 'Technology',
      isFavorite: true,
      isDefault: true,
      usage: 156,
      lastUsed: new Date(),
      config: {,
  channelSettings: {,

          category: 'Science & Technology',
          tags: ['tech', 'technology', 'innovation', 'gadgets', 'AI'],
          description: 'Exploring the latest in technology and innovation',
          keywords: ['technology', 'artificial intelligence', 'gadgets'],
          country: 'US',
          language: 'en' }
      }
    },
    { id: '2',
      name: 'Daily Upload Schedule',
      description: 'Optimized schedule for daily content creators',
      type: 'scheduling',
      category: 'Scheduling',
      isFavorite: false,
      isDefault: false,
      usage: 89,
      config: {,
  schedulingSettings: {,

          publishTime: '14:00',
          timezone: 'America/New_York',
          frequency: 'daily',
          daysOfWeek: [1, 2, 3, 4, 5],
          maxPerDay: 2,
          spreadOverDay: true }
      }
    },
    {
      id: '3',
      name: 'Tutorial Video Template',
      description: 'Structured template for educational content',
      type: 'video',
      category: 'Education',
      isFavorite: true,
      isDefault: false,
      usage: 234,
      config: {,
  videoSettings: {,

          title: '[Tutorial] {topic} - Complete Guide',
          description: 'In this tutorial, you will learn...',
          tags: ['tutorial', 'how to', 'guide', 'education'],
          category: 'Education',
          visibility: 'public',
          thumbnail: 'tutorial_thumb.jpg',
          endScreen: true,
          cards: true,
          comments: true,
          likes: true
}
      }
    },
    { id: '4',
      name: 'Full Monetization',
      description: 'Enable all monetization features',
      type: 'monetization',
      category: 'Revenue',
      isFavorite: false,
      isDefault: false,
      usage: 67,
      config: {,
  monetizationSettings: {,

          enabled: true,
          midrollAds: true,
          productPlacement: false,
          paidPromotion: false,
          merchShelf: true,
          channelMemberships: true,
          superChat: true }
      }
    }]);

  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [editDialog, setEditDialog] = useState(false);
  const [createDialog, setCreateDialog] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [filter, setFilter] = useState('all');

  const templateCategories = [ { value: 'all', label: 'All Templates' },
    { value: 'channel', label: 'Channel Setup' },
    { value: 'video', label: 'Video Creation' },
    { value: 'scheduling', label: 'Scheduling' },
    { value: 'monetization', label: 'Monetization' } ];

  const handleApplyTemplate = (template: Template) => {
    // Apply template logic
    console.log('Applying, template:', template)};

  const handleToggleFavorite = (templateId: string) => {
    setTemplates(prev => prev.map(t => {}
      t.id === templateId ? { ...t, isFavorite: !t.isFavorite } : t
    ))};

  const handleDuplicateTemplate = (template: Template) => {
const newTemplate: Template = {
      ...template,
      id: `template-${Date.now()}`,
      name: `${template.name} (Copy)`,
      isDefault: false,
      usage: 0
};
    setTemplates(prev => [...prev, newTemplate])};

  const handleDeleteTemplate = (templateId: string) => {
    setTemplates(prev => prev.filter(t => t.id !== templateId))};

  const getTemplateIcon = (type: string) => {
    switch (type) {
      case 'channel': return <Settings />;
      case 'video': return <VideoLibrary />;
      case 'scheduling': return <Schedule />;
      case 'monetization': return <MonetizationOn />;
      default: return <Category />}
  };

  const filteredTemplates = filter === 'all'
    ? templates
    : templates.filter(t => t.type === filter);

  const renderTemplateCard = (template: Template) => (
    <Grid item xs={12} sm={6} md={4} key={template.id}>
      <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Avatar sx={{ bgcolor: 'primary.main' }}>
              {getTemplateIcon(template.type)}
            </Avatar>
      <IconButton
              onClick={() => handleToggleFavorite(template.id}
              size="small"
            >
              {template.isFavorite ? <Star color="warning" /> </>: <StarBorder />}
            </IconButton>
          </Box>

          <Typography variant="h6" gutterBottom>
            {template.name}
          </Typography>

          <Typography variant="body2" color="text.secondary" gutterBottom>
            {template.description}
          </Typography>

          <Box sx={{ display: 'flex', gap: 1, mt: 2, flexWrap: 'wrap' }}>
            <Chip
              label={template.type}
              size="small"
              color="primary"
              variant="outlined"
            />
            {template.isDefault && (
              <Chip label="Default" size="small" color="success" />
            )}
            <Chip
              label={`Used ${template.usage} times`}
              size="small"
              variant="outlined"
            />
          </Box>
        </CardContent>

        <CardActions>
          <Button
            size="small"
            startIcon={<CheckCircle />}
            onClick={() => handleApplyTemplate(template}
          >
            Apply
          </Button>
          <Button
            size="small"
            startIcon={<Edit />}
            onClick={() => {
              setSelectedTemplate(template);
              setEditDialog(true)}}
          >
            Edit
          </Button>
          <IconButton
            size="small"
            onClick={() => handleDuplicateTemplate(template}
          >
            <ContentCopy fontSize="small" />
          </IconButton>
          {!template.isDefault && (
            <IconButton
              size="small"
              onClick={() => handleDeleteTemplate(template.id}
            >
              <Delete fontSize="small" />
            </IconButton>
          )}
        </CardActions>
      </Card>
    </Grid>
  );

  const renderTemplateForm = () => (
    <Box>
      <TextField
        fullWidth
        label="Template Name"
        value={selectedTemplate?.name || ''}
        sx={{ mb: 2 }}
      />
      
      <TextField
        fullWidth
        multiline
        rows={3}
        label="Description"
        value={selectedTemplate?.description || ''}
        sx={{ mb: 2 }}
      />

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Template Type</InputLabel>
        <Select value={selectedTemplate?.type || 'channel'}>
          <MenuItem value="channel">Channel Setup</MenuItem>
          <MenuItem value="video">Video Creation</MenuItem>
          <MenuItem value="scheduling">Scheduling</MenuItem>
          <MenuItem value="monetization">Monetization</MenuItem>
        </Select>
      </FormControl>

      <Divider sx={{ my: 3 }} />

      {/* Template-specific settings */}
      {selectedTemplate?.type === 'channel' && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Channel Settings
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Category</InputLabel>
                <Select value={selectedTemplate?.config.channelSettings?.category || ''}>
                  <MenuItem value="Science & Technology">Science & Technology</MenuItem>
                  <MenuItem value="Education">Education</MenuItem>
                  <MenuItem value="Entertainment">Entertainment</MenuItem>
                  <MenuItem value="Gaming">Gaming</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Tags (comma, separated)"
                value={selectedTemplate?.config.channelSettings?.tags.join(', ') || ''}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Channel Description"
                value={selectedTemplate?.config.channelSettings?.description || ''}
              />
            </Grid>
          </Grid>
        </Box>
      )}
      {selectedTemplate?.type === 'video' && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Video Settings
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Title Template"
                value={selectedTemplate?.config.videoSettings?.title || ''}
                helperText="Use {topic} as a placeholder"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Default Visibility</InputLabel>
                <Select value={selectedTemplate?.config.videoSettings?.visibility || 'public'}>
                  <MenuItem value="public">Public</MenuItem>
                  <MenuItem value="unlisted">Unlisted</MenuItem>
                  <MenuItem value="private">Private</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormGroup>
                <FormControlLabel
                  control={<Switch checked={selectedTemplate?.config.videoSettings?.comments || false} />}
                  label="Enable Comments"
                />
                <FormControlLabel
                  control={<Switch checked={selectedTemplate?.config.videoSettings?.likes || false} />}
                  label="Enable Likes"
                />
                <FormControlLabel
                  control={<Switch checked={selectedTemplate?.config.videoSettings?.endScreen || false} />}
                  label="Add End Screen"
                />
              </FormGroup>
            </Grid>
          </Grid>
        </Box>
      )}
      {selectedTemplate?.type === 'scheduling' && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Scheduling Settings
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Publish Time"
                type="time"
                value={selectedTemplate?.config.schedulingSettings?.publishTime || '14:00'}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Frequency</InputLabel>
                <Select value={selectedTemplate?.config.schedulingSettings?.frequency || 'daily'}>
                  <MenuItem value="daily">Daily</MenuItem>
                  <MenuItem value="weekly">Weekly</MenuItem>
                  <MenuItem value="custom">Custom</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Max Videos Per Day"
                type="number"
                value={selectedTemplate?.config.schedulingSettings?.maxPerDay || 2}
              />
            </Grid>
            <Grid item xs={6}>
              <FormControlLabel
                control={<Switch checked={selectedTemplate?.config.schedulingSettings?.spreadOverDay || false} />}
                label="Spread Over Day"
              />
            </Grid>
          </Grid>
        </Box>
      )}
      {selectedTemplate?.type === 'monetization' && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Monetization Settings
          </Typography>
          <FormGroup>
            <FormControlLabel
              control={<Switch checked={selectedTemplate?.config.monetizationSettings?.enabled || false} />}
              label="Enable Monetization"
            />
            <FormControlLabel
              control={<Switch checked={selectedTemplate?.config.monetizationSettings?.midrollAds || false} />}
              label="Mid-roll Ads"
            />
            <FormControlLabel
              control={<Switch checked={selectedTemplate?.config.monetizationSettings?.merchShelf || false} />}
              label="Merchandise Shelf"
            />
            <FormControlLabel
              control={<Switch checked={selectedTemplate?.config.monetizationSettings?.channelMemberships || false} />}
              label="Channel Memberships"
            />
            <FormControlLabel
              control={<Switch checked={selectedTemplate?.config.monetizationSettings?.superChat || false} />}
              label="Super Chat & Super Stickers"
            />
          </FormGroup>
        </Box>
      )}
    </Box>
  );

  return (
    <>
      <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          Channel Templates
        </Typography>
      <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Upload />}
          >
            Import
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setCreateDialog(true}
          >
            Create Template
          </Button>
        </Box>
      </Box>

      {/* Filters */}
      <Tabs
        value={tabValue}
        onChange={(_, newValue) => {
          setTabValue(newValue</>
  );
          setFilter(templateCategories[newValue].value)}}
        sx={{ mb: 3 }}
      >
        {templateCategories.map((cat, index) => (
          <Tab key={cat.value} label={cat.label} />
        ))}
      </Tabs>

      {/* Templates Grid */}
      <Grid container spacing={3}>
        {filteredTemplates.map(renderTemplateCard)}
      </Grid>

      {/* Edit Template Dialog */}
      <Dialog
        open={editDialog}
        onClose={() => setEditDialog(false}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Edit Template
        </DialogTitle>
        <DialogContent>
          {renderTemplateForm()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(false}>
            Cancel
          </Button>
          <Button variant="contained" startIcon={<Save />}>
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create Template Dialog */}
      <Dialog
        open={createDialog}
        onClose={() => setCreateDialog(false}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Create New Template
        </DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            Templates help you standardize settings across multiple channels and videos.
          </Alert>
          {renderTemplateForm()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialog(false}>
            Cancel
          </Button>
          <Button variant="contained" startIcon={<Add />}>
            Create Template
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )};