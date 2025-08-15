import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Avatar,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Menu,
  MenuItem,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField
          placeholder="Search videos..."
          size="small"
          InputProps={{
            startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
          }}
          sx={{ flex: 1, minWidth: 200 }}
        />
        
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Status</InputLabel>
          <Select
            value={videoFilter}
            onChange={(e) => setVideoFilter(e.target.value)}
            label="Status"
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="published">Published</MenuItem>
            <MenuItem value="scheduled">Scheduled</MenuItem>
            <MenuItem value="processing">Processing</MenuItem>
            <MenuItem value="draft">Draft</MenuItem>
          </Select>
        </FormControl>
        
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Sort By</InputLabel>
          <Select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            label="Sort By"
          >
            <MenuItem value="date">Date</MenuItem>
            <MenuItem value="views">Views</MenuItem>
            <MenuItem value="revenue">Revenue</MenuItem>
            <MenuItem value="engagement">Engagement</MenuItem>
          </Select>
        </FormControl>
        
        <Button
          variant="outlined"
          startIcon={<Refresh />}
        >
          Refresh
        </Button>
      </Box>

      {/* Video List */}
      <List>
        {videos.map((video, index) => (
          <React.Fragment key={video.id}>
            <ListItem alignItems="flex-start">
              <ListItemAvatar>
                <Avatar
                  variant="rounded"
                  src={video.thumbnail}
                  sx={{ width: 120, height: 67.5, mr: 2 }}
                >
                  <PlayCircle />
                </Avatar>
              </ListItemAvatar>
              
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle1" fontWeight="medium">
                      {video.title}
                    </Typography>
                    <Chip
                      label={video.status}
                      size="small"
                      color={getStatusColor(video.status)}
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {video.status === 'published' && video.publishedAt
                        ? `Published ${formatDistanceToNow(video.publishedAt, { addSuffix: true });
}
                        : video.status === 'scheduled' && video.scheduledAt
                        ? `Scheduled for ${format(video.scheduledAt, 'PPp')}
                        : `Status: ${video.status}`}
                    </Typography>
                    
                    {video.status === 'published' && (
                      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                        <Chip
                          icon={<Visibility />}
                          label={video.views.toLocaleString()}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          icon={<ThumbUp />}
                          label={video.likes.toLocaleString()}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          icon={<Comment />}
                          label={video.comments.toLocaleString()}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          icon={<AttachMoney />}
                          label={`$${video.revenue.toFixed(2)}`}
                          size="small"
                          variant="outlined"
                          color="success"
                        />
                      </Box>
                    )}
                    {video.status === 'published' && (
                      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          CTR: {video.ctr}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          AVD: {video.avd}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Duration: {video.duration}
                        </Typography>
                      </Box>
                    )}
                  </Box>
                }
              />
              
              <ListItemSecondaryAction>
                <IconButton
                  edge="end"
                  onClick={((e) => handleVideoMenu(e, video)}
                >
                  <MoreVert />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
            {index < videos.length - 1 && <Divider component="li" />}
          </React.Fragment>
        ))}
      </List>

      {/* Video Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl) && Boolean(selectedVideo)}
        onClose={handleCloseMenu}
      >
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <Edit fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <ContentCopy fontSize="small" />
          </ListItemIcon>
          <ListItemText>Duplicate</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <Share fontSize="small" />
          </ListItemIcon>
          <ListItemText>Share</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <Download fontSize="small" />
          </ListItemIcon>
          <ListItemText>Download</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleCloseMenu} sx={{ color: 'error.main' }}>
          <ListItemIcon>
            <Delete fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );

  const renderChannelSettings = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Channel Configuration" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Channel Name"
                value={channel?.name || ''}
                fullWidth
              />
              <TextField
                label="Channel Handle"
                value={channel?.handle || ''}
                fullWidth
              />
              <TextField
                label="Description"
                multiline
                rows={4}
                fullWidth
              />
              <FormControlLabel
                control={<Switch checked={channel?.monetized || false} />}
                label="Monetization Enabled"
              />
              <FormControlLabel
                control={<Switch checked={true} />}
                label="Auto-publish Videos"
              />
              <Button variant="contained">
                Save Changes
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Upload Defaults" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl fullWidth>
                <InputLabel>Default Category</InputLabel>
                <Select value="technology" label="Default Category">
                  <MenuItem value="technology">Science & Technology</MenuItem>
                  <MenuItem value="education">Education</MenuItem>
                  <MenuItem value="entertainment">Entertainment</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Default Tags"
                placeholder="tech, ai, innovation"
                fullWidth
              />
              <FormControl fullWidth>
                <InputLabel>Default Visibility</InputLabel>
                <Select value="public" label="Default Visibility">
                  <MenuItem value="public">Public</MenuItem>
                  <MenuItem value="unlisted">Unlisted</MenuItem>
                  <MenuItem value="private">Private</MenuItem>
                </Select>
              </FormControl>
              <FormControlLabel
                control={<Switch checked={true} />}
                label="Enable Comments"
              />
              <FormControlLabel
                control={<Switch checked={true} />}
                label="Enable Likes"
              />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Alert severity="info">
          Changes to channel settings may take up to 24 hours to fully propagate across YouTube's systems.
        </Alert>
      </Grid>
    </Grid>
  );

  if (loading) {
    return (
    <>
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    )}

  return (
    <Box>
      <Tabs
        value={tabValue}
        onChange={(e, newValue) => setTabValue(newValue)}
        variant={isMobile ? 'scrollable' : 'standard'}
        scrollButtons={isMobile ? 'auto' : false}
      >
        <Tab label="Overview" icon={<Dashboard />} iconPosition="start" />
        <Tab label="Videos" icon={<VideoLibrary />} iconPosition="start" />
        <Tab label="Analytics" icon={<BarChart />} iconPosition="start" />
        <Tab label="Settings" icon={<Settings />} iconPosition="start" />
      </Tabs>
      <TabPanel value={tabValue} index={0}>
        {renderChannelOverview()}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {renderVideoHistory()}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Typography>Analytics content coming soon...</Typography>
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        {renderChannelSettings()}
      </TabPanel>
    </Box>
  </>
  )};