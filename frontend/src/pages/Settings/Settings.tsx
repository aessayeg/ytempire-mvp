/**
 * Settings Screen Component
 * Comprehensive settings management for user preferences and system configuration
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField
                          fullWidth
                          label="Display Name"
                          value={settings.profile.displayName}
                          onChange={(e) => handleSettingChange('profile', 'displayName', e.target.value)}
                        />
                      </Grid>
                      <Grid item xs={12}>
                        <TextField
                          fullWidth
                          label="Email Address"
                          value={settings.profile.email}
                          onChange={(e) => handleSettingChange('profile', 'email', e.target.value)}
                          type="email"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <InputLabel>Timezone</InputLabel>
                          <Select
                            value={settings.profile.timezone}
                            onChange={(e) => handleSettingChange('profile', 'timezone', e.target.value)}
                            label="Timezone"
                          >
                            <MenuItem value="America/New_York">Eastern Time</MenuItem>
                            <MenuItem value="America/Chicago">Central Time</MenuItem>
                            <MenuItem value="America/Denver">Mountain Time</MenuItem>
                            <MenuItem value="America/Los_Angeles">Pacific Time</MenuItem>
                            <MenuItem value="Europe/London">GMT</MenuItem>
                            <MenuItem value="Europe/Paris">CET</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <InputLabel>Language</InputLabel>
                          <Select
                            value={settings.profile.language}
                            onChange={(e) => handleSettingChange('profile', 'language', e.target.value)}
                            label="Language"
                          >
                            <MenuItem value="en">English</MenuItem>
                            <MenuItem value="es">Spanish</MenuItem>
                            <MenuItem value="fr">French</MenuItem>
                            <MenuItem value="de">German</MenuItem>
                            <MenuItem value="ja">Japanese</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                    </Grid>
                  </Grid>
                </Grid>
              </CardContent>
            </TabPanel>

            {/* Notification Settings */}
            <TabPanel value={activeTab} index={1}>
              <CardHeader title="Notification Preferences" />
              <CardContent>
                <List>
                  {Object.entries(settings.notifications).map(([key, value]) => (
                    <ListItem key={key}>
                      <ListItemIcon>
                        <Notifications />
                      </ListItemIcon>
                      <ListItemText
                        primary={key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                        secondary={`Receive notifications for ${key.toLowerCase().replace(/([A-Z])/g, ' $1')}`}
                      />
                      <ListItemSecondaryAction>
                        <Switch
                          checked={value as boolean}
                          onChange={(e) => handleSettingChange('notifications', key, e.target.checked)}
                        />
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </TabPanel>

            {/* Appearance Settings */}
            <TabPanel value={activeTab} index={2}>
              <CardHeader title="Appearance" />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <FormControl component="fieldset">
                      <FormLabel component="legend">Theme</FormLabel>
                      <RadioGroup
                        value={settings.appearance.theme}
                        onChange={(e) => handleSettingChange('appearance', 'theme', e.target.value)}
                      >
                        <FormControlLabel value="light" control={<Radio />} label="Light" />
                        <FormControlLabel value="dark" control={<Radio />} label="Dark" />
                        <FormControlLabel value="auto" control={<Radio />} label="Auto (System)" />
                      </RadioGroup>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Interface Density</InputLabel>
                      <Select
                        value={settings.appearance.density}
                        onChange={(e) => handleSettingChange('appearance', 'density', e.target.value)}
                        label="Interface Density"
                      >
                        <MenuItem value="compact">Compact</MenuItem>
                        <MenuItem value="comfortable">Comfortable</MenuItem>
                        <MenuItem value="spacious">Spacious</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Sidebar Style</InputLabel>
                      <Select
                        value={settings.appearance.sidebar}
                        onChange={(e) => handleSettingChange('appearance', 'sidebar', e.target.value)}
                        label="Sidebar Style"
                      >
                        <MenuItem value="collapsed">Collapsed</MenuItem>
                        <MenuItem value="expanded">Expanded</MenuItem>
                        <MenuItem value="auto">Auto-hide</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>
              </CardContent>
            </TabPanel>

            {/* Privacy Settings */}
            <TabPanel value={activeTab} index={3}>
              <CardHeader title="Privacy & Data" />
              <CardContent>
                <List>
                  {Object.entries(settings.privacy).map(([key, value]) => (
                    <ListItem key={key}>
                      <ListItemIcon>
                        <Security />
                      </ListItemIcon>
                      <ListItemText
                        primary={key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())
                        secondary={`Allow ${key.toLowerCase().replace(/([A-Z])/g, ' $1')} to improve service quality`}
                      />
                      <ListItemSecondaryAction>
                        <Switch
                          checked={value as boolean}
                          onChange={(e) => handleSettingChange('privacy', key, e.target.checked)}
                        />
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
                
                <Divider sx={{ my: 3 }} />
                
                <Alert severity="info" icon={<Info />}>
                  We respect your privacy. Your data is encrypted and never shared with third parties 
                  without your explicit consent.
                </Alert>
                
                <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                  <Button variant="outlined" startIcon={<Download />}>
                    Download My Data
                  </Button>
                  <Button variant="outlined" color="error" startIcon={<Delete />}>
                    Delete Account
                  </Button>
                </Box>
              </CardContent>
            </TabPanel>

            {/* API Keys */}
            <TabPanel value={activeTab} index={4}>
              <CardHeader 
                title="API Keys" 
                action={
                  <Button variant="contained" startIcon={<Add />} onClick={handleAddApiKey}>
                    Add API Key
                  </Button>
                }
              />
              <CardContent>
                <List>
                  {mockAPIKeys.map((api) => (
                    <ListItem key={api.id} sx={{ border: 1, borderColor: 'divider', mb: 1, borderRadius: 1 }}>
                      <ListItemIcon>
                        <Api color={api.status === 'active' ? 'primary' : 'disabled'} />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle1">{api.name}</Typography>
                            <Chip 
                              label={api.status} 
                              size="small" 
                              color={getStatusColor(api.status) as any}
                              variant="outlined"
                            />
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              {api.service} • Last used: {api.lastUsed}
                            </Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={api.usage} 
                              sx={{ mt: 1, height: 4, borderRadius: 2 }}
                              color={api.usage > 80 ? 'error' : 'primary'}
                            />
                            <Typography variant="caption" color="text.secondary">
                              Usage: {api.usage}%
                            </Typography>
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <IconButton onClick={() => handleEditApiKey(api)}>
                          <Edit />
                        </IconButton>
                        <IconButton color="error" onClick={() => setShowDeleteDialog(true)}>
                          <Delete />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </TabPanel>

            {/* Automation Settings */}
            <TabPanel value={activeTab} index={5}>
              <CardHeader title="Automation Settings" />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <List>
                      <ListItem>
                        <ListItemText primary="Auto-publish videos" secondary="Automatically publish videos after processing" />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.automation.autoPublish}
                            onChange={(e) => handleSettingChange('automation', 'autoPublish', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Smart scheduling" secondary="Use AI to optimize publish times" />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.automation.smartScheduling}
                            onChange={(e) => handleSettingChange('automation', 'smartScheduling', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Auto-generate thumbnails" secondary="Create thumbnails automatically" />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.automation.autoThumbnails}
                            onChange={(e) => handleSettingChange('automation', 'autoThumbnails', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                    </List>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>Quality Controls</Typography>
                    <Box sx={{ px: 2 }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Quality Threshold: {settings.automation.qualityThreshold}%
                      </Typography>
                      <Slider
                        value={settings.automation.qualityThreshold}
                        onChange={(e, value) => handleSettingChange('automation', 'qualityThreshold', value)}
                        min={0}
                        max={100}
                        marks={[ { value: 0, label: '0%' },
                          { value: 50, label: '50%' },
                          { value: 100, label: '100%' }  ]
                      />
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Monthly Cost Limit"
                      type="number"
                      value={settings.automation.costLimit}
                      onChange={(e) => handleSettingChange('automation', 'costLimit', parseInt(e.target.value))}
                      InputProps={ {
                        startAdornment: '$',
                        endAdornment: 'USD' }}
                      helperText="Videos will be paused when this limit is reached"
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </TabPanel>

            {/* AI Settings */}
            <TabPanel value={activeTab} index={6}>
              <CardHeader title="AI Configuration" />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Preferred AI Model</InputLabel>
                      <Select
                        value={settings.ai.preferredModel}
                        onChange={(e) => handleSettingChange('ai', 'preferredModel', e.target.value)}
                        label="Preferred AI Model"
                      >
                        <MenuItem value="gpt-3.5-turbo">GPT-3.5 Turbo</MenuItem>
                        <MenuItem value="gpt-4">GPT-4</MenuItem>
                        <MenuItem value="claude-3">Claude 3</MenuItem>
                        <MenuItem value="gemini-pro">Gemini Pro</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Voice Style</InputLabel>
                      <Select
                        value={settings.ai.voiceStyle}
                        onChange={(e) => handleSettingChange('ai', 'voiceStyle', e.target.value)}
                        label="Voice Style"
                      >
                        <MenuItem value="professional">Professional</MenuItem>
                        <MenuItem value="casual">Casual</MenuItem>
                        <MenuItem value="energetic">Energetic</MenuItem>
                        <MenuItem value="calm">Calm</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Content Creativity: {settings.ai.creativity}%
                    </Typography>
                    <Slider
                      value={settings.ai.creativity}
                      onChange={(e, value) => handleSettingChange('ai', 'creativity', value)}
                      min={0}
                      max={100}
                      marks={[ { value: 0, label: 'Conservative' },
                        { value: 50, label: 'Balanced' },
                        { value: 100, label: 'Creative' }  ]
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <List>
                      <ListItem>
                        <ListItemText primary="Auto-generate descriptions" />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.ai.generateDescription}
                            onChange={(e) => handleSettingChange('ai', 'generateDescription', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Auto-generate tags" />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.ai.generateTags}
                            onChange={(e) => handleSettingChange('ai', 'generateTags', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>
              </CardContent>
            </TabPanel>

            {/* Billing Settings */}
            <TabPanel value={activeTab} index={7}>
              <CardHeader title="Subscription & Billing" />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Current Plan: {mockSubscriptionPlan.name}
                        </Typography>
                        <Typography variant="h4" color="primary" gutterBottom>
                          {mockSubscriptionPlan.price}
                        </Typography>
                        <List dense>
                          {mockSubscriptionPlan.features.map((feature, index) => (
                            <ListItem key={index}>
                              <ListItemIcon>
                                <CheckCircle color="success" fontSize="small" />
                              </ListItemIcon>
                              <ListItemText primary={feature} />
                            </ListItem>
                          ))}
                        </List>
                        <Box sx={{ mt: 2 }}>
                          <Button variant="outlined" fullWidth>
                            Change Plan
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Usage This Month</Typography>
                    {Object.entries(mockSubscriptionPlan.usage).map(([key, usage]) => (
                      <Box key={key} sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2">
                            {key.charAt(0).toUpperCase() + key.slice(1)}
                          </Typography>
                          <Typography variant="body2">
                            {usage.current} / {usage.limit}
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={(usage.current / usage.limit) * 100}
                          color={usage.current / usage.limit > 0.8 ? 'warning' : 'primary'}
                        />
                      </Box>
                    ))}
                    <Divider sx={{ my: 2 }} />
                    
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      <Button variant="outlined" startIcon={<CreditCard />} size="small">
                        Payment Methods
                      </Button>
                      <Button variant="outlined" startIcon={<Receipt />} size="small">
                        Billing History
                      </Button>
                      <Button variant="outlined" startIcon={<Download />} size="small">
                        Download Invoice
                      </Button>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>

      {/* API Key Dialog */}
      <Dialog open={showApiDialog} onClose={() => setShowApiDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selectedApi ? 'Edit API Key' : 'Add New API Key'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField fullWidth label="Service Name" defaultValue={selectedApi?.name || ''} />
            </Grid>
            <Grid item xs={12}>
              <TextField fullWidth label="API Key" type="password" defaultValue="••••••••••••••••" />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Service Type</InputLabel>
                <Select defaultValue={selectedApi?.service || ''} label="Service Type">
                  <MenuItem value="GPT-4">OpenAI GPT-4</MenuItem>
                  <MenuItem value="Voice Synthesis">ElevenLabs</MenuItem>
                  <MenuItem value="Channel Management">YouTube API</MenuItem>
                  <MenuItem value="Content Generation">Claude API</MenuItem>
                  <MenuItem value="Image Generation">DALL-E</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowApiDialog(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setShowApiDialog(false)}>
            {selectedApi ? 'Update' : 'Add'} API Key
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onClose={() => setShowDeleteDialog(false)}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this API key? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDeleteDialog(false)}>Cancel</Button>
          <Button color="error" variant="contained" onClick={() => setShowDeleteDialog(false)}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )};

export default Settings;