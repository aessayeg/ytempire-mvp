import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField
            fullWidth
            label="Your Name"
            value={userProfile.name}
            onChange={(e) => setUserProfile({ ...userProfile, name: e.target.value)})}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Email"
            type="email"
            value={userProfile.email}
            onChange={(e) => setUserProfile({ ...userProfile, email: e.target.value)})}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Company (Optional)"
            value={userProfile.company}
            onChange={(e) => setUserProfile({ ...userProfile, company: e.target.value)})}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Your Role</InputLabel>
            <Select
              value={userProfile.role}
              onChange={(e) => setUserProfile({ ...userProfile, role: e.target.value)})}
              label="Your Role"
            >
              <MenuItem value="content-creator">Content Creator</MenuItem>
              <MenuItem value="marketer">Digital Marketer</MenuItem>
              <MenuItem value="business-owner">Business Owner</MenuItem>
              <MenuItem value="agency">Agency</MenuItem>
              <MenuItem value="other">Other</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            What are your main goals? (Select all that, apply)
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {['Grow subscribers', 'Increase revenue', 'Save time', 'Improve quality', 'Scale content'].map((goal) => (_<Chip
                key={goal}
                label={goal}
                onClick={() => {
                  const goals = userProfile.goals.includes(goal)
                    ? userProfile.goals.filter(g => g !== goal)
                    : [...userProfile.goals, goal];
                  setUserProfile({ ...userProfile, goals })}}
                color={userProfile.goals.includes(goal) ? 'primary' : 'default'}
                icon={userProfile.goals.includes(goal) ? <Check /> : undefined}
              />
            ))}
          </Box>
        </Grid>
        
        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>How often do you plan to upload?</InputLabel>
            <Select
              value={userProfile.uploadFrequency}
              onChange={(e) => setUserProfile({ ...userProfile, uploadFrequency: e.target.value)})}
              label="How often do you plan to upload?"
            >
              <MenuItem value="daily">Daily</MenuItem>
              <MenuItem value="few-times-week">Few times a week</MenuItem>
              <MenuItem value="weekly">Weekly</MenuItem>
              <MenuItem value="monthly">Monthly</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
    </Box>
  );

  const renderChannelsStep = () => (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Connect Your YouTube Channels
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        We'll need permission to manage your YouTube channels
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          YTEmpire uses OAuth 2.0 for secure authentication. We never store your YouTube password.
        </Typography>
      </Alert>
      
      {channels.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <YouTube sx={{ fontSize: 60, color: 'error.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            No channels connected yet
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Connect your YouTube channel to start creating and publishing videos automatically
          </Typography>
          
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Authorization, Steps:
            </Typography>
            <List sx={{ maxWidth: 400, mx: 'auto', textAlign: 'left' }}>
              <ListItem>
                <ListItemIcon>
                  <FiberManualRecord sx={{ fontSize: 8 }} />
                </ListItemIcon>
                <ListItemText primary="Click the connect button below" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <FiberManualRecord sx={{ fontSize: 8 }} />
                </ListItemIcon>
                <ListItemText primary="Sign in to your YouTube account" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <FiberManualRecord sx={{ fontSize: 8 }} />
                </ListItemIcon>
                <ListItemText primary="Grant YTEmpire permission to manage videos" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <FiberManualRecord sx={{ fontSize: 8 }} />
                </ListItemIcon>
                <ListItemText primary="You're all set!" />
              </ListItem>
            </List>
          </Box>
          
          <Button
            variant="contained"
            color="error"
            size="large"
            startIcon={<YouTube />}
            onClick={connectChannel}
            disabled={loading}
            sx={{ mt: 2 }}
          >
            {loading ? 'Connecting...' : 'Connect YouTube Channel'}
          </Button>
        </Paper>
      ) : (
        <Box>
          {channels.map((channel) => (
            <Paper key={channel.id} sx={{ p: 2, mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Avatar sx={{ bgcolor: 'error.main' }}>
                    <YouTube />
                  </Avatar>
                  <Box>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {channel.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {channel.handle} • {channel.subscribers.toLocaleString()} subscribers
                    </Typography>
                  </Box>
                </Box>
                <Chip
                  label="Connected"
                  color="success"
                  icon={<CheckCircle />}
                />
              </Box>
            </Paper>
          ))}
          <Button
            variant="outlined"
            startIcon={<Add />}
            sx={{ mt: 2 }}
          >
            Add Another Channel
          </Button>
        </Box>
      )}
    </Box>
  );

  const renderPreferencesStep = () => (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Set Your Preferences
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Customize how YTEmpire works for you
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Choose Your Plan
          </Typography>
          <Grid container spacing={2}>
            {plans.map((plan) => (
              <Grid item xs={12} sm={4} key={plan.id}>
                <Paper
                  sx={ {
                    p: 2,
                    border: 2,
                    borderColor: selectedPlan === plan.id ? 'primary.main' : 'divider',
                    cursor: 'pointer',
                    position: 'relative' }}
                  onClick={() => setSelectedPlan(plan.id)}
                >
                  {plan.recommended && (
                    <Chip
                      label="RECOMMENDED"
                      color="primary"
                      size="small"
                      sx={{ position: 'absolute', top: -10, right: 10 }}
                    />
                  )}
                  <Typography variant="h6" fontWeight="bold">
                    {plan.name}
                  </Typography>
                  <Typography variant="h4" color="primary.main" sx={{ my: 1 }}>
                    {plan.price}
                  </Typography>
                  <List dense>
                    {plan.features.map((feature, index) => (
                      <ListItem key={index} sx={{ px: 0 }}>
                        <ListItemIcon sx={{ minWidth: 28 }}>
                          <CheckCircle fontSize="small" color="success" />
                        </ListItemIcon>
                        <ListItemText primary={feature} />
                      </ListItem>
                    ))}
                  </List>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Automation Settings
          </Typography>
          <List>
            <ListItem>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={preferences.autoUpload}
                    onChange={(e) => setPreferences({ ...preferences, autoUpload: e.target.checked })}
                  />
                }
                label="Auto-upload videos when ready"
              />
            </ListItem>
            <ListItem>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={preferences.emailNotifications}
                    onChange={(e) => setPreferences({ ...preferences, emailNotifications: e.target.checked })}
                  />
                }
                label="Email notifications for important updates"
              />
            </ListItem>
            <ListItem>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={preferences.aiOptimization}
                    onChange={(e) => setPreferences({ ...preferences, aiOptimization: e.target.checked })}
                  />
                }
                label="AI optimization for titles and descriptions"
              />
            </ListItem>
            <ListItem>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={preferences.qualityCheck}
                    onChange={(e) => setPreferences({ ...preferences, qualityCheck: e.target.checked })}
                  />
                }
                label="Quality check before publishing"
              />
            </ListItem>
          </List>
        </Grid>
      </Grid>
    </Box>
  );

  const renderTutorialStep = () => (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Quick Tutorial
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Learn the basics in just 2 minutes
      </Typography>
      
      <Paper sx={{ p: 3, bgcolor: 'primary.light', color: 'primary.contrastText', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <PlayCircle sx={{ fontSize: 40 }} />
          <Box>
            <Typography variant="h6">
              Interactive Walkthrough
            </Typography>
            <Typography variant="body2">
              We'll guide you through the main features step by step
            </Typography>
          </Box>
        </Box>
      </Paper>
      
      <Grid container spacing={2}>
        {tutorialSteps.map((step, index) => (
          <Grid item xs={12} key={step.id}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  {index + 1}
                </Avatar>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle1" fontWeight="bold">
                    {step.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {step.description}
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
      
      <Box sx={{ mt: 3, textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          startIcon={<PlayCircle />}
          onClick={() => setShowTutorial(true)}
        >
          Start Interactive Tutorial
        </Button>
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          You can always access this tutorial later from the Help menu
        </Typography>
      </Box>
    </Box>
  );

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return null; // Welcome screen is separate
      case 1:
        return renderProfileStep();
      case 2:
        return renderChannelsStep();
      case 3:
        return renderPreferencesStep();
      case 4:
        return renderTutorialStep(),
  default:
        return null}
  };

  if (showWelcome) {
    return renderWelcomeScreen()}

  return (
    <>
      <Box sx={{ maxWidth: 900, mx: 'auto', p: 3 }}>
      {showCelebration && (
        <Confetti
          width={window.innerWidth}
          height={window.innerHeight}
          recycle={false}
          numberOfPieces={200}
        />
      )}
      <Box sx={{ mb: 4 }}>
        <LinearProgress
          variant="determinate"
          value={(completedSteps.length / steps.length) * 100}
          sx={{ height: 8, borderRadius: 1 }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Step {activeStep + 1} of {steps.length}
        </Typography>
      </Box>
      <Stepper activeStep={activeStep} orientation={isMobile ? 'vertical' : 'horizontal'}>
        {steps.map((step, index) => (
          <Step key={step.id} completed={completedSteps.includes(index)}>
            <StepLabel
              icon={step.icon}
              optional={
                step.skippable && (
                  <Typography variant="caption">Optional</Typography>
                )}
            >
              {step.title}
            </StepLabel>
          </Step>
        ))}
      </Stepper>
      
      <Box sx={{ mt: 4 }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderStepContent(activeStep)}
          </motion.div>
        </AnimatePresence>
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
          startIcon={<ArrowBack />}
        >
          Back
        </Button>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          {steps[activeStep].skippable && (
            <Button onClick={handleSkip}>
              Skip
            </Button>
          )}
          <Button
            variant="contained"
            onClick={handleNext}
            endIcon={activeStep === steps.length - 1 ? <Celebration /> </>: <ArrowForward />}
          >
            {activeStep === steps.length - 1 ? 'Complete Setup' : 'Continue'}
          </Button>
        </Box>
      </Box>
      
      {/* Loading Backdrop */}
      <Backdrop open={loading} sx={{ zIndex: theme.zIndex.drawer + 1 }}>
        <CircularProgress color="inherit" />
      </Backdrop>
    </Box>
  </>
  )};