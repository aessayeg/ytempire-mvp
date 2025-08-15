import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
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
  TextField
                fullWidth
                label="Experiment Name"
                value={newExperiment.name}
                onChange={(e) => setNewExperiment({ ...newExperiment, name: e.target.value)})}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Description"
                value={newExperiment.description}
                onChange={(e) => setNewExperiment({ ...newExperiment, description: e.target.value)})}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Hypothesis"
                value={newExperiment.hypothesis}
                onChange={(e) => setNewExperiment({ ...newExperiment, hypothesis: e.target.value)})}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Target Metric</InputLabel>
                <Select
                  value={newExperiment.target_metric}
                  onChange={(e) => setNewExperiment({ ...newExperiment, target_metric: e.target.value)})}
                >
                  <MenuItem value="conversion_rate">Conversion Rate</MenuItem>
                  <MenuItem value="revenue">Revenue</MenuItem>
                  <MenuItem value="engagement">Engagement</MenuItem>
                  <MenuItem value="retention">Retention</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                type="number"
                label="Duration (days)"
                value={newExperiment.duration_days}
                onChange={(e) => setNewExperiment({ ...newExperiment, duration_days: parseInt(e.target.value) })}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                type="number"
                label="Min Sample Size"
                value={newExperiment.min_sample_size}
                onChange={(e) => setNewExperiment({ ...newExperiment, min_sample_size: parseInt(e.target.value) })}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={createExperiment} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )};