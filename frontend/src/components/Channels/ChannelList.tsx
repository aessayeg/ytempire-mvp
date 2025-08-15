import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Grid,
  Chip,
  Avatar,
  IconButton,
  Menu,
  MenuItem,
  Skeleton,
  Alert,
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
              label="Channel Name"
              fullWidth
              value={editForm.name || ''}
              onChange={(e) => setEditForm({ ...editForm, name: e.target.value)});
}
            />
            <TextField
              label="Description"
              fullWidth
              multiline
              rows={3}
              value={editForm.description || ''}
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value)});
}
            />
            <FormControl fullWidth>
              <InputLabel>Upload Schedule</InputLabel>
              <Select
                value={editForm.upload_schedule || ''}
                label="Upload Schedule"
                onChange={(e: SelectChangeEvent) => setEditForm({ ...editForm, upload_schedule: e.target.value)});
}
              >
                <MenuItem value="daily">Daily</MenuItem>
                <MenuItem value="weekly">Weekly</MenuItem>
                <MenuItem value="biweekly">Bi-weekly</MenuItem>
                <MenuItem value="monthly">Monthly</MenuItem>
                <MenuItem value="custom">Custom</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleEditSubmit} variant="contained">
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>
    </>
  )};