import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  CircularProgress,
  MenuItem,
  Chip,
  Stack,
  Divider
} from '@mui/material';
import { RocketLaunch, CheckCircle, Star } from '@mui/icons-material';
import { api } from '../services/api';

interface BetaFormData {
  full_name: string;
  email: string;
  company: string;
  use_case: string;
  expected_volume: string;
  referral_source: string;
}

const BetaSignup: React.FC = () => {
  const [formData, setFormData] = useState<BetaFormData>({
    full_name: '',
    email: '',
    company: '',
    use_case: '',
    expected_volume: '10-50',
    referral_source: ''
  });
  
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);

  const volumeOptions = [
    '1-10',
    '10-50',
    '50-100',
    '100-500',
    '500+'
  ];

  const referralOptions = [
    'Google Search',
    'Reddit',
    'Twitter/X',
    'Friend/Colleague',
    'YouTube',
    'Other'
  ];

  const handleSubmit = async (_e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await api.post('/beta/signup', formData);
      setSuccess(true);
      setApiKey(response.data.api_key);
    } catch (err: unknown) {
      setError(err.response?.data?.detail || 'Failed to submit application');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (_e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  if (success) {
    return (
      <Container maxWidth="md" sx={{ mt: 8 }}>
        <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
          <CheckCircle color="success" sx={{ fontSize: 64, mb: 2 }} />
          <Typography variant="h4" gutterBottom>
            Welcome to YTEmpire Beta! 🎉
          </Typography>
          <Typography variant="body1" paragraph>
            Your application has been approved! Check your email for login credentials.
          </Typography>
          
          {apiKey && (
            <Paper sx={{ p: 2, bgcolor: 'grey.100', mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Your API Key (save this):
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                {apiKey}
              </Typography>
            </Paper>
          )}
          
          <Box sx={{ mt: 4 }}>
            <Button
              variant="contained"
              size="large"
              href="/login"
              sx={{ mr: 2 }}
            >
              Go to Login
            </Button>
            <Button
              variant="outlined"
              size="large"
              href="/docs"
            >
              View Documentation
            </Button>
          </Box>
        </Paper>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <RocketLaunch sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h3" gutterBottom>
            Join YTEmpire Beta
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Get early access to AI-powered YouTube automation
          </Typography>
        </Box>

        <Stack direction="row" spacing={1} justifyContent="center" sx={{ mb: 4 }}>
          <Chip icon={<Star />} label="$50 Free Credits" color="primary" />
          <Chip icon={<Star />} label="5x API Rate Limit" color="primary" />
          <Chip icon={<Star />} label="Priority Support" color="primary" />
        </Stack>

        <Divider sx={{ mb: 4 }} />

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'grid', gap: 3 }}>
            <TextField
              fullWidth
              label="Full Name"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              required
              disabled={loading}
            />

            <TextField
              fullWidth
              label="Email"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              required
              disabled={loading}
            />

            <TextField
              fullWidth
              label="Company (Optional)"
              name="company"
              value={formData.company}
              onChange={handleChange}
              disabled={loading}
            />

            <TextField
              fullWidth
              label="Use Case"
              name="use_case"
              value={formData.use_case}
              onChange={handleChange}
              multiline
              rows={3}
              required
              disabled={loading}
              helperText="Describe how you plan to use YTEmpire (min 10 characters)"
            />

            <TextField
              fullWidth
              select
              label="Expected Videos per Month"
              name="expected_volume"
              value={formData.expected_volume}
              onChange={handleChange}
              required
              disabled={loading}
            >
              {volumeOptions.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              fullWidth
              select
              label="How did you hear about us?"
              name="referral_source"
              value={formData.referral_source}
              onChange={handleChange}
              disabled={loading}
            >
              {referralOptions.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </TextField>

            <Button
              type="submit"
              variant="contained"
              size="large"
              disabled={loading}
              sx={{ mt: 2 }}
            >
              {loading ? (
                <>
                  <CircularProgress size={24} sx={{ mr: 1 }} />
                  Submitting...
                </>
              ) : (
                'Apply for Beta Access'
              )}
            </Button>
          </Box>
        </form>

        <Typography variant="body2" color="text.secondary" sx={{ mt: 3, textAlign: 'center' }}>
          By applying, you agree to our Terms of Service and Privacy Policy.
          Beta access is limited and applications are reviewed within 24 hours.
        </Typography>
      </Paper>
    </Container>
  );
};

export default BetaSignup;