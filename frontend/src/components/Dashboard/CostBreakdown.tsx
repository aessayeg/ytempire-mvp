import React from 'react'
import { Paper, Typography, Box, List, ListItem, ListItemText, LinearProgress } from '@mui/material'

interface CostCategory {
  name: string
  amount: number
  percentage: number
  color: string
}

const mockCostCategories: CostCategory[] = [
  { name: 'Script Generation', amount: 85, percentage: 35, color: '#8884d8' },
  { name: 'Voice Synthesis', amount: 60, percentage: 25, color: '#82ca9d' },
  { name: 'Thumbnail Creation', amount: 36, percentage: 15, color: '#ffc658' },
  { name: 'Video Processing', amount: 48, percentage: 20, color: '#ff7c7c' },
  { name: 'Infrastructure', amount: 12, percentage: 5, color: '#8dd1e1' },
]

export const CostBreakdown: React.FC = () => {
  const totalCost = mockCostCategories.reduce((sum, cat) => sum + cat.amount, 0)

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Monthly Cost Breakdown
      </Typography>
      <Typography variant="h4" sx={{ mb: 3 }}>
        ${totalCost}
      </Typography>
      <List>
        {mockCostCategories.map((category) => (
          <ListItem key={category.name} sx={{ px: 0 }}>
            <ListItemText
              primary={
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">{category.name}</Typography>
                  <Typography variant="body2" fontWeight="medium">
                    ${category.amount} ({category.percentage}%)
                  </Typography>
                </Box>
              }
              secondary={
                <LinearProgress
                  variant="determinate"
                  value={category.percentage}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: '#e0e0e0',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: category.color,
                    },
                  }}
                />
              }
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  )
}