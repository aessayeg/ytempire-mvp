/**
 * Mobile Optimized Dashboard Component
 * Demonstrates extended mobile responsive design features
 */
import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  IconButton,
  Avatar,
  List,
  ListItem,
  Drawer,
  AppBar,
  Toolbar,
  Badge,
  Fab,
  Skeleton
} from '@mui/material';
import {
  Menu as MenuIcon,
  Search as SearchIcon,
  Close as CloseIcon,
  Add as AddIcon,
  Notifications as NotificationsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

interface MobileDataCard {
  title: string;
  value: string;
  change: number;
  changeType: 'positive' | 'negative' | 'neutral';
  icon?: React.ReactNode;
}

interface MobileListItem {
  id: string;
  title: string;
  subtitle: string;
  avatar: string;
  action?: React.ReactNode;
}

const MobileOptimizedDashboard: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showMobileModal, setShowMobileModal] = useState(false);

  // Sample data
  const dataCards: MobileDataCard[] = [
    {
      title: 'Total Revenue',
      value: '$12,450',
      change: 15.3,
      changeType: 'positive',
      icon: <TrendingUpIcon />
    },
    {
      title: 'Video Views',
      value: '2.4M',
      change: -5.2,
      changeType: 'negative',
      icon: <TrendingDownIcon />
    },
    {
      title: 'Subscribers',
      value: '48.2K',
      change: 8.7,
      changeType: 'positive'
    },
    {
      title: 'Avg. Cost/Video',
      value: '$2.45',
      change: -12.1,
      changeType: 'positive'
    }
  ];

  const listItems: MobileListItem[] = [
    {
      id: '1',
      title: 'Gaming Channel Update',
      subtitle: 'Video published 2 hours ago',
      avatar: '/avatars/gaming.jpg'
    },
    {
      id: '2', 
      title: 'Tech Review Generated',
      subtitle: 'Processing completed successfully',
      avatar: '/avatars/tech.jpg'
    },
    {
      id: '3',
      title: 'Cooking Tutorial',
      subtitle: 'Scheduled for tomorrow 9 AM',
      avatar: '/avatars/cooking.jpg'
    }
  ];

  useEffect(() => {
    // Simulate loading
    setTimeout(() => setIsLoading(false), 2000);
  }, []);

  const handlePullToRefresh = async () => {
    setIsRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setIsRefreshing(false);
    }, 1500);
  };

  const renderDataCard = (card: MobileDataCard, index: number) => (
    <div key={index} className="data-card-mobile">
      <div className="data-card-mobile-header">
        <span className="data-card-mobile-title">{card.title}</span>
        {card.icon}
      </div>
      <div className="data-card-mobile-value">{card.value}</div>
      <div className={`data-card-mobile-change ${card.changeType}`}>
        {card.changeType === 'positive' ? '↗' : '↘'} {Math.abs(card.change)}%
      </div>
    </div>
  );

  const renderListItem = (item: MobileListItem) => (
    <div key={item.id} className="mobile-list-item">
      <div className="mobile-list-item-avatar">
        <Avatar sx={{ width: 40, height: 40 }}>{item.title[0]}</Avatar>
      </div>
      <div className="mobile-list-item-content">
        <div className="mobile-list-item-title">{item.title}</div>
        <div className="mobile-list-item-subtitle">{item.subtitle}</div>
      </div>
      <div className="mobile-list-item-action">
        <IconButton size="small">
          <MenuIcon />
        </IconButton>
      </div>
    </div>
  );

  const renderSkeleton = () => (
    <Box sx={{ p: 2 }}>
      {[1, 2, 3, 4].map((index) => (
        <div key={index} className="skeleton skeleton-card" />
      ))}
    </Box>
  );

  return (
    <Box className="mobile-dashboard-container">
      {/* Mobile App Bar */}
      <AppBar position="sticky" sx={{ display: { md: 'none' } }}>
        <Toolbar>
          <IconButton edge="start" color="inherit">
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            YTEmpire
          </Typography>
          <IconButton color="inherit">
            <Badge badgeContent={4} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Pull to Refresh Container */}
      <div className={`mobile-pull-refresh ${isRefreshing ? 'pulling' : ''}`}>
        <div className="mobile-pull-indicator">
          <RefreshIcon />
        </div>

        {/* Search Bar */}
        <Box sx={{ p: 2, display: { md: 'none' } }}>
          <div className="mobile-search-container">
            <SearchIcon className="mobile-search-icon" />
            <input
              className="mobile-search-input"
              placeholder="Search channels, videos..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                className="mobile-search-clear"
                onClick={() => setSearchQuery('')}
              >
                <CloseIcon />
              </button>
            )}
          </div>
        </Box>

        {/* Main Content */}
        <Box sx={{ 
          pb: { xs: 8, md: 0 }, // Bottom padding for mobile nav
          px: 2
        }}>
          {isLoading ? (
            renderSkeleton()
          ) : (
            <>
              {/* Data Cards Grid */}
              <Box sx={{ 
                display: 'grid',
                gridTemplateColumns: { 
                  xs: '1fr',
                  sm: '1fr 1fr',
                  lg: '1fr 1fr 1fr 1fr'
                },
                gap: 2,
                mb: 3
              }}>
                {dataCards.map(renderDataCard)}
              </Box>

              {/* Mobile List */}
              <div className="mobile-list">
                {listItems.map(renderListItem)}
              </div>

              {/* Chart Container with Horizontal Scroll */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Performance Overview
                </Typography>
                <div className="mobile-chart">
                  <div className="mobile-chart-scroll">
                    <Box sx={{ 
                      height: 200,
                      background: 'linear-gradient(45deg, #f3f4f6, #e5e7eb)',
                      borderRadius: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      minWidth: 400
                    }}>
                      <Typography color="textSecondary">
                        Chart Component Placeholder
                      </Typography>
                    </Box>
                  </div>
                </div>
              </Box>

              {/* Form Example */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Quick Actions
                </Typography>
                <form className="form-mobile-stack">
                  <div className="form-mobile-group">
                    <label>Video Title</label>
                    <input
                      type="text"
                      placeholder="Enter video title..."
                      style={{
                        padding: '12px',
                        border: '1px solid #d1d5db',
                        borderRadius: '8px',
                        fontSize: '16px'
                      }}
                    />
                  </div>
                  <div className="form-mobile-row two-cols">
                    <div className="form-mobile-group">
                      <label>Category</label>
                      <select
                        style={{
                          padding: '12px',
                          border: '1px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '16px'
                        }}
                      >
                        <option>Gaming</option>
                        <option>Tech</option>
                        <option>Cooking</option>
                      </select>
                    </div>
                    <div className="form-mobile-group">
                      <label>Priority</label>
                      <select
                        style={{
                          padding: '12px',
                          border: '1px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '16px'
                        }}
                      >
                        <option>High</option>
                        <option>Medium</option>
                        <option>Low</option>
                      </select>
                    </div>
                  </div>
                  <button
                    type="submit"
                    className="btn-responsive"
                    style={{
                      background: '#3b82f6',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '12px 24px',
                      fontSize: '16px',
                      fontWeight: 600
                    }}
                  >
                    Generate Video
                  </button>
                </form>
              </Box>
            </>
          )}
        </Box>
      </div>

      {/* Mobile Bottom Navigation */}
      <Box sx={{ display: { md: 'none' } }}>
        <div className="mobile-nav-tabs">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: '📊' },
            { id: 'videos', label: 'Videos', icon: '🎥' },
            { id: 'analytics', label: 'Analytics', icon: '📈' },
            { id: 'settings', label: 'Settings', icon: '⚙️' }
          ].map((tab) => (
            <a
              key={tab.id}
              href="#"
              className={`mobile-nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={(e) => {
                e.preventDefault();
                setActiveTab(tab.id);
              }}
            >
              <span className="mobile-nav-tab-icon">{tab.icon}</span>
              <span className="mobile-nav-tab-label">{tab.label}</span>
            </a>
          ))}
        </div>
      </Box>

      {/* Floating Action Button */}
      <Fab
        color="primary"
        sx={{
          position: 'fixed',
          bottom: { xs: 80, md: 20 },
          right: 20,
          display: { md: 'none' }
        }}
        onClick={() => setShowMobileModal(true)}
      >
        <AddIcon />
      </Fab>

      {/* Full Screen Mobile Modal */}
      <div className={`mobile-modal-fullscreen ${showMobileModal ? 'active' : ''}`}>
        <div className="mobile-modal-header">
          <button
            className="mobile-modal-close"
            onClick={() => setShowMobileModal(false)}
          >
            <CloseIcon />
          </button>
          <div className="mobile-modal-title">New Video</div>
          <button
            style={{
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 16px',
              fontSize: '14px',
              fontWeight: 600
            }}
          >
            Save
          </button>
        </div>
        <div className="mobile-modal-body">
          <Typography paragraph>
            This is a full-screen mobile modal optimized for mobile devices.
            It slides up from the bottom and provides a native app-like experience.
          </Typography>
          
          <form className="form-mobile-stack">
            <div className="form-mobile-group">
              <label>Title</label>
              <input
                type="text"
                placeholder="Video title..."
                style={{
                  padding: '12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '16px'
                }}
              />
            </div>
            
            <div className="form-mobile-group">
              <label>Description</label>
              <textarea
                placeholder="Video description..."
                rows={4}
                style={{
                  padding: '12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '16px',
                  resize: 'vertical'
                }}
              />
            </div>

            <div className="form-mobile-row two-cols">
              <div className="form-mobile-group">
                <label>Duration</label>
                <select
                  style={{
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '16px'
                  }}
                >
                  <option>Short (< 1 min)</option>
                  <option>Medium (1-5 min)</option>
                  <option>Long (> 5 min)</option>
                </select>
              </div>
              
              <div className="form-mobile-group">
                <label>Quality</label>
                <select
                  style={{
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '16px'
                  }}
                >
                  <option>High</option>
                  <option>Medium</option>
                  <option>Auto</option>
                </select>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* Modal Overlay */}
      {showMobileModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.3)',
            zIndex: 9998
          }}
          onClick={() => setShowMobileModal(false)}
        />
      )}
    </Box>
  );
};

export default MobileOptimizedDashboard;