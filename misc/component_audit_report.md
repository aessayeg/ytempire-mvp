# Frontend Component Usage Audit Report
==================================================
Total Components Analyzed: 83
Orphaned Components: 43
Used Components: 40

## ORPHANED COMPONENTS (Not imported/used anywhere)
------------------------------------------------------------

### components\Accessibility/
- **AccessibleButton** (components\Accessibility\AccessibleButton.tsx)
  - Exports: AccessibleButton
- **FocusTrap** (components\Accessibility\FocusTrap.tsx)
  - Exports: FocusTrap, useFocusManagement

### components\Auth/
- **EmailVerification** (components\Auth\EmailVerification.tsx)
  - Exports: EmailVerification
- **ForgotPasswordForm** (components\Auth\ForgotPasswordForm.tsx)
  - Exports: ForgotPasswordForm
- **LoginForm** (components\Auth\LoginForm.tsx)
  - Exports: LoginForm
- **RegisterForm** (components\Auth\RegisterForm.tsx)
  - Exports: RegisterForm

### components\Channels/
- **ChannelDashboard** (components\Channels\ChannelDashboard.tsx)
  - Exports: ChannelDashboard
- **ChannelHealthDashboard** (components\Channels\ChannelHealthDashboard.tsx)
  - Exports: ChannelHealthDashboard
- **ChannelList** (components\Channels\ChannelList.tsx)
  - Exports: ChannelList
- **ChannelTemplates** (components\Channels\ChannelTemplates.tsx)
  - Exports: ChannelTemplates

### components\Charts/
- **AdvancedCharts** (components\Charts\AdvancedCharts.tsx)
  - Exports: BubbleChart, HeatmapChart, FunnelVisualization, RadarVisualization, TreemapVisualization, CohortChart
- **ChannelPerformanceCharts** (components\Charts\ChannelPerformanceCharts.tsx)
  - Exports: ChannelPerformanceCharts

### components\Common/
- **ErrorMessage** (components\Common\ErrorMessage.tsx)
  - Exports: ErrorMessage

### components\CostTracking/
- **CostVisualization** (components\CostTracking\CostVisualization.tsx)
  - Exports: CostVisualization

### components\Dashboard/
- **CustomizableWidgets** (components\Dashboard\CustomizableWidgets.tsx)
  - Exports: CustomizableWidgets
- **EnhancedMetricsDashboard** (components\Dashboard\EnhancedMetricsDashboard.tsx)
  - Exports: EnhancedMetricsDashboard
- **MainDashboard** (components\Dashboard\MainDashboard.tsx)
  - Exports: MainDashboard
- **MetricsDashboard** (components\Dashboard\MetricsDashboard.tsx)
  - Exports: MetricsDashboard

### components\ErrorBoundary/
- **index** (components\ErrorBoundary\index.tsx)
  - Exports: ErrorBoundary, withErrorBoundary , RouteErrorBoundary, AsyncBoundary , useErrorHandler , ErrorFallback, MinimalErrorFallback 
- **useErrorHandler** (components\ErrorBoundary\useErrorHandler.tsx)
  - Exports: useErrorHandler, withErrorHandling

### components\Loading/
- **index** (components\Loading\index.tsx)
  - Exports: LoadingButton,
  LoadingButtonGroup,
, LoadingOverlay,
  InlineLoader,
  ImageLoadingPlaceholder,
, LoadingSkeleton,
  DashboardSkeleton,
  VideoQueueSkeleton,
  ChannelListSkeleton,

- **LoadingButton** (components\Loading\LoadingButton.tsx)
  - Exports: LoadingButtonGroup, LoadingButton
- **LoadingOverlay** (components\Loading\LoadingOverlay.tsx)
  - Exports: InlineLoader, ImageLoadingPlaceholder, LoadingOverlay

### components\Mobile/
- **MobileOptimizedDashboard** (components\Mobile\MobileOptimizedDashboard.tsx)
  - Exports: MobileOptimizedDashboard
- **MobileResponsiveSystem** (components\Mobile\MobileResponsiveSystem.tsx)
  - Exports: MobileResponsiveSystem

### components\Monitoring/
- **CostTrackingDashboard** (components\Monitoring\CostTrackingDashboard.tsx)
  - Exports: CostTrackingDashboard
- **LiveVideoGenerationMonitor** (components\Monitoring\LiveVideoGenerationMonitor.tsx)
  - Exports: LiveVideoGenerationMonitor
- **SystemHealthMonitors** (components\Monitoring\SystemHealthMonitors.tsx)
  - Exports: SystemHealthMonitors

### components\Navigation/
- **EnhancedNavigation** (components\Navigation\EnhancedNavigation.tsx)
  - Exports: EnhancedNavigation

### components\Onboarding/
- **OnboardingFlow** (components\Onboarding\OnboardingFlow.tsx)
  - Exports: OnboardingFlow

### components\UserJourney/
- **BetaUserJourneyOptimizer** (components\UserJourney\BetaUserJourneyOptimizer.tsx)
  - Exports: BetaUserJourneyOptimizer

### components\VideoQueue/
- **VideoQueueInterface** (components\VideoQueue\VideoQueueInterface.tsx)
  - Exports: VideoQueueInterface

### components\Videos/
- **PublishingControls** (components\Videos\PublishingControls.tsx)
  - Exports: PublishingControls
- **VideoApproval** (components\Videos\VideoApproval.tsx)
  - Exports: VideoApproval
- **VideoEngagementStats** (components\Videos\VideoEngagementStats.tsx)
  - Exports: VideoEngagementStats
- **VideoFilters** (components\Videos\VideoFilters.tsx)
  - Exports: VideoFilters
- **VideoGenerationForm** (components\Videos\VideoGenerationForm.tsx)
  - Exports: VideoGenerationForm
- **VideoList** (components\Videos\VideoList.tsx)
  - Exports: VideoList
- **VideoPerformanceChart** (components\Videos\VideoPerformanceChart.tsx)
  - Exports: VideoPerformanceChart
- **VideoPreview** (components\Videos\VideoPreview.tsx)
  - Exports: VideoPreview
- **VideoSearch** (components\Videos\VideoSearch.tsx)
  - Exports: VideoSearch
- **VideoUploadProgress** (components\Videos\VideoUploadProgress.tsx)
  - Exports: VideoUploadProgress
- **YouTubeUploadStatus** (components\Videos\YouTubeUploadStatus.tsx)
  - Exports: YouTubeUploadStatus


## USED COMPONENTS SUMMARY
----------------------------------------

### components/ (3 components)
- **Button** - Used in 133 places
- **Card** - Used in 97 places
- **Input** - Used in 41 places

### components\Accessibility/ (3 components)
- **ScreenReaderAnnouncer** - Used in 4 places
- **index** - Used in 2 places
- **SkipNavigation** - Used in 2 places

### components\Analytics/ (2 components)
- **AnalyticsDashboard** - Used in 2 places
- **UserBehaviorDashboard** - Used in 2 places

### components\Auth/ (1 components)
- **TwoFactorAuth** - Used in 1 places

### components\BulkOperations/ (1 components)
- **EnhancedBulkOperations** - Used in 2 places

### components\Channels/ (1 components)
- **BulkOperations** - Used in 2 places

### components\Charts/ (2 components)
- **index** - Used in 3 places
- **ChartComponents** - Used in 1 places

### components\Common/ (2 components)
- **HelpTooltip** - Used in 3 places
- **InlineHelp** - Used in 2 places

### components\Dashboard/ (9 components)
- **MetricCard** - Used in 8 places
- **VideoQueue** - Used in 5 places
- **DashboardHeader** - Used in 4 places
- **RealTimeMetrics** - Used in 3 places
- **RecentActivity** - Used in 3 places
- **BusinessIntelligenceDashboard** - Used in 2 places
- **DashboardLayout** - Used in 2 places
- **RevenueDashboard** - Used in 2 places
- **CostBreakdown** - Used in 1 places

### components\ErrorBoundary/ (3 components)
- **ErrorBoundary** - Used in 8 places
- **ErrorFallback** - Used in 2 places
- **RouteErrorBoundary** - Used in 2 places

### components\Experiments/ (1 components)
- **ABTestDashboard** - Used in 2 places

### components\Layout/ (2 components)
- **Header** - Used in 13 places
- **Sidebar** - Used in 2 places

### components\Loading/ (1 components)
- **LoadingSkeleton** - Used in 4 places

### components\PWA/ (1 components)
- **InstallPrompt** - Used in 2 places

### components\Performance/ (1 components)
- **PerformanceDashboard** - Used in 2 places

### components\ThemeToggle/ (1 components)
- **ThemeToggle** - Used in 1 places

### components\UILibrary/ (1 components)
- **ComponentLibrary** - Used in 73 places

### components\VideoEditor/ (1 components)
- **VideoEditor** - Used in 1 places

### components\Videos/ (4 components)
- **VideoPlayer** - Used in 4 places
- **GenerationProgress** - Used in 3 places
- **VideoCard** - Used in 2 places
- **VideoMetrics** - Used in 2 places


## MOST USED COMPONENTS (Top 10)
----------------------------------------
- **Button** (133 usages)
  - Path: components\Button.tsx
  - Imported by: components\Accessibility\AccessibleButton.tsx, components\Analytics\AnalyticsDashboard.tsx, components\Analytics\UserBehaviorDashboard.tsx...
- **Card** (97 usages)
  - Path: components\Card.tsx
  - Imported by: components\Analytics\AnalyticsDashboard.tsx, components\Analytics\UserBehaviorDashboard.tsx, components\BulkOperations\EnhancedBulkOperations.tsx...
- **ComponentLibrary** (73 usages)
  - Path: components\UILibrary\ComponentLibrary.tsx
  - Imported by: App.tsx, components\BulkOperations\EnhancedBulkOperations.tsx, components\Channels\BulkOperations.tsx...
- **Input** (41 usages)
  - Path: components\Input.tsx
  - Imported by: components\Analytics\AnalyticsDashboard.tsx, components\Analytics\UserBehaviorDashboard.tsx, components\Auth\ForgotPasswordForm.tsx...
- **Header** (13 usages)
  - Path: components\Layout\Header.tsx
  - Imported by: components\Analytics\AnalyticsDashboard.tsx, components\Channels\ChannelDashboard.tsx, components\Dashboard\BusinessIntelligenceDashboard.tsx...
- **MetricCard** (8 usages)
  - Path: components\Dashboard\MetricCard.tsx
  - Imported by: pages\Dashboard\Dashboard.tsx, pages\Dashboard\DashboardRealtime.tsx
- **ErrorBoundary** (8 usages)
  - Path: components\ErrorBoundary\ErrorBoundary.tsx
  - Imported by: App.tsx, components\Dashboard\MainDashboard.tsx, components\ErrorBoundary\RouteErrorBoundary.tsx...
- **VideoQueue** (5 usages)
  - Path: components\Dashboard\VideoQueue.tsx
  - Imported by: pages\Dashboard\Dashboard.tsx
- **ScreenReaderAnnouncer** (4 usages)
  - Path: components\Accessibility\ScreenReaderAnnouncer.tsx
  - Imported by: App.tsx, components\Dashboard\MainDashboard.tsx
- **DashboardHeader** (4 usages)
  - Path: components\Dashboard\DashboardHeader.tsx
  - Imported by: pages\Dashboard\Dashboard.tsx, pages\Dashboard\DashboardRealtime.tsx