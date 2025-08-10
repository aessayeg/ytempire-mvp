# YTEMPIRE React Engineer - Getting Started Guide
**Document Version**: 2.0  
**Last Updated**: January 2025  
**Document Type**: Setup & Development Guide

---

## 1. Prerequisites

### 1.1 System Requirements

```yaml
Operating System:
  - macOS 12+ (recommended)
  - Ubuntu 20.04+
  - Windows 10+ with WSL2

Hardware:
  - CPU: 4+ cores recommended
  - RAM: 8GB minimum, 16GB recommended
  - Storage: 10GB free space

Software:
  - Node.js: 18.0.0 or higher
  - npm: 9.0.0 or higher
  - Git: 2.30.0 or higher
  - VS Code: Latest version (recommended)
```

### 1.2 Required Access

```yaml
Accounts & Access:
  - GitHub: Repository access (ytempire/frontend)
  - Figma: Design file access (request from UI/UX Designer)
  - Slack: Join #frontend-team channel
  - Jira: Project board access
  - Confluence: Documentation space

Development Tools:
  - Chrome DevTools
  - React Developer Tools extension
  - Redux DevTools extension (for Zustand)
  
API Access:
  - Development API key (request from Backend Team)
  - Test user credentials
```

---

## 2. Environment Setup

### 2.1 Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/ytempire/frontend.git
cd frontend

# 2. Install Node.js (if not installed)
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# 3. Install dependencies
npm ci  # Use ci for exact version installation

# 4. Copy environment variables
cp .env.example .env.local

# 5. Update .env.local with your values
# Edit the file with your preferred editor
code .env.local
```

### 2.2 Environment Configuration

```bash
# .env.local - Development environment
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
VITE_CDN_URL=http://localhost:8000/static

# Feature flags
VITE_ENABLE_WEBSOCKET=true
VITE_ENABLE_ANALYTICS=false
VITE_ENABLE_DEBUG=true
VITE_MOCK_API=false

# Development settings
VITE_LOG_LEVEL=debug
VITE_POLLING_INTERVAL=60000

# API Keys (get from team)
VITE_YOUTUBE_API_KEY=your_dev_key_here
VITE_SENTRY_DSN=your_sentry_dsn_here
```

### 2.3 VS Code Setup

```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,
  "emmet.includeLanguages": {
    "typescript": "typescriptreact"
  }
}
```

### 2.4 Recommended VS Code Extensions

```json
// .vscode/extensions.json
{
  "recommendations": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "bradlc.vscode-tailwindcss",
    "dsznajder.es7-react-js-snippets",
    "formulahendry.auto-rename-tag",
    "naumovs.color-highlight",
    "pflannery.vscode-versionlens",
    "yzhang.markdown-all-in-one",
    "ms-vscode.vscode-typescript-next"
  ]
}
```

---

## 3. Development Workflow

### 3.1 Daily Development Flow

```bash
# 1. Start your day - pull latest changes
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/YTE-123-channel-creation-modal

# 3. Start development server
npm run dev
# Server runs at http://localhost:5173

# 4. Start testing in watch mode (separate terminal)
npm run test:watch

# 5. Check TypeScript types (separate terminal)
npm run type-check:watch

# 6. Make changes and test
# The dev server will hot-reload automatically

# 7. Run linting before commit
npm run lint

# 8. Commit changes
git add .
git commit -m "feat(channels): add channel creation modal"

# 9. Push and create PR
git push origin feature/YTE-123-channel-creation-modal
# Create PR on GitHub
```

### 3.2 Available Scripts

```json
{
  "scripts": {
    // Development
    "dev": "vite",                          // Start dev server
    "dev:host": "vite --host",              // Expose to network
    
    // Building
    "build": "tsc && vite build",           // Production build
    "build:analyze": "vite build --analyze", // Analyze bundle
    "preview": "vite preview",              // Preview production build
    
    // Testing
    "test": "vitest",                       // Run tests once
    "test:watch": "vitest --watch",         // Watch mode
    "test:coverage": "vitest --coverage",   // Coverage report
    "test:ui": "vitest --ui",              // UI test runner
    
    // Type Checking
    "type-check": "tsc --noEmit",          // Check types
    "type-check:watch": "tsc --noEmit -w",  // Watch mode
    
    // Linting & Formatting
    "lint": "eslint src --ext .ts,.tsx",   // Lint code
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write src/**/*.{ts,tsx,css}",
    
    // Utilities
    "clean": "rm -rf dist node_modules",    // Clean project
    "deps:check": "npm outdated",           // Check outdated
    "deps:update": "npm update",            // Update deps
  }
}
```

### 3.3 Git Workflow

```bash
# Branch naming conventions
feature/YTE-{ticket}-{description}   # New features
fix/YTE-{ticket}-{description}       # Bug fixes
refactor/YTE-{ticket}-{description}  # Refactoring
docs/YTE-{ticket}-{description}      # Documentation

# Commit message format
<type>(<scope>): <subject>

# Examples
feat(channels): add channel creation modal
fix(auth): resolve token refresh race condition
refactor(dashboard): optimize metric calculations
docs(readme): update setup instructions

# PR workflow
1. Create feature branch from develop
2. Make changes and commit
3. Push branch to origin
4. Create PR with template
5. Request review from team lead
6. Address feedback
7. Merge after approval
```

---

## 4. Project Structure Guide

### 4.1 Where to Put Things

```yaml
New Component:
  Path: src/components/{category}/{ComponentName}/
  Files:
    - ComponentName.tsx       # Component implementation
    - ComponentName.test.tsx  # Tests
    - ComponentName.types.ts  # Types (if complex)
    - index.ts               # Public export

New Page/Screen:
  Path: src/pages/{PageName}/
  Note: Max 20-25 screens for MVP

New Store:
  Path: src/stores/use{StoreName}.ts
  Note: Max 5 stores for MVP

New Service:
  Path: src/services/{serviceName}.ts
  Pattern: Class-based service with methods

New Hook:
  Path: src/hooks/use{HookName}.ts
  Prefix: Always start with 'use'

New Utility:
  Path: src/utils/{utilityName}.ts
  Types: formatters, validators, helpers

Type Definitions:
  Path: src/types/{domain}.types.ts
  Categories: api, models, components
```

### 4.2 Import Order Convention

```typescript
// 1. React and third-party libraries
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Button } from '@mui/material';

// 2. Internal absolute imports
import { useAuthStore } from '@/stores/useAuthStore';
import { channelService } from '@/services/channels';

// 3. Internal relative imports
import { ChannelCard } from '../components/ChannelCard';
import { formatCurrency } from '../utils/formatters';

// 4. Types
import type { Channel } from '@/types/models.types';

// 5. Styles
import styles from './Dashboard.module.css';
```

---

## 5. Common Tasks

### 5.1 Creating a New Component

```bash
# 1. Check component count (must be under 40)
# Current count: Check src/components/

# 2. Create component structure
mkdir -p src/components/features/channels/ChannelCreator
cd src/components/features/channels/ChannelCreator

# 3. Create files
touch ChannelCreator.tsx
touch ChannelCreator.test.tsx
touch ChannelCreator.types.ts
touch index.ts

# 4. Implement component following pattern
# See Development Standards document

# 5. Write tests (minimum 70% coverage)

# 6. Export from index
echo "export { ChannelCreator } from './ChannelCreator';" > index.ts

# 7. Add to components barrel export
# Edit src/components/index.ts
```

### 5.2 Adding a New API Endpoint

```typescript
// 1. Add types to src/types/api.types.ts
export interface CreateVideoRequest {
  channelId: string;
  topic: string;
  // ...
}

// 2. Add to service in src/services/videos.ts
async createVideo(data: CreateVideoRequest): Promise<Video> {
  return apiClient.post('/videos', data);
}

// 3. Use in component or store
const handleCreateVideo = async (data: CreateVideoRequest) => {
  try {
    const video = await videoService.createVideo(data);
    // Handle success
  } catch (error) {
    errorHandler.handle(error);
  }
};
```

### 5.3 Setting Up a New Store

```typescript
// src/stores/useNewStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface NewStore {
  // State
  data: any[];
  loading: boolean;
  
  // Actions
  fetchData: () => Promise<void>;
  updateData: (id: string, updates: any) => void;
}

export const useNewStore = create<NewStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        data: [],
        loading: false,
        
        // Actions
        fetchData: async () => {
          set({ loading: true });
          try {
            // Fetch data
            set({ data: [], loading: false });
          } catch (error) {
            set({ loading: false });
          }
        },
        
        updateData: (id, updates) => {
          set((state) => ({
            data: state.data.map(item =>
              item.id === id ? { ...item, ...updates } : item
            )
          }));
        }
      }),
      {
        name: 'new-store',
        // Only persist what's needed
        partialize: (state) => ({ data: state.data })
      }
    )
  )
);
```

---

## 6. Debugging & Troubleshooting

### 6.1 Common Issues

#### Build Errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite

# Check for TypeScript errors
npm run type-check
```

#### Performance Issues

```typescript
// Use React DevTools Profiler
// 1. Open Chrome DevTools
// 2. Go to Profiler tab
// 3. Start recording
// 4. Perform actions
// 5. Stop and analyze

// Add performance marks
performance.mark('myComponent-start');
// Component logic
performance.mark('myComponent-end');
performance.measure('myComponent', 'myComponent-start', 'myComponent-end');
```

#### State Management Issues

```typescript
// Enable Zustand devtools
const useStore = create()(
  devtools(
    (set) => ({
      // store implementation
    }),
    {
      name: 'MyStore', // Shows in Redux DevTools
    }
  )
);

// Debug in console
const state = useStore.getState();
console.log('Current state:', state);
```

### 6.2 Useful Debug Commands

```bash
# Check bundle size
npm run build:analyze

# Find large dependencies
npm ls --depth=0 | grep -E "^├|^└" | sort -k2 -hr

# Check for security vulnerabilities
npm audit

# Fix vulnerabilities
npm audit fix

# Check TypeScript performance
npx tsc --diagnostics

# Profile build time
time npm run build
```

---

## 7. Resources & Help

### 7.1 Documentation

- **Internal Docs**: Confluence (Frontend Space)
- **API Docs**: http://localhost:8000/api/docs
- **Design System**: Figma (request access)
- **Component Library**: Storybook (post-MVP)

### 7.2 Key Contacts

```yaml
Frontend Team Lead:
  Slack: @frontend-lead
  Expertise: Architecture, code reviews
  
Dashboard Specialist:
  Slack: @dashboard-dev
  Expertise: Charts, data visualization
  
UI/UX Designer:
  Slack: @design-team
  Expertise: Figma, design specs
  
Backend Team:
  Channel: #backend-team
  API Issues: @api-developer
```

### 7.3 Escalation Path

1. **Check documentation** - This guide and others
2. **Search in Slack** - #frontend-team history
3. **Ask team member** - Dashboard Specialist or Designer
4. **Ask Team Lead** - For architectural decisions
5. **Escalate to CTO** - For blocking issues

### 7.4 Learning Resources

```yaml
React 18:
  - https://react.dev/
  - https://react.dev/learn

TypeScript:
  - https://www.typescriptlang.org/docs/
  - https://github.com/typescript-cheatsheets/react

Zustand:
  - https://github.com/pmndrs/zustand
  - https://docs.pmnd.rs/zustand/getting-started

Material-UI:
  - https://mui.com/material-ui/
  - https://mui.com/material-ui/getting-started/

Recharts:
  - https://recharts.org/
  - https://recharts.org/en-US/examples

Vite:
  - https://vitejs.dev/
  - https://vitejs.dev/guide/
```

---

## 8. Week 1 Checklist

### Your First Week Goals

```markdown
## Day 1
- [ ] Environment setup complete
- [ ] Repository cloned and running
- [ ] Slack channels joined
- [ ] Met team members

## Day 2-3
- [ ] First PR submitted (small fix or improvement)
- [ ] Familiar with codebase structure
- [ ] Development workflow understood
- [ ] API endpoints reviewed

## Day 4-5
- [ ] Built 2-3 components
- [ ] Tests written for components
- [ ] Zustand store interaction understood
- [ ] Attended all team meetings

## End of Week 1
- [ ] Authentication flow understood
- [ ] Basic layout structure familiar
- [ ] Channel list component started
- [ ] Documentation reviewed
- [ ] Questions list prepared for Week 2
```

---

**Document Status**: FINAL - Consolidated Version  
**Next Review**: Onboarding Review Week 2  
**Owner**: Frontend Team Lead  
**Questions**: Contact via #frontend-team Slack