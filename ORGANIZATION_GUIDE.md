# YTEmpire Project Organization Guide

## 📁 Project Structure & File Organization

This guide documents where different types of files should be placed in the YTEmpire MVP project to maintain a clean and organized codebase.

### Directory Structure Overview

```
ytempire-mvp/
├── _documentation/          # All project documentation
│   ├── AI-ML TL/           # AI/ML team documentation
│   ├── Backend TL/          # Backend team documentation
│   ├── Frontend TL/         # Frontend team documentation
│   ├── Platform OPS TL/     # DevOps team documentation
│   ├── Data TL/             # Data team documentation
│   └── *.md                 # Project-wide documentation
│
├── backend/                 # Backend application
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core utilities & frameworks
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic services
│   │   └── tasks/          # Background tasks
│   ├── tests/              # Backend tests
│   └── alembic/            # Database migrations
│
├── frontend/                # Frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   ├── stores/         # State management
│   │   └── utils/          # Utility functions
│   ├── public/
│   │   └── tools/          # Development tools (PWA generator, etc.)
│   └── tests/
│       └── accessibility/  # Accessibility tests
│
├── ml-pipeline/            # Machine Learning pipeline
│   ├── src/                # ML source code
│   ├── services/           # ML services
│   └── quality_scoring/    # Quality assessment
│
├── infrastructure/         # Infrastructure & DevOps
│   ├── docker/            # Docker configurations
│   ├── kubernetes/        # K8s configurations
│   ├── monitoring/        # Monitoring setup
│   ├── security/          # Security configurations
│   │   ├── reports/       # Security scan reports
│   │   └── *.py          # Security scripts
│   ├── scaling/          # Auto-scaling configurations
│   └── scripts/          # Infrastructure scripts
│
├── scripts/               # Project-wide utility scripts
│   ├── ml/               # ML-specific scripts
│   ├── security/         # Security scripts
│   └── disaster-recovery/# DR scripts
│
├── tests/                # Project-wide tests
│   └── performance/      # Performance tests
│
└── misc/                 # SHOULD BE EMPTY (temporary files only)
```

## 📋 File Placement Guidelines

### Documentation Files (.md, .txt)
- **Project Documentation**: `_documentation/`
- **Team-Specific Docs**: `_documentation/[Team] TL/`
- **API Documentation**: `backend/docs/` or `_documentation/`
- **Frontend Documentation**: `frontend/docs/` or `_documentation/`

### Python Scripts (.py)
- **Backend API Code**: `backend/app/`
- **ML Pipeline Code**: `ml-pipeline/src/` or `ml-pipeline/services/`
- **Security Scripts**: `infrastructure/security/`
- **Data Pipeline Scripts**: `backend/app/services/` or separate `data-pipeline/`
- **Test Scripts**: `[component]/tests/`
- **Utility Scripts**: `scripts/`

### TypeScript/JavaScript Files (.ts, .tsx, .js, .jsx)
- **React Components**: `frontend/src/components/`
- **Frontend Services**: `frontend/src/services/`
- **Frontend Utils**: `frontend/src/utils/`
- **Frontend Tests**: `frontend/tests/`
- **Build Tools**: `frontend/public/tools/`

### Configuration Files
- **Docker Files**: `infrastructure/docker/` or root for main docker-compose
- **Kubernetes**: `infrastructure/kubernetes/`
- **CI/CD**: `.github/workflows/`
- **Environment**: Root directory (`.env`, `.env.example`)

### Reports & Results
- **Security Reports**: `infrastructure/security/reports/`
- **Performance Reports**: `tests/performance/reports/`
- **Analytics Reports**: `_documentation/` or `data-pipeline/reports/`

## 🚀 Recently Reorganized Files

The following files were moved from `misc/` to their proper locations:

1. **Documentation** → `_documentation/`
   - `YTEmpire_MVP_Progress_Report_Week0_Week1.md`
   - `week0-week1-progress-analysis.md`
   - `dashboard_access_guide.md`

2. **Backend Core** → `backend/app/core/`
   - `error_handling_framework.py`

3. **Security** → `infrastructure/security/`
   - `security_compliance_check.py`
   - `run_security_scan.py`
   - `security_scan_simple.py`

4. **Security Reports** → `infrastructure/security/reports/`
   - `compliance_report.json`
   - `security_scan_results.json`

5. **Frontend Tests** → `frontend/tests/accessibility/`
   - `test-accessibility.ts`

6. **Frontend Tools** → `frontend/public/tools/`
   - `generate-pwa-icons.html`

## ✅ Best Practices

1. **Never leave files in `misc/`** - It should only contain temporary files during development
2. **Create proper subdirectories** when adding new categories of files
3. **Follow naming conventions**:
   - Use kebab-case for files: `user-service.py`
   - Use PascalCase for React components: `UserProfile.tsx`
   - Use UPPER_CASE for documentation: `README.md`, `SETUP.md`
4. **Group related files** in feature-specific folders
5. **Keep tests close** to the code they test
6. **Document new directories** by adding README files

## 🔍 Quick Reference

| File Type | Location |
|-----------|----------|
| API Endpoints | `backend/app/api/v1/endpoints/` |
| React Components | `frontend/src/components/` |
| Database Models | `backend/app/models/` |
| ML Models | `ml-pipeline/src/` |
| Docker Configs | `infrastructure/docker/` or root |
| Security Scripts | `infrastructure/security/` |
| Test Files | `[component]/tests/` |
| Documentation | `_documentation/` |
| CI/CD Workflows | `.github/workflows/` |
| Utility Scripts | `scripts/` |

## 📝 Notes

- The `misc/` folder should remain empty except during active development
- All production code should be in appropriate module directories
- Documentation should be colocated with code when specific to a component
- Project-wide documentation goes in `_documentation/`
- Always create a README in new directories explaining their purpose

---

*Last Updated: August 12, 2025*
*Maintained by: Development Team*