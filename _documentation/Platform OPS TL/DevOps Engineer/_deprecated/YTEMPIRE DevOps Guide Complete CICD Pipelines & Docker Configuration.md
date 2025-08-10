# YTEMPIRE DevOps Guide: Complete CI/CD Pipelines & Docker Configuration
**Version 1.0 | January 2025**  
**Owner: DevOps Engineering Team**  
**Approved By: Platform Operations Lead**

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [GitHub Actions Workflows](#github-actions-workflows)
3. [Docker Configuration Files](#docker-configuration-files)
4. [Build Optimization Strategies](#build-optimization-strategies)
5. [Security Scanning Integration](#security-scanning-integration)
6. [Deployment Pipelines](#deployment-pipelines)
7. [Release Management](#release-management)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Best Practices and Guidelines](#best-practices-and-guidelines)

---

## Executive Summary

This document provides comprehensive CI/CD pipeline configurations and Docker containerization strategies for YTEMPIRE's platform. These configurations enable automated building, testing, and deployment of our services while maintaining security and performance standards.

### Key Pipeline Features
- **Automated Testing**: Unit, integration, and E2E test execution with >90% coverage
- **Security Scanning**: Container and dependency vulnerability scanning at every stage
- **Multi-Environment Support**: Development, staging, and production deployments
- **Performance Optimization**: Build times under 5 minutes with intelligent caching
- **Reliability**: Zero-downtime deployments with automated rollback capabilities

### Technology Stack
- **CI/CD Platform**: GitHub Actions
- **Container Registry**: Google Container Registry (GCR)
- **Container Runtime**: Docker with BuildKit
- **Orchestration**: Kubernetes (GKE)
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Security**: Trivy, Snyk, OWASP ZAP

---

## GitHub Actions Workflows

### 1. Main CI/CD Pipeline

```yaml
# .github/workflows/main-pipeline.yml
name: YTEMPIRE Main CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - development
        - staging
        - production

env:
  GCP_PROJECT_ID: ytempire-production
  GKE_CLUSTER: ytempire-prod-cluster
  GKE_ZONE: us-central1
  REGISTRY: gcr.io
  SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}

jobs:
  # Job 1: Code Quality and Security Checks
  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better analysis

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cache/pip
            ~/.npm
            ~/.cache/pre-commit
          key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements*.txt', '**/package-lock.json') }}

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          npm ci
          pre-commit install

      - name: Run pre-commit checks
        run: |
          pre-commit run --all-files

      - name: Run Python linters
        run: |
          # Format checking
          black --check .
          
          # Style checking
          flake8 . --count --statistics
          
          # Static analysis
          pylint src/ --fail-under=8.0
          
          # Type checking
          mypy src/ --strict

      - name: Run JavaScript/TypeScript linters
        run: |
          # ESLint
          npm run lint
          
          # TypeScript compilation
          npm run type-check
          
          # Prettier formatting
          npm run format:check

      - name: Run security checks
        run: |
          # Python security
          bandit -r src/ -ll
          safety check
          pip-audit
          
          # JavaScript security
          npm audit --audit-level=moderate
          
          # Secret scanning
          detect-secrets scan --baseline .secrets.baseline
          trufflehog filesystem . --json

      - name: SonarQube analysis
        uses: SonarSource/sonarqube-scan-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
        with:
          args: >
            -Dsonar.projectKey=ytempire
            -Dsonar.sources=.
            -Dsonar.host.url=${{ secrets.SONAR_HOST_URL }}

  # Job 2: Unit Tests
  unit-tests:
    name: Unit Tests - ${{ matrix.service }}
    runs-on: ubuntu-latest
    needs: code-quality
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        service: [api, processor, frontend, worker]
        include:
          - service: api
            language: python
            test-command: pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
          - service: processor
            language: python
            test-command: pytest tests/unit/ -v --cov=src --cov-report=xml
          - service: frontend
            language: node
            test-command: npm test -- --coverage --watchAll=false
          - service: worker
            language: python
            test-command: pytest tests/unit/ -v --cov=src --cov-report=xml
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup ${{ matrix.language }}
        uses: actions/setup-${{ matrix.language }}@v4
        with:
          ${{ matrix.language }}-version: ${{ matrix.language == 'python' && '3.11' || '18' }}
          cache: ${{ matrix.language == 'python' && 'pip' || 'npm' }}

      - name: Install dependencies
        run: |
          cd services/${{ matrix.service }}
          if [ "${{ matrix.language }}" = "python" ]; then
            pip install -r requirements.txt -r requirements-test.txt
          else
            npm ci
          fi

      - name: Run unit tests
        run: |
          cd services/${{ matrix.service }}
          ${{ matrix.test-command }}

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          file: ./services/${{ matrix.service }}/coverage.xml
          flags: ${{ matrix.service }}-unit
          name: ${{ matrix.service }}-unit-tests

      - name: Archive test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ matrix.service }}
          path: |
            services/${{ matrix.service }}/coverage/
            services/${{ matrix.service }}/test-results/
            services/${{ matrix.service }}/.coverage

  # Job 3: Build Docker Images
  build-images:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: unit-tests
    timeout-minutes: 30
    permissions:
      contents: read
      packages: write
      id-token: write
    strategy:
      matrix:
        service: [api, processor, frontend, worker]
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        with:
          driver-opts: |
            network=host
            image=moby/buildkit:latest

      - name: Log in to Google Container Registry
        uses: docker/login-action@v3
        with:
          registry: gcr.io
          username: _json_key
          password: ${{ secrets.GCP_SA_KEY }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: |
            gcr.io/${{ env.GCP_PROJECT_ID }}/ytempire-${{ matrix.service }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: ./services/${{ matrix.service }}
          file: ./services/${{ matrix.service }}/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: |
            type=gha
            type=registry,ref=gcr.io/${{ env.GCP_PROJECT_ID }}/ytempire-${{ matrix.service }}:buildcache
          cache-to: |
            type=gha,mode=max
            type=registry,ref=gcr.io/${{ env.GCP_PROJECT_ID }}/ytempire-${{ matrix.service }}:buildcache,mode=max
          build-args: |
            VERSION=${{ github.sha }}
            BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
            VCS_REF=${{ github.sha }}
          platforms: linux/amd64
          provenance: true
          sbom: true

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ fromJSON(steps.meta.outputs.json).tags[0] }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

      - name: Upload Trivy scan results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Snyk container scan
        continue-on-error: true
        run: |
          docker run --rm \
            -e SNYK_TOKEN=${{ secrets.SNYK_TOKEN }} \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v $(pwd):/project \
            snyk/snyk:docker test ${{ fromJSON(steps.meta.outputs.json).tags[0] }} \
            --severity-threshold=high

      - name: Sign container image
        env:
          COSIGN_EXPERIMENTAL: 1
        run: |
          cosign sign --yes ${{ fromJSON(steps.meta.outputs.json).tags[0] }}

      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: ${{ fromJSON(steps.meta.outputs.json).tags[0] }}
          format: spdx-json
          output-file: sbom-${{ matrix.service }}.spdx.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom-${{ matrix.service }}
          path: sbom-${{ matrix.service }}.spdx.json

  # Job 4: Integration Tests
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: build-images
    timeout-minutes: 30
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Kind cluster
        uses: engineerd/setup-kind@v0.5.0
        with:
          version: v0.17.0
          config: .github/kind-config.yaml

      - name: Load Docker images into Kind
        run: |
          for service in api processor frontend worker; do
            docker pull gcr.io/${{ env.GCP_PROJECT_ID }}/ytempire-${service}:${{ github.sha }}
            kind load docker-image gcr.io/${{ env.GCP_PROJECT_ID }}/ytempire-${service}:${{ github.sha }}
          done

      - name: Install test dependencies
        run: |
          # Install Helm
          curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
          
          # Install kubectl
          curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
          sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

      - name: Deploy test environment
        run: |
          # Create namespace
          kubectl create namespace test-env
          
          # Deploy dependencies
          helm repo add bitnami https://charts.bitnami.com/bitnami
          helm install postgresql bitnami/postgresql -n test-env
          helm install redis bitnami/redis -n test-env
          
          # Apply test manifests
          kubectl apply -f k8s/test/ -n test-env
          
          # Wait for pods to be ready
          kubectl wait --for=condition=ready pod -l app=ytempire -n test-env --timeout=300s

      - name: Run integration tests
        run: |
          # Port forward services
          kubectl port-forward svc/ytempire-api 8080:80 -n test-env &
          sleep 5
          
          # Run tests
          pip install -r requirements-test.txt
          pytest tests/integration/ -v --junit-xml=integration-results.xml

      - name: Collect logs on failure
        if: failure()
        run: |
          kubectl logs -l app=ytempire -n test-env --all-containers=true > pod-logs.txt
          kubectl describe pods -n test-env > pod-descriptions.txt
          kubectl get events -n test-env > events.txt

      - name: Upload test artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: integration-test-results
          path: |
            integration-results.xml
            pod-logs.txt
            pod-descriptions.txt
            events.txt

  # Job 5: End-to-End Tests
  e2e-tests:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    timeout-minutes: 45
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install Playwright
        run: |
          npm ci
          npx playwright install --with-deps

      - name: Run E2E tests
        run: |
          npm run test:e2e
        env:
          TEST_URL: ${{ secrets.STAGING_URL }}
          TEST_USER: ${{ secrets.TEST_USER }}
          TEST_PASSWORD: ${{ secrets.TEST_PASSWORD }}

      - name: Upload Playwright report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 30

  # Job 6: Performance Tests
  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    timeout-minutes: 60
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup K6
        run: |
          sudo gpg -k
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6

      - name: Run load tests
        run: |
          k6 run tests/performance/load-test.js \
            --out json=load-test-results.json \
            --summary-export=summary.json
        env:
          K6_CLOUD_TOKEN: ${{ secrets.K6_CLOUD_TOKEN }}
          BASE_URL: ${{ secrets.STAGING_URL }}

      - name: Analyze performance results
        run: |
          python scripts/analyze-performance.py load-test-results.json

      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-test-results
          path: |
            load-test-results.json
            summary.json
            performance-report.html

  # Job 7: Deploy to Staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build-images, integration-tests]
    if: github.ref == 'refs/heads/develop'
    timeout-minutes: 30
    environment:
      name: staging
      url: https://staging.ytempire.com
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Get GKE credentials
        uses: google-github-actions/get-gke-credentials@v1
        with:
          cluster_name: ytempire-staging-cluster
          location: us-central1

      - name: Install Helm
        uses: azure/setup-helm@v3
        with:
          version: '3.12.0'

      - name: Deploy with Helm
        run: |
          helm upgrade --install ytempire ./helm/ytempire \
            --namespace ytempire-staging \
            --create-namespace \
            --values helm/ytempire/values.staging.yaml \
            --set-string image.tag=${{ github.sha }} \
            --wait \
            --timeout 10m

      - name: Run smoke tests
        run: |
          ./scripts/smoke-tests.sh staging

      - name: Update deployment status
        uses: chrnorm/deployment-status@v2
        if: always()
        with:
          token: ${{ github.token }}
          environment-url: https://staging.ytempire.com
          state: ${{ job.status }}
          deployment-id: ${{ steps.deployment.outputs.deployment_id }}

      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Staging Deployment ${{ job.status }}
            Commit: ${{ github.sha }}
            Author: ${{ github.actor }}
            Message: ${{ github.event.head_commit.message }}
          webhook_url: ${{ env.SLACK_WEBHOOK }}

  # Job 8: Deploy to Production
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [deploy-staging, e2e-tests, performance-tests]
    if: github.ref == 'refs/heads/main'
    timeout-minutes: 45
    environment:
      name: production
      url: https://ytempire.com
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY_PROD }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Get GKE credentials
        uses: google-github-actions/get-gke-credentials@v1
        with:
          cluster_name: ytempire-prod-cluster
          location: us-central1

      - name: Install Helm
        uses: azure/setup-helm@v3
        with:
          version: '3.12.0'

      - name: Create deployment record
        id: deployment
        uses: chrnorm/deployment-action@v2
        with:
          token: ${{ github.token }}
          environment: production
          description: Production deployment of ${{ github.sha }}

      - name: Backup current state
        run: |
          ./scripts/backup-production-state.sh

      - name: Blue-Green Deployment
        run: |
          # Deploy to blue environment
          helm upgrade --install ytempire-blue ./helm/ytempire \
            --namespace ytempire-core \
            --values helm/ytempire/values.production.yaml \
            --values helm/ytempire/values.blue.yaml \
            --set-string image.tag=${{ github.sha }} \
            --wait \
            --timeout 15m
          
          # Run health checks on blue
          ./scripts/health-check.sh blue
          
          # Run canary tests
          ./scripts/canary-tests.sh blue
          
          # Switch traffic to blue
          kubectl patch service ytempire-api -n ytempire-core \
            -p '{"spec":{"selector":{"deployment":"blue"}}}'
          
          # Monitor for 5 minutes
          ./scripts/monitor-deployment.sh 300
          
          # Update green environment
          helm upgrade --install ytempire-green ./helm/ytempire \
            --namespace ytempire-core \
            --values helm/ytempire/values.production.yaml \
            --values helm/ytempire/values.green.yaml \
            --set-string image.tag=${{ github.sha }} \
            --wait \
            --timeout 15m

      - name: Run production validation
        run: |
          ./scripts/production-validation.sh

      - name: Create release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ github.run_number }}
          release_name: Production Release v${{ github.run_number }}
          body: |
            ## Production Deployment
            
            **Commit**: ${{ github.sha }}
            **Author**: ${{ github.actor }}
            **Message**: ${{ github.event.head_commit.message }}
            
            ### Changes
            ${{ github.event.compare }}
            
            ### Deployment Details
            - Blue environment updated first
            - Traffic switched after validation
            - Green environment updated
            - All health checks passed

      - name: Update deployment status
        uses: chrnorm/deployment-status@v2
        if: always()
        with:
          token: ${{ github.token }}
          environment-url: https://ytempire.com
          state: ${{ job.status }}
          deployment-id: ${{ steps.deployment.outputs.deployment_id }}

      - name: Notify teams
        if: always()
        run: |
          ./scripts/notify-deployment.sh production ${{ job.status }}
```

### 2. Automated Testing Workflow

```yaml
# .github/workflows/automated-tests.yml
name: Automated Test Suite

on:
  schedule:
    - cron: '0 */4 * * *'  # Every 4 hours
  workflow_dispatch:
    inputs:
      test-suite:
        description: 'Test suite to run'
        required: true
        default: 'all'
        type: choice
        options:
        - all
        - performance
        - security
        - chaos
        - regression

env:
  TEST_ENVIRONMENT: staging

jobs:
  performance-tests:
    name: Performance Test Suite
    runs-on: ubuntu-latest
    if: contains(fromJson('["all", "performance"]'), github.event.inputs.test-suite || 'all')
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup test environment
        run: |
          # Install K6
          sudo gpg -k
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
          
          # Install Artillery
          npm install -g artillery

      - name: Run load tests
        run: |
          # K6 load test
          k6 run tests/performance/load-test.js \
            --out cloud \
            --vus 100 \
            --duration 30m
          
          # Artillery stress test
          artillery run tests/performance/stress-test.yml

      - name: Run soak tests
        run: |
          k6 run tests/performance/soak-test.js \
            --vus 50 \
            --duration 2h

      - name: Analyze results
        run: |
          python scripts/analyze-performance.py \
            --input k6-results.json \
            --output performance-report.html \
            --threshold-file thresholds.yaml

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: performance-test-results
          path: |
            performance-report.html
            k6-results.json
            artillery-report.json

  security-tests:
    name: Security Test Suite
    runs-on: ubuntu-latest
    if: contains(fromJson('["all", "security"]'), github.event.inputs.test-suite || 'all')
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run OWASP ZAP scan
        run: |
          docker run -v $(pwd):/zap/wrk/:rw \
            -t owasp/zap2docker-stable zap-full-scan.py \
            -t https://${{ env.TEST_ENVIRONMENT }}.ytempire.com \
            -g gen.conf \
            -r zap-report.html \
            -w zap-report.md \
            -J zap-report.json \
            -a

      - name: Run Nuclei security scan
        run: |
          docker run --rm -v $(pwd):/app projectdiscovery/nuclei \
            -u https://${{ env.TEST_ENVIRONMENT }}.ytempire.com \
            -t /app/security-templates/ \
            -severity critical,high \
            -o nuclei-report.txt

      - name: Run SSL/TLS scan
        run: |
          docker run --rm drwetter/testssl.sh \
            https://${{ env.TEST_ENVIRONMENT }}.ytempire.com \
            > ssl-report.txt

      - name: Run dependency check
        run: |
          # Python dependencies
          pip install safety
          safety check --json > python-deps-report.json
          
          # Node dependencies
          npm audit --json > node-deps-report.json
          
          # Container scanning
          trivy image --format json \
            gcr.io/${{ env.GCP_PROJECT_ID }}/ytempire-api:latest \
            > container-scan-report.json

      - name: Generate security report
        run: |
          python scripts/generate-security-report.py \
            --zap-report zap-report.json \
            --nuclei-report nuclei-report.txt \
            --ssl-report ssl-report.txt \
            --deps-reports python-deps-report.json,node-deps-report.json \
            --container-report container-scan-report.json \
            --output security-report.html

      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-test-results
          path: |
            security-report.html
            zap-report.*
            nuclei-report.txt
            ssl-report.txt
            *-deps-report.json
            container-scan-report.json

  chaos-engineering:
    name: Chaos Engineering Tests
    runs-on: ubuntu-latest
    if: contains(fromJson('["all", "chaos"]'), github.event.inputs.test-suite || 'all')
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Get GKE credentials
        uses: google-github-actions/get-gke-credentials@v1
        with:
          cluster_name: ytempire-staging-cluster
          location: us-central1

      - name: Install Litmus Chaos
        run: |
          kubectl apply -f https://litmuschaos.github.io/litmus/litmus-operator-v2.14.0.yaml
          kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=litmus -n litmus --timeout=300s

      - name: Run chaos experiments
        run: |
          # Pod delete experiment
          kubectl apply -f tests/chaos/pod-delete-experiment.yaml
          
          # Network latency experiment
          kubectl apply -f tests/chaos/network-latency-experiment.yaml
          
          # CPU stress experiment
          kubectl apply -f tests/chaos/cpu-stress-experiment.yaml
          
          # Wait for experiments to complete
          sleep 600

      - name: Collect chaos results
        run: |
          kubectl get chaosresults -A -o json > chaos-results.json
          python scripts/analyze-chaos-results.py chaos-results.json

      - name: Generate chaos report
        run: |
          python scripts/generate-chaos-report.py \
            --results chaos-results.json \
            --output chaos-report.html

      - name: Upload chaos results
        uses: actions/upload-artifact@v3
        with:
          name: chaos-test-results
          path: |
            chaos-report.html
            chaos-results.json

  regression-tests:
    name: Regression Test Suite
    runs-on: ubuntu-latest
    if: contains(fromJson('["all", "regression"]'), github.event.inputs.test-suite || 'all')
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup test environment
        run: |
          pip install -r requirements-test.txt
          npm ci

      - name: Run API regression tests
        run: |
          pytest tests/regression/api/ -v \
            --json-report --json-report-file=api-regression-results.json

      - name: Run UI regression tests
        run: |
          npm run test:regression
          
      - name: Run visual regression tests
        run: |
          npm run test:visual

      - name: Compare with baseline
        run: |
          python scripts/compare-regression-results.py \
            --current api-regression-results.json \
            --baseline baseline-results.json \
            --output regression-report.html

      - name: Upload regression results
        uses: actions/upload-artifact@v3
        with:
          name: regression-test-results
          path: |
            regression-report.html
            api-regression-results.json
            visual-regression-results/

  test-summary:
    name: Test Summary Report
    runs-on: ubuntu-latest
    needs: [performance-tests, security-tests, chaos-engineering, regression-tests]
    if: always()
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v3

      - name: Generate summary report
        run: |
          python scripts/generate-test-summary.py \
            --performance performance-test-results/performance-report.html \
            --security security-test-results/security-report.html \
            --chaos chaos-test-results/chaos-report.html \
            --regression regression-test-results/regression-report.html \
            --output test-summary-report.html

      - name: Send summary to Slack
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              text: "Automated Test Suite Completed",
              attachments: [{
                color: "${{ job.status == 'success' && 'good' || 'danger' }}",
                title: "Test Results Summary",
                fields: [
                  { title: "Performance Tests", value: "${{ needs.performance-tests.result }}", short: true },
                  { title: "Security Tests", value: "${{ needs.security-tests.result }}", short: true },
                  { title: "Chaos Tests", value: "${{ needs.chaos-engineering.result }}", short: true },
                  { title: "Regression Tests", value: "${{ needs.regression-tests.result }}", short: true }
                ]
              }]
            }
          webhook_url: ${{ env.SLACK_WEBHOOK }}

      - name: Upload summary report
        uses: actions/upload-artifact@v3
        with:
          name: test-summary-report
          path: test-summary-report.html
```

### 3. Dependency Update Workflow

```yaml
# .github/workflows/dependency-updates.yml
name: Dependency Updates

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM UTC
  workflow_dispatch:

jobs:
  update-dependencies:
    name: Update Dependencies
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Update Python dependencies
        run: |
          # Install pip-tools
          pip install pip-tools
          
          # Update all Python services
          for service in api processor worker; do
            echo "Updating Python dependencies for $service..."
            cd services/$service
            
            # Update production dependencies
            pip-compile requirements.in --upgrade --resolver=backtracking
            
            # Update dev dependencies
            if [ -f requirements-dev.in ]; then
              pip-compile requirements-dev.in --upgrade --resolver=backtracking
            fi
            
            cd ../..
          done

      - name: Update Node.js dependencies
        run: |
          # Update frontend dependencies
          cd services/frontend
          
          # Update dependencies
          npm update
          
          # Audit and fix vulnerabilities
          npm audit fix
          
          # Update to latest minor versions
          npx npm-check-updates -u -t minor
          npm install
          
          cd ../..

      - name: Update Docker base images
        run: |
          # Script to update Docker base images
          python scripts/update-docker-images.py

      - name: Update Helm chart dependencies
        run: |
          cd helm/ytempire
          helm dependency update
          cd ../..

      - name: Run tests
        run: |
          # Run basic tests to ensure updates don't break anything
          make test-unit

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'chore: automated dependency updates'
          title: '[Automated] Weekly Dependency Updates'
          body: |
            ## Automated Dependency Updates
            
            This PR contains automated dependency updates for the week.
            
            ### Updates included:
            - Python dependencies (pip-compile)
            - Node.js dependencies (npm)
            - Docker base images
            - Helm chart dependencies
            
            ### Testing
            - [x] Unit tests passed
            - [ ] Integration tests (manual review required)
            - [ ] Security scan (automated in PR checks)
            
            ### Manual Review Required
            Please review the following:
            1. Check for breaking changes in updated packages
            2. Verify Docker image compatibility
            3. Test critical user flows
            
            Auto-generated by GitHub Actions workflow.
          branch: deps/automated-update-${{ github.run_number }}
          delete-branch: true
          labels: |
            dependencies
            automated
          assignees: |
            platform-ops-lead
          reviewers: |
            platform-ops-lead
            tech-lead
```

---

## Docker Configuration Files

### 1. API Service Dockerfile

```dockerfile
# services/api/Dockerfile
# Multi-stage build for optimal size and security
ARG PYTHON_VERSION=3.11

# Stage 1: Dependencies
FROM python:${PYTHON_VERSION}-slim as dependencies

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Build
FROM python:${PYTHON_VERSION}-slim as build

# Copy dependencies from previous stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY . .

# Compile Python files for optimization
RUN python -m compileall -b .

# Remove source files to reduce image size
RUN find . -name "*.py" -type f -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 3: Runtime
FROM python:${PYTHON_VERSION}-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ytempire && \
    useradd -r -g ytempire -u 1000 -m -s /bin/bash ytempire

# Set up application directory
WORKDIR /app
RUN chown -R ytempire:ytempire /app

# Copy compiled application from build stage
COPY --from=build --chown=ytempire:ytempire /app /app
COPY --from=dependencies --chown=ytempire:ytempire /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies --chown=ytempire:ytempire /usr/local/bin /usr/local/bin

# Security configurations
USER ytempire

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PYTHONPATH=/app \
    APP_ENV=production \
    PORT=8080

# Expose port
EXPOSE 8080

# Run the application with gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--max-requests", "10000", \
     "--max-requests-jitter", "1000", \
     "app.main:app"]
```

### 2. Video Processor Dockerfile

```dockerfile
# services/processor/Dockerfile
ARG CUDA_VERSION=12.0.0
ARG UBUNTU_VERSION=22.04

# Base image with CUDA support
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download AI models (with retry logic)
RUN mkdir -p /models && \
    for i in {1..3}; do \
        wget --tries=3 --timeout=30 -O /models/video_model.pt \
            https://storage.googleapis.com/ytempire-models/video_model_v1.pt && break || sleep 5; \
    done && \
    for i in {1..3}; do \
        wget --tries=3 --timeout=30 -O /models/audio_model.pt \
            https://storage.googleapis.com/ytempire-models/audio_model_v1.pt && break || sleep 5; \
    done

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r ytempire && \
    useradd -r -g ytempire -u 1000 -m -s /bin/bash ytempire && \
    chown -R ytempire:ytempire /app /models

# Create necessary directories
RUN mkdir -p /tmp/processing /var/log/ytempire && \
    chown -R ytempire:ytempire /tmp/processing /var/log/ytempire

USER ytempire

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/models \
    PROCESSING_PATH=/tmp/processing \
    LOG_PATH=/var/log/ytempire

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8081/health').raise_for_status()"

# Expose metrics port
EXPOSE 8081

# Run processor
CMD ["python3", "-m", "processor.main"]
```

### 3. Frontend Dockerfile

```dockerfile
# services/frontend/Dockerfile
ARG NODE_VERSION=18

# Stage 1: Dependencies
FROM node:${NODE_VERSION}-alpine as dependencies

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies with exact versions
RUN npm ci --only=production

# Stage 2: Build
FROM node:${NODE_VERSION}-alpine as build

WORKDIR /app

# Copy package files and install all dependencies (including dev)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code
COPY . .

# Build application
RUN npm run build

# Stage 3: Runtime
FROM nginx:alpine as runtime

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    tzdata \
    tini

# Copy nginx configuration
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
COPY nginx/security-headers.conf /etc/nginx/conf.d/security-headers.conf

# Copy built application from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy runtime configuration script
COPY docker/runtime-config.sh /docker-entrypoint.d/40-runtime-config.sh
RUN chmod +x /docker-entrypoint.d/40-runtime-config.sh

# Create non-root user
RUN adduser -D -H -u 1000 -s /sbin/nologin ytempire && \
    chown -R ytempire:ytempire /usr/share/nginx/html && \
    chown -R ytempire:ytempire /var/cache/nginx && \
    chown -R ytempire:ytempire /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R ytempire:ytempire /var/run/nginx.pid

# Configure nginx to run as non-root
RUN sed -i 's/user  nginx;/user  ytempire;/g' /etc/nginx/nginx.conf

USER ytempire

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

EXPOSE 80

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["nginx", "-g", "daemon off;"]
```

### 4. Worker Service Dockerfile

```dockerfile
# services/worker/Dockerfile
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim as dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

FROM python:${PYTHON_VERSION}-slim as runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r ytempire && \
    useradd -r -g ytempire -u 1000 -m -s /bin/bash ytempire

WORKDIR /app

COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

COPY --chown=ytempire:ytempire . .

USER ytempire

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
```

### 5. Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.9'

x-common-variables: &common-variables
  LOG_LEVEL: ${LOG_LEVEL:-debug}
  ENVIRONMENT: ${ENVIRONMENT:-development}
  
x-healthcheck-defaults: &healthcheck-defaults
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s

services:
  # API Service
  api:
    build:
      context: ./services/api
      dockerfile: Dockerfile
      target: runtime
      args:
        PYTHON_VERSION: "3.11"
    container_name: ytempire-api
    environment:
      <<: *common-variables
      APP_ENV: development
      DATABASE_URL: postgresql://ytempire:password@postgres:5432/ytempire
      REDIS_URL: redis://redis:6379/0
      JWT_SECRET: ${JWT_SECRET:-dev-secret-change-in-production}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      YOUTUBE_CLIENT_ID: ${YOUTUBE_CLIENT_ID}
      YOUTUBE_CLIENT_SECRET: ${YOUTUBE_CLIENT_SECRET}
    ports:
      - "8080:8080"
    volumes:
      - ./services/api:/app:cached
      - api-cache:/app/__pycache__
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - ytempire-network
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]

  # Video Processor
  processor:
    build:
      context: ./services/processor
      dockerfile: Dockerfile
      args:
        CUDA_VERSION: "12.0.0"
        UBUNTU_VERSION: "22.04"
    container_name: ytempire-processor
    environment:
      <<: *common-variables
      QUEUE_URL: redis://redis:6379/1
      STORAGE_BUCKET: ytempire-dev-videos
      GPU_MEMORY_FRACTION: ${GPU_MEMORY_FRACTION:-0.8}
      MODEL_CACHE_DIR: /models
    volumes:
      - ./services/processor:/app:cached
      - processor-models:/models
      - processor-workspace:/workspace
    depends_on:
      - redis
      - minio
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - ytempire-network
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "python3", "-c", "import requests; requests.get('http://localhost:8081/health').raise_for_status()"]

  # Frontend
  frontend:
    build:
      context: ./services/frontend
      dockerfile: Dockerfile
      target: build
      args:
        NODE_VERSION: "18"
    container_name: ytempire-frontend
    command: npm run dev
    environment:
      <<: *common-variables
      NODE_ENV: development
      VITE_API_URL: ${VITE_API_URL:-http://localhost:8080}
      VITE_WS_URL: ${VITE_WS_URL:-ws://localhost:8080}
    ports:
      - "3000:3000"
    volumes:
      - ./services/frontend:/app:cached
      - /app/node_modules
    networks:
      - ytempire-network
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:3000"]

  # Worker Service
  worker:
    build:
      context: ./services/worker
      dockerfile: Dockerfile
      args:
        PYTHON_VERSION: "3.11"
    container_name: ytempire-worker
    environment:
      <<: *common-variables
      CELERY_BROKER_URL: redis://redis:6379/2
      CELERY_RESULT_BACKEND: redis://redis:6379/3
      DATABASE_URL: postgresql://ytempire:password@postgres:5432/ytempire
    volumes:
      - ./services/worker:/app:cached
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    networks:
      - ytempire-network
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "celery", "-A", "worker.celery_app", "inspect", "ping"]

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: ytempire-postgres
    environment:
      POSTGRES_USER: ytempire
      POSTGRES_PASSWORD: password
      POSTGRES_DB: ytempire
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=en_US.UTF-8"
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/00-init.sql:ro
      - ./scripts/seed-data.sql:/docker-entrypoint-initdb.d/01-seed.sql:ro
    networks:
      - ytempire-network
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD-SHELL", "pg_isready -U ytempire -d ytempire"]
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: ytempire-redis
    command: >
      redis-server
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --appendfilename "appendonly.aof"
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - ytempire-network
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "redis-cli", "ping"]
    restart: unless-stopped

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: ytempire-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/alerts/:/etc/prometheus/alerts/:ro
      - prometheus-data:/prometheus
    networks:
      - ytempire-network
    restart: unless-stopped

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: ytempire-grafana
    environment:
      GF_SECURITY_ADMIN_USER: ${GF_ADMIN_USER:-admin}
      GF_SECURITY_ADMIN_PASSWORD: ${GF_ADMIN_PASSWORD:-admin}
      GF_USERS_ALLOW_SIGN_UP: "false"
      GF_INSTALL_PLUGINS: grafana-clock-panel,grafana-simple-json-datasource
    ports:
      - "3001:3000"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - ytempire-network
    restart: unless-stopped

  # Jaeger
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: ytempire-jaeger
    environment:
      COLLECTOR_ZIPKIN_HOST_PORT: ":9411"
      COLLECTOR_OTLP_ENABLED: "true"
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    networks:
      - ytempire-network
    restart: unless-stopped

  # N8N Automation
  n8n:
    image: n8nio/n8n:latest
    container_name: ytempire-n8n
    environment:
      N8N_BASIC_AUTH_ACTIVE: "true"
      N8N_BASIC_AUTH_USER: ${N8N_USER:-admin}
      N8N_BASIC_AUTH_PASSWORD: ${N8N_PASSWORD:-changeme}
      N8N_HOST: localhost
      N8N_PORT: 5678
      N8N_PROTOCOL: http
      NODE_ENV: development
      WEBHOOK_URL: http://n8n:5678/
      N8N_METRICS: "true"
    ports:
      - "5678:5678"
    volumes:
      - n8n-data:/home/node/.n8n
    networks:
      - ytempire-network
    restart: unless-stopped

  # MinIO Object Storage
  minio:
    image: minio/minio:latest
    container_name: ytempire-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minioadmin}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minioadmin}
      MINIO_PROMETHEUS_AUTH_TYPE: public
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data
    networks:
      - ytempire-network
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
    restart: unless-stopped

  # Mailhog (Email testing)
  mailhog:
    image: mailhog/mailhog:latest
    container_name: ytempire-mailhog
    ports:
      - "1025:1025"
      - "8025:8025"
    networks:
      - ytempire-network
    restart: unless-stopped

networks:
  ytempire-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
  api-cache:
    driver: local
  processor-models:
    driver: local
  processor-workspace:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
  n8n-data:
    driver: local
  minio-data:
    driver: local
```

### 6. Environment Configuration

```bash
# .env.example
# Copy this file to .env and update values

# Environment
ENVIRONMENT=development
LOG_LEVEL=debug

# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=ytempire
DB_USER=ytempire
DB_PASSWORD=changeme

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# API Configuration
API_PORT=8080
JWT_SECRET=your-secret-key-here-change-in-production
JWT_EXPIRY=86400
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Video Processing
GPU_MEMORY_FRACTION=0.8
MAX_CONCURRENT_VIDEOS=3
VIDEO_TIMEOUT=900
VIDEO_MAX_SIZE_MB=500

# External Services
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
YOUTUBE_CLIENT_ID=...
YOUTUBE_CLIENT_SECRET=...
YOUTUBE_REFRESH_TOKEN=...

# Storage
STORAGE_TYPE=minio
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=ytempire-videos
GCS_BUCKET=ytempire-dev-videos
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_S3_BUCKET=

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
JAEGER_ENABLED=true
JAEGER_AGENT_HOST=jaeger
JAEGER_AGENT_PORT=6831

# N8N
N8N_USER=admin
N8N_PASSWORD=changeme

# Email (Development)
SMTP_HOST=mailhog
SMTP_PORT=1025
SMTP_USER=
SMTP_PASSWORD=
SMTP_TLS=false
EMAIL_FROM=noreply@ytempire.local

# Feature Flags
FEATURE_VIDEO_OPTIMIZATION=true
FEATURE_AI_ENHANCEMENT=true
FEATURE_BATCH_PROCESSING=false
FEATURE_PREMIUM_MODELS=false

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Security
ENCRYPTION_KEY=32-byte-key-for-encryption-change-me
API_KEY_SALT=random-salt-for-api-keys-change-me
```

---

## Build Optimization Strategies

### 1. Docker Build Optimization

```dockerfile
# .dockerignore
# Exclude unnecessary files from Docker build context

# Version control
.git
.gitignore
.gitattributes

# Development files
.env
.env.*
!.env.example
.vscode
.idea
*.swp
*.swo
.DS_Store

# Test files
tests/
test/
*_test.py
*_test.go
*.test.js
coverage/
.coverage
htmlcov/
.pytest_cache/
.tox/

# Documentation
docs/
*.md
!README.md
LICENSE

# Build artifacts
build/
dist/
*.egg-info/
__pycache__/
*.pyc
*.pyo
node_modules/
npm-debug.log
yarn-error.log

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml
Jenkinsfile

# Docker files
Dockerfile*
docker-compose*.yml
!docker-compose.yml

# Temporary files
tmp/
temp/
*.tmp
*.log
```

### 2. Multi-Stage Build Best Practices

```dockerfile
# Optimized multi-stage Dockerfile example
ARG PYTHON_VERSION=3.11
ARG ALPINE_VERSION=3.18

# Base stage with common dependencies
FROM python:${PYTHON_VERSION}-alpine${ALPINE_VERSION} as base

# Install common system dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates \
    && rm -rf /var/cache/apk/*

# Dependencies stage
FROM base as dependencies

# Install build dependencies
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    python3-dev

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Remove build dependencies
RUN apk del .build-deps

# Build stage
FROM base as build

# Copy dependencies
COPY --from=dependencies /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages

# Copy and compile application
WORKDIR /app
COPY . .
RUN python -m compileall -b .

# Runtime stage
FROM base as runtime

# Create non-root user
RUN adduser -D -u 1000 ytempire

# Copy compiled application
WORKDIR /app
COPY --from=build --chown=ytempire:ytempire /app /app
COPY --from=dependencies --chown=ytempire:ytempire /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages

USER ytempire

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "-m", "app.main"]
```

### 3. Build Cache Optimization

```yaml
# .github/workflows/build-cache.yml
name: Optimized Build with Caching

on:
  push:
    branches: [main, develop]

env:
  REGISTRY: gcr.io
  GCP_PROJECT_ID: ytempire-production

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        with:
          driver-opts: |
            image=moby/buildkit:master
            network=host

      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: _json_key
          password: ${{ secrets.GCP_SA_KEY }}

      - name: Build with advanced caching
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.GCP_PROJECT_ID }}/ytempire-api:latest
          cache-from: |
            type=registry,ref=${{ env.REGISTRY }}/${{ env.GCP_PROJECT_ID }}/ytempire-api:buildcache
            type=gha
          cache-to: |
            type=registry,ref=${{ env.REGISTRY }}/${{ env.GCP_PROJECT_ID }}/ytempire-api:buildcache,mode=max
            type=gha,mode=max
          build-args: |
            BUILDKIT_INLINE_CACHE=1
```

### 4. Parallel Build Strategy

```makefile
# Makefile for parallel builds
.PHONY: all build test deploy clean

# Configuration
DOCKER_BUILDKIT := 1
COMPOSE_DOCKER_CLI_BUILD := 1
BUILDKIT_PROGRESS := plain

# Services
SERVICES := api processor frontend worker

# Parallel build all services
all: $(SERVICES)

$(SERVICES):
	@echo "Building $@..."
	@docker buildx build \
		--cache-from type=local,src=/tmp/.buildx-cache-$@ \
		--cache-to type=local,dest=/tmp/.buildx-cache-$@,mode=max \
		--platform linux/amd64,linux/arm64 \
		--tag ytempire-$@:latest \
		--load \
		services/$@

# Parallel test execution
test:
	@echo "Running tests in parallel..."
	@parallel --jobs 4 --will-cite ::: \
		"docker run --rm ytempire-api:latest pytest" \
		"docker run --rm ytempire-processor:latest pytest" \
		"docker run --rm ytempire-frontend:latest npm test" \
		"docker run --rm ytempire-worker:latest pytest"

# Clean build cache
clean:
	@docker buildx prune -f
	@rm -rf /tmp/.buildx-cache-*
```

---

## Security Scanning Integration

### 1. Comprehensive Security Pipeline

```yaml
# .github/workflows/security-scan.yml
name: Security Scanning Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM UTC

jobs:
  # Container Security Scanning
  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [api, processor, frontend, worker]
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build image
        run: |
          docker build -t ytempire-${{ matrix.service }}:scan ./services/${{ matrix.service }}

      - name: Run Trivy scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ytempire-${{ matrix.service }}:scan
          format: 'sarif'
          output: 'trivy-${{ matrix.service }}.sarif'
          severity: 'CRITICAL,HIGH,MEDIUM'
          vuln-type: 'os,library'
          ignore-unfixed: true

      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-${{ matrix.service }}.sarif'

      - name: Run Grype scanner
        uses: anchore/scan-action@v3
        with:
          image: ytempire-${{ matrix.service }}:scan
          fail-build: true
          severity-cutoff: high
          output-format: sarif
          output-file: grype-${{ matrix.service }}.sarif

      - name: Upload Grype results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'grype-${{ matrix.service }}.sarif'

  # Dependency Scanning
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'YTEMPIRE'
          path: '.'
          format: 'ALL'
          args: >
            --enableRetired
            --enableExperimental
            --nvdApiKey ${{ secrets.NVD_API_KEY }}

      - name: Upload OWASP results
        uses: actions/upload-artifact@v3
        with:
          name: dependency-check-report
          path: reports/

      - name: Python Safety Check
        run: |
          pip install safety
          find . -name requirements.txt -exec safety check --file {} \;

      - name: NPM Audit
        run: |
          find . -name package.json -not -path "*/node_modules/*" -execdir npm audit --json \; > npm-audit.json

  # Secret Scanning
  secrets-scan:
    name: Secrets Scanning
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: TruffleHog scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
          extra_args: --debug --only-verified

      - name: Gitleaks scan
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Detect Secrets
        run: |
          pip install detect-secrets
          detect-secrets scan --baseline .secrets.baseline
          detect-secrets audit .secrets.baseline

  # SAST Scanning
  sast-scan:
    name: Static Application Security Testing
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/owasp-top-ten
            p/dockerfile
            p/kubernetes

      - name: Run Bandit for Python
        run: |
          pip install bandit
          bandit -r services/api services/processor services/worker -f sarif -o bandit.sarif

      - name: Upload Bandit results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: bandit.sarif

      - name: Run ESLint security plugin
        run: |
          cd services/frontend
          npm install --save-dev eslint-plugin-security
          npx eslint --ext .js,.jsx,.ts,.tsx . --format sarif --output-file eslint.sarif

      - name: Upload ESLint results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: services/frontend/eslint.sarif

  # License Compliance
  license-scan:
    name: License Compliance Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run FOSSA scan
        uses: fossas/fossa-action@main
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}

      - name: Run License Finder
        run: |
          gem install license_finder
          license_finder approval add MIT
          license_finder approval add Apache-2.0
          license_finder approval add BSD-3-Clause
          license_finder report --format json > licenses.json

      - name: Upload license report
        uses: actions/upload-artifact@v3
        with:
          name: license-report
          path: licenses.json

  # Infrastructure Security
  infrastructure-scan:
    name: Infrastructure Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: .
          framework: all
          output_format: sarif
          output_file_path: checkov.sarif

      - name: Upload Checkov results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: checkov.sarif

      - name: Run Terrascan
        run: |
          docker run --rm -v $(pwd):/src accurics/terrascan scan -t docker,k8s,helm

      - name: Run Kubesec
        run: |
          find . -name "*.yaml" -o -name "*.yml" | \
          grep -E "(deployment|pod|statefulset|daemonset)" | \
          xargs -I {} docker run -v $(pwd):/app kubesec/kubesec scan /app/{}
```

### 2. Runtime Security Configuration

```yaml
# k8s/security/runtime-protection.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: falco-rules
  namespace: ytempire-security
data:
  ytempire_rules.yaml: |
    - rule: Unexpected Process in Container
      desc: Detect unexpected process execution in containers
      condition: >
        container and
        container.image.repository contains "ytempire" and
        not proc.name in (python, node, nginx, gunicorn, celery, ffmpeg) and
        not proc.pname in (python, node, sh, bash)
      output: >
        Unexpected process started in container
        (user=%user.name container=%container.name process=%proc.name command=%proc.cmdline)
      priority: WARNING
      tags: [container, process, ytempire]

    - rule: Write to System Directory
      desc: Detect writes to system directories
      condition: >
        container and
        container.image.repository contains "ytempire" and
        (fd.name startswith /etc/ or
         fd.name startswith /usr/ or
         fd.name startswith /bin/) and
        write and
        not proc.name in (package_manager_binaries)
      output: >
        Write to system directory in container
        (user=%user.name container=%container.name file=%fd.name)
      priority: ERROR
      tags: [container, filesystem, ytempire]

    - rule: Suspicious Network Activity
      desc: Detect suspicious network connections
      condition: >
        container and
        container.image.repository contains "ytempire" and
        outbound and
        not (fd.rip in (allowed_outbound_ips) or
             fd.rport in (80, 443, 5432, 6379, 9092))
      output: >
        Suspicious network connection from container
        (container=%container.name connection=%fd.name)
      priority: WARNING
      tags: [network, container, ytempire]

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: falco
  namespace: ytempire-security
spec:
  selector:
    matchLabels:
      app: falco
  template:
    metadata:
      labels:
        app: falco
    spec:
      serviceAccountName: falco
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      containers:
      - name: falco
        image: falcosecurity/falco:latest
        securityContext:
          privileged: true
        args:
          - /usr/bin/falco
          - -K
          - /var/run/secrets/kubernetes.io/serviceaccount/token
          - -k
          - https://kubernetes.default
          - -pk
        volumeMounts:
        - mountPath: /host/var/run/docker.sock
          name: docker-socket
        - mountPath: /host/dev
          name: dev-fs
        - mountPath: /host/proc
          name: proc-fs
          readOnly: true
        - mountPath: /host/boot
          name: boot-fs
          readOnly: true
        - mountPath: /host/lib/modules
          name: lib-modules
          readOnly: true
        - mountPath: /host/usr
          name: usr-fs
          readOnly: true
        - mountPath: /etc/falco
          name: falco-config
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
      - name: dev-fs
        hostPath:
          path: /dev
      - name: proc-fs
        hostPath:
          path: /proc
      - name: boot-fs
        hostPath:
          path: /boot
      - name: lib-modules
        hostPath:
          path: /lib/modules
      - name: usr-fs
        hostPath:
          path: /usr
      - name: falco-config
        configMap:
          name: falco-rules
```

### 3. Container Security Policies

```yaml
# k8s/security/pod-security-policies.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: ytempire-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  runAsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  readOnlyRootFilesystem: true

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ytempire-api-pdb
  namespace: ytempire-core
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: ytempire-api

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ytempire-core-quota
  namespace: ytempire-core
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    limits.cpu: "200"
    limits.memory: "400Gi"
    persistentvolumeclaims: "10"
    services.loadbalancers: "5"

---
apiVersion: v1
kind: LimitRange
metadata:
  name: ytempire-core-limits
  namespace: ytempire-core
spec:
  limits:
  - max:
      cpu: "4"
      memory: "16Gi"
    min:
      cpu: "100m"
      memory: "128Mi"
    default:
      cpu: "500m"
      memory: "1Gi"
    defaultRequest:
      cpu: "200m"
      memory: "512Mi"
    type: Container
```

---

## Deployment Pipelines

### 1. Blue-Green Deployment Strategy

```yaml
# helm/ytempire/templates/blue-green-deployment.yaml
{{- if .Values.blueGreen.enabled }}
---
# Blue Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ytempire.fullname" . }}-blue
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "ytempire.labels" . | nindent 4 }}
    deployment: blue
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "ytempire.selectorLabels" . | nindent 6 }}
      deployment: blue
  template:
    metadata:
      labels:
        {{- include "ytempire.selectorLabels" . | nindent 8 }}
        deployment: blue
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "{{ .Values.metrics.port }}"
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "ytempire.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
      - name: {{ .Chart.Name }}
        securityContext:
          {{- toYaml .Values.securityContext | nindent 12 }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: {{ .Values.service.targetPort }}
          protocol: TCP
        livenessProbe:
          httpGet:
            path: {{ .Values.healthcheck.liveness.path }}
            port: http
          initialDelaySeconds: {{ .Values.healthcheck.liveness.initialDelaySeconds }}
          periodSeconds: {{ .Values.healthcheck.liveness.periodSeconds }}
        readinessProbe:
          httpGet:
            path: {{ .Values.healthcheck.readiness.path }}
            port: http
          initialDelaySeconds: {{ .Values.healthcheck.readiness.initialDelaySeconds }}
          periodSeconds: {{ .Values.healthcheck.readiness.periodSeconds }}
        resources:
          {{- toYaml .Values.resources | nindent 12 }}

---
# Green Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ytempire.fullname" . }}-green
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "ytempire.labels" . | nindent 4 }}
    deployment: green
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "ytempire.selectorLabels" . | nindent 6 }}
      deployment: green
  template:
    metadata:
      labels:
        {{- include "ytempire.selectorLabels" . | nindent 8 }}
        deployment: green
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "{{ .Values.metrics.port }}"
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "ytempire.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
      - name: {{ .Chart.Name }}
        securityContext:
          {{- toYaml .Values.securityContext | nindent 12 }}
        image: "{{ .Values.image.repository }}:{{ .Values.blueGreen.greenTag | default .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: {{ .Values.service.targetPort }}
          protocol: TCP
        livenessProbe:
          httpGet:
            path: {{ .Values.healthcheck.liveness.path }}
            port: http
          initialDelaySeconds: {{ .Values.healthcheck.liveness.initialDelaySeconds }}
          periodSeconds: {{ .Values.healthcheck.liveness.periodSeconds }}
        readinessProbe:
          httpGet:
            path: {{ .Values.healthcheck.readiness.path }}
            port: http
          initialDelaySeconds: {{ .Values.healthcheck.readiness.initialDelaySeconds }}
          periodSeconds: {{ .Values.healthcheck.readiness.periodSeconds }}
        resources:
          {{- toYaml .Values.resources | nindent 12 }}

---
# Service pointing to active deployment
apiVersion: v1
kind: Service
metadata:
  name: {{ include "ytempire.fullname" . }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "ytempire.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
  - port: {{ .Values.service.port }}
    targetPort: http
    protocol: TCP
    name: http
  selector:
    {{- include "ytempire.selectorLabels" . | nindent 4 }}
    deployment: {{ .Values.blueGreen.activeDeployment }}
{{- end }}
```

### 2. Canary Deployment Configuration

```yaml
# k8s/deployments/canary-deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: ytempire-api-canary
  namespace: ytempire-core
spec:
  selector:
    app: ytempire-api
    version: canary
  ports:
  - port: 80
    targetPort: 8080

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ytempire-api-canary
  namespace: ytempire-core
spec:
  replicas: 1  # Start with minimal replicas
  selector:
    matchLabels:
      app: ytempire-api
      version: canary
  template:
    metadata:
      labels:
        app: ytempire-api
        version: canary
    spec:
      containers:
      - name: api
        image: gcr.io/ytempire-production/ytempire-api:canary
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        env:
        - name: VERSION
          value: "canary"
        - name: FEATURE_FLAGS
          value: "new_feature_x=true,experimental_y=true"

---
# Istio VirtualService for traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ytempire-api-canary
  namespace: ytempire-core
spec:
  hosts:
  - api.ytempire.com
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: ytempire-api-canary
        port:
          number: 80
  - route:
    - destination:
        host: ytempire-api
        port:
          number: 80
      weight: 95
    - destination:
        host: ytempire-api-canary
        port:
          number: 80
      weight: 5
```

### 3. Progressive Rollout Script

```bash
#!/bin/bash
# scripts/progressive-rollout.sh

set -euo pipefail

# Configuration
NAMESPACE="ytempire-core"
SERVICE="ytempire-api"
CANARY_DEPLOYMENT="ytempire-api-canary"
STABLE_DEPLOYMENT="ytempire-api"
MAX_CANARY_WEIGHT=100
STEP_WEIGHT=10
STEP_DURATION=300  # 5 minutes between steps
ERROR_THRESHOLD=0.05  # 5% error rate threshold

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check metrics for canary health
check_canary_health() {
    local error_rate=$(kubectl exec -n monitoring prometheus-0 -- \
        promtool query instant \
        'rate(http_requests_total{job="ytempire-api",version="canary",status=~"5.."}[5m]) / rate(http_requests_total{job="ytempire-api",version="canary"}[5m])' | \
        grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    if (( $(echo "$error_rate > $ERROR_THRESHOLD" | bc -l) )); then
        return 1
    fi
    return 0
}

# Update traffic weight
update_traffic_weight() {
    local weight=$1
    log_info "Updating canary traffic weight to ${weight}%"
    
    kubectl patch virtualservice ${SERVICE}-canary -n ${NAMESPACE} --type merge -p "{
        \"spec\": {
            \"http\": [{
                \"route\": [{
                    \"destination\": {
                        \"host\": \"${STABLE_DEPLOYMENT}\",
                        \"port\": {\"number\": 80}
                    },
                    \"weight\": $((100 - weight))
                }, {
                    \"destination\": {
                        \"host\": \"${CANARY_DEPLOYMENT}\",
                        \"port\": {\"number\": 80}
                    },
                    \"weight\": ${weight}
                }]
            }]
        }
    }"
}

# Rollback canary deployment
rollback_canary() {
    log_error "Rolling back canary deployment"
    update_traffic_weight 0
    kubectl scale deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} --replicas=0
    exit 1
}

# Main rollout logic
main() {
    log_info "Starting progressive rollout for ${SERVICE}"
    
    # Ensure canary deployment exists
    kubectl get deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} || {
        log_error "Canary deployment not found"
        exit 1
    }
    
    # Start with 0% traffic to canary
    update_traffic_weight 0
    
    # Scale canary deployment
    kubectl scale deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} --replicas=3
    
    # Wait for canary pods to be ready
    kubectl wait --for=condition=ready pod -l app=${SERVICE},version=canary -n ${NAMESPACE} --timeout=300s
    
    # Progressive traffic shift
    for weight in $(seq ${STEP_WEIGHT} ${STEP_WEIGHT} ${MAX_CANARY_WEIGHT}); do
        update_traffic_weight ${weight}
        log_info "Waiting ${STEP_DURATION} seconds before next step..."
        sleep ${STEP_DURATION}
        
        # Check canary health
        if ! check_canary_health; then
            log_error "Canary health check failed at ${weight}% traffic"
            rollback_canary
        fi
        
        log_info "Canary healthy at ${weight}% traffic"
    done
    
    # Full promotion
    log_info "Canary deployment successful, promoting to stable"
    
    # Update stable deployment with canary image
    CANARY_IMAGE=$(kubectl get deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} -o jsonpath='{.spec.template.spec.containers[0].image}')
    kubectl set image deployment/${STABLE_DEPLOYMENT} -n ${NAMESPACE} api=${CANARY_IMAGE}
    
    # Wait for stable rollout
    kubectl rollout status deployment/${STABLE_DEPLOYMENT} -n ${NAMESPACE}
    
    # Reset traffic to stable only
    update_traffic_weight 0
    
    # Scale down canary
    kubectl scale deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} --replicas=0
    
    log_info "Progressive rollout completed successfully"
}

# Run main function
main "$@"
```

---

## Release Management

### 1. Semantic Versioning Workflow

```yaml
# .github/workflows/release.yml
name: Release Management

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v1.2.3)'
        required: true
      prerelease:
        description: 'Is this a pre-release?'
        required: false
        default: false
        type: boolean

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Validate version
        run: |
          VERSION="${{ github.event.inputs.version || github.ref_name }}"
          if ! [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
            echo "Error: Invalid version format. Use v1.2.3 or v1.2.3-beta1"
            exit 1
          fi
          echo "VERSION=$VERSION" >> $GITHUB_ENV

      - name: Generate changelog
        id: changelog
        uses: mikepenz/release-changelog-builder-action@v3
        with:
          configuration: ".github/changelog-config.json"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Build release artifacts
        run: |
          make build-all
          make package-release VERSION=${{ env.VERSION }}

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ env.VERSION }}
          name: Release ${{ env.VERSION }}
          body: ${{ steps.changelog.outputs.changelog }}
          draft: false
          prerelease: ${{ github.event.inputs.prerelease || false }}
          files: |
            dist/ytempire-${{ env.VERSION }}.tar.gz
            dist/ytempire-${{ env.VERSION }}.zip
            dist/checksums.txt
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Update Helm chart
        run: |
          # Update Chart.yaml version
          sed -i "s/version: .*/version: ${VERSION#v}/" helm/ytempire/Chart.yaml
          sed -i "s/appVersion: .*/appVersion: ${VERSION}/" helm/ytempire/Chart.yaml
          
          # Package Helm chart
          helm package helm/ytempire
          
          # Push to Helm repository
          helm repo index . --url https://charts.ytempire.com
          gsutil cp ytempire-*.tgz gs://ytempire-helm-charts/
          gsutil cp index.yaml gs://ytempire-helm-charts/

      - name: Trigger deployment workflows
        uses: actions/github-script@v6
        with:
          script: |
            // Trigger staging deployment
            await github.rest.actions.createWorkflowDispatch({
              owner: context.repo.owner,
              repo: context.repo.repo,
              workflow_id: 'deploy-staging.yml',
              ref: 'main',
              inputs: {
                version: '${{ env.VERSION }}'
              }
            });
            
            // Create deployment issue
            const issue = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Deploy ${context.payload.ref} to Production`,
              body: `## Release ${context.payload.ref}
              
              ### Checklist
              - [ ] Staging deployment successful
              - [ ] E2E tests passed
              - [ ] Performance benchmarks met
              - [ ] Security scan clean
              - [ ] Documentation updated
              - [ ] Rollback plan reviewed
              
              ### Deployment Steps
              1. Review staging environment
              2. Approve production deployment
              3. Monitor deployment progress
              4. Verify production health
              
              /cc @platform-ops @tech-lead`,
              labels: ['deployment', 'production']
            });
```

### 2. Rollback Procedures

```bash
#!/bin/bash
# scripts/rollback.sh

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-staging}
NAMESPACE="ytempire-${ENVIRONMENT}"
ROLLBACK_VERSION=${2:-}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get deployment history
get_deployment_history() {
    log_info "Deployment history for ${ENVIRONMENT}:"
    helm history ytempire -n ${NAMESPACE} --max 10
}

# Perform rollback
perform_rollback() {
    local revision=$1
    
    log_warn "Rolling back to revision ${revision}"
    
    # Create backup of current state
    kubectl get all -n ${NAMESPACE} -o yaml > rollback-backup-$(date +%s).yaml
    
    # Perform Helm rollback
    helm rollback ytempire ${revision} -n ${NAMESPACE} --wait --timeout 10m
    
    # Verify rollback
    helm status ytempire -n ${NAMESPACE}
    
    # Run health checks
    ./scripts/health-check.sh ${ENVIRONMENT}
}

# Interactive rollback
interactive_rollback() {
    get_deployment_history
    
    echo ""
    read -p "Enter revision number to rollback to (or 'cancel'): " revision
    
    if [[ "${revision}" == "cancel" ]]; then
        log_info "Rollback cancelled"
        exit 0
    fi
    
    # Confirm rollback
    read -p "Are you sure you want to rollback to revision ${revision}? (yes/no): " confirm
    
    if [[ "${confirm}" == "yes" ]]; then
        perform_rollback ${revision}
    else
        log_info "Rollback cancelled"
        exit 0
    fi
}

# Main execution
main() {
    log_info "Rollback procedure for ${ENVIRONMENT} environment"
    
    if [[ -n "${ROLLBACK_VERSION}" ]]; then
        # Direct rollback to specified version
        perform_rollback ${ROLLBACK_VERSION}
    else
        # Interactive mode
        interactive_rollback
    fi
    
    log_info "Rollback completed"
}

main "$@"
```

### 3. Release Validation

```python
#!/usr/bin/env python3
# scripts/validate-release.py

import os
import sys
import json
import yaml
import subprocess
from typing import Dict, List, Tuple
import requests

class ReleaseValidator:
    """Validate release before deployment"""
    
    def __init__(self, version: str, environment: str = "staging"):
        self.version = version
        self.environment = environment
        self.validation_results = []
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        
        print(f"Validating release {self.version} for {self.environment}")
        
        checks = [
            ("Version Format", self.validate_version_format),
            ("Docker Images", self.validate_docker_images),
            ("Helm Chart", self.validate_helm_chart),
            ("API Compatibility", self.validate_api_compatibility),
            ("Database Migrations", self.validate_database_migrations),
            ("Configuration", self.validate_configuration),
            ("Security Scan", self.validate_security),
            ("Performance Baseline", self.validate_performance)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\nRunning {check_name}...")
            try:
                result, message = check_func()
                self.validation_results.append({
                    "check": check_name,
                    "passed": result,
                    "message": message
                })
                
                if result:
                    print(f"✓ {check_name}: PASSED")
                else:
                    print(f"✗ {check_name}: FAILED - {message}")
                    all_passed = False
                    
            except Exception as e:
                print(f"✗ {check_name}: ERROR - {str(e)}")
                self.validation_results.append({
                    "check": check_name,
                    "passed": False,
                    "message": f"Error: {str(e)}"
                })
                all_passed = False
        
        self.generate_report()
        return all_passed
    
    def validate_version_format(self) -> Tuple[bool, str]:
        """Validate version follows semantic versioning"""
        
        import re
        pattern = r'^v?\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?
        
        if re.match(pattern, self.version):
            return True, "Version format is valid"
        else:
            return False, f"Version {self.version} does not follow semantic versioning"
    
    def validate_docker_images(self) -> Tuple[bool, str]:
        """Validate all Docker images exist and are properly tagged"""
        
        services = ["api", "processor", "frontend", "worker"]
        missing_images = []
        
        for service in services:
            image = f"gcr.io/ytempire-production/ytempire-{service}:{self.version}"
            
            # Check if image exists
            result = subprocess.run(
                ["docker", "manifest", "inspect", image],
                capture_output=True
            )
            
            if result.returncode != 0:
                missing_images.append(service)
        
        if missing_images:
            return False, f"Missing images for services: {', '.join(missing_images)}"
        
        return True, "All Docker images are available"
    
    def validate_helm_chart(self) -> Tuple[bool, str]:
        """Validate Helm chart"""
        
        # Lint Helm chart
        result = subprocess.run(
            ["helm", "lint", "./helm/ytempire"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False, f"Helm lint failed: {result.stderr}"
        
        # Dry run installation
        result = subprocess.run(
            [
                "helm", "install", "ytempire", "./helm/ytempire",
                "--dry-run", "--debug",
                "--set", f"image.tag={self.version}",
                "-n", "test"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False, f"Helm dry-run failed: {result.stderr}"
        
        return True, "Helm chart validation passed"
    
    def validate_api_compatibility(self) -> Tuple[bool, str]:
        """Check API backward compatibility"""
        
        # Run API compatibility tests
        result = subprocess.run(
            ["npm", "run", "test:api-compatibility"],
            cwd="tests/compatibility",
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False, "API compatibility tests failed"
        
        return True, "API is backward compatible"
    
    def validate_database_migrations(self) -> Tuple[bool, str]:
        """Validate database migrations"""
        
        # Check for pending migrations
        result = subprocess.run(
            ["python", "manage.py", "showmigrations", "--plan"],
            cwd="services/api",
            capture_output=True,
            text=True
        )
        
        if "[ ]" in result.stdout:
            return False, "Pending database migrations found"
        
        # Test migration rollback
        result = subprocess.run(
            ["python", "manage.py", "migrate", "--fake", "--run-syncdb"],
            cwd="services/api",
            capture_output=True
        )
        
        if result.returncode != 0:
            return False, "Database migration test failed"
        
        return True, "Database migrations are valid"
    
    def validate_configuration(self) -> Tuple[bool, str]:
        """Validate configuration files"""
        
        config_files = [
            "k8s/production/configmap.yaml",
            "helm/ytempire/values.production.yaml"
        ]
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                return False, f"Configuration file missing: {config_file}"
            
            # Validate YAML syntax
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                return False, f"Invalid YAML in {config_file}: {e}"
        
        return True, "Configuration files are valid"
    
    def validate_security(self) -> Tuple[bool, str]:
        """Run security validation"""
        
        # Check for critical vulnerabilities
        result = subprocess.run(
            [
                "trivy", "image",
                "--severity", "CRITICAL,HIGH",
                "--exit-code", "1",
                f"gcr.io/ytempire-production/ytempire-api:{self.version}"
            ],
            capture_output=True
        )
        
        if result.returncode != 0:
            return False, "Critical vulnerabilities found in images"
        
        return True, "Security scan passed"
    
    def validate_performance(self) -> Tuple[bool, str]:
        """Check performance baselines"""
        
        # Run performance tests
        result = subprocess.run(
            ["k6", "run", "--quiet", "tests/performance/baseline.js"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False, "Performance baseline not met"
        
        return True, "Performance validation passed"
    
    def generate_report(self):
        """Generate validation report"""
        
        report = {
            "version": self.version,
            "environment": self.environment,
            "timestamp": subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                capture_output=True,
                text=True
            ).stdout.strip(),
            "results": self.validation_results,
            "summary": {
                "total_checks": len(self.validation_results),
                "passed": sum(1 for r in self.validation_results if r["passed"]),
                "failed": sum(1 for r in self.validation_results if not r["passed"])
            }
        }
        
        # Save report
        report_file = f"release-validation-{self.version}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation report saved to {report_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: validate-release.py <version> [environment]")
        sys.exit(1)
    
    version = sys.argv[1]
    environment = sys.argv[2] if len(sys.argv) > 2 else "staging"
    
    validator = ReleaseValidator(version, environment)
    
    if validator.validate_all():
        print("\n✅ Release validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Release validation FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Monitoring and Observability

### 1. Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ytempire-production'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

# Load rules
rule_files:
  - /etc/prometheus/rules/*.yml

# Scrape configurations
scrape_configs:
  # Kubernetes API server
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  # Kubernetes nodes
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
    - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - action: labelmap
      regex: __meta_kubernetes_node_label_(.+)

  # Kubernetes pods
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
    - action: labelmap
      regex: __meta_kubernetes_pod_label_(.+)
    - source_labels: [__meta_kubernetes_namespace]
      action: replace
      target_label: kubernetes_namespace
    - source_labels: [__meta_kubernetes_pod_name]
      action: replace
      target_label: kubernetes_pod_name

  # YTEMPIRE specific targets
  - job_name: 'ytempire-api'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names: ['ytempire-core']
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: ytempire-api
    - source_labels: [__meta_kubernetes_pod_container_port_name]
      action: keep
      regex: metrics

  - job_name: 'ytempire-processor'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names: ['ytempire-processing']
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: ytempire-processor
```

### 2. Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "uid": "ytempire-overview",
    "title": "YTEMPIRE Platform Overview",
    "tags": ["ytempire", "platform", "overview"],
    "timezone": "browser",
    "schemaVersion": 30,
    "version": 1,
    "refresh": "30s",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"ytempire-api\"}[5m])) by (method, status)",
            "legendFormat": "{{method}} - {{status}}"
          }
        ]
      },
      {
        "title": "Video Processing Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(videos_processed_total[5m])) by (status)",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Resource Utilization",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "avg(rate(container_cpu_usage_seconds_total{namespace=\"ytempire-core\"}[5m])) by (pod)",
            "legendFormat": "CPU - {{pod}}"
          },
          {
            "expr": "avg(container_memory_usage_bytes{namespace=\"ytempire-core\"}) by (pod)",
            "legendFormat": "Memory - {{pod}}"
          }
        ]
      },
      {
        "title": "Cost per Video",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
        "targets": [
          {
            "expr": "avg(video_processing_cost_dollars)"
          }
        ]
      },
      {
        "title": "Daily Active Channels",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16},
        "targets": [
          {
            "expr": "count(increase(channel_activity_total[24h]) > 0)"
          }
        ]
      },
      {
        "title": "Infrastructure Cost",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 16},
        "targets": [
          {
            "expr": "sum(infrastructure_cost_hourly_dollars) * 24"
          }
        ]
      },
      {
        "title": "System Uptime",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 16},
        "targets": [
          {
            "expr": "avg(up{job=~\"ytempire-.*\"})"
          }
        ]
      }
    ]
  }
}
```

### 3. Alerting Rules

```yaml
# monitoring/prometheus/rules/alerts.yml
groups:
  - name: ytempire.platform
    interval: 30s
    rules:
      # API Alerts
      - alert: APIHighErrorRate
        expr: |
          (sum(rate(http_requests_total{job="ytempire-api",status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total{job="ytempire-api"}[5m])) by (service)) > 0.05
        for: 5m
        labels:
          severity: critical
          team: platform-ops
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          description: "{{ $labels.service }} has error rate of {{ $value | humanizePercentage }}"
          runbook_url: "https://wiki.ytempire.com/runbooks/api-high-error-rate"

      - alert: APIHighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{job="ytempire-api"}[5m])) by (le, service)
          ) > 0.5
        for: 10m
        labels:
          severity: warning
          team: platform-ops
        annotations:
          summary: "High API latency on {{ $labels.service }}"
          description: "95th percentile latency is {{ $value }}s"

      # Video Processing Alerts
      - alert: VideoProcessingBacklog
        expr: |
          sum(video_processing_queue_depth) > 1000
        for: 15m
        labels:
          severity: warning
          team: platform-ops
        annotations:
          summary: "Video processing backlog growing"
          description: "Queue depth is {{ $value }} videos"

      - alert: VideoProcessingFailureRate
        expr: |
          sum(rate(video_processing_failures_total[5m])) 
          / 
          sum(rate(video_processing_total[5m])) > 0.1
        for: 10m
        labels:
          severity: critical
          team: platform-ops
        annotations:
          summary: "High video processing failure rate"
          description: "Failure rate is {{ $value | humanizePercentage }}"

      # Infrastructure Alerts
      - alert: PodCrashLooping
        expr: |
          rate(kube_pod_container_status_restarts_total{namespace=~"ytempire-.*"}[15m]) > 0
        for: 5m
        labels:
          severity: warning
          team: platform-ops
        annotations:
          summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"
          description: "Pod has restarted {{ $value }} times in the last 15 minutes"

      - alert: NodeMemoryPressure
        expr: |
          kube_node_status_condition{condition="MemoryPressure",status="true"} > 0
        for: 5m
        labels:
          severity: critical
          team: platform-ops
        annotations:
          summary: "Node {{ $labels.node }} under memory pressure"
          description: "Node has been under memory pressure for 5 minutes"

      - alert: PersistentVolumeSpaceLow
        expr: |
          kubelet_volume_stats_available_bytes / kubelet_volume_stats_capacity_bytes < 0.1
        for: 10m
        labels:
          severity: warning
          team: platform-ops
        annotations:
          summary: "PV {{ $labels.persistentvolumeclaim }} space low"
          description: "Only {{ $value | humanizePercentage }} space remaining"

      # Cost Alerts
      - alert: HighCostPerVideo
        expr: |
          avg(video_processing_cost_dollars) > 1.5
        for: 30m
        labels:
          severity: warning
          team: platform-ops
        annotations:
          summary: "Cost per video exceeds threshold"
          description: "Average cost per video is ${{ $value }}"

      - alert: DailyBudgetExceeded
        expr: |
          sum(increase(infrastructure_cost_total_dollars[24h])) > 5000
        for: 5m
        labels:
          severity: critical
          team: platform-ops
        annotations:
          summary: "Daily infrastructure budget exceeded"
          description: "Daily cost is ${{ $value }}"
```

---

## Troubleshooting Guide

### 1. Common Issues and Solutions

```markdown
# YTEMPIRE CI/CD Troubleshooting Guide

## Build Failures

### Issue: Docker build fails with "no space left on device"
**Symptoms:**
- Build fails during image layer creation
- Error message: "no space left on device"

**Solution:**
```bash
# Clean up Docker system
docker system prune -a -f

# Remove unused volumes
docker volume prune -f

# Check disk space
df -h

# If using GitHub Actions, add cleanup step
- name: Cleanup disk space
  run: |
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /opt/ghc
    sudo rm -rf "/usr/local/share/boost"
    sudo rm -rf "$AGENT_TOOLSDIRECTORY"
```

### Issue: Build cache not working
**Symptoms:**
- Every build takes full time
- No cache hits reported

**Solution:**
```yaml
# Ensure cache configuration is correct
- name: Build with proper caching
  uses: docker/build-push-action@v5
  with:
    cache-from: |
      type=gha
      type=registry,ref=${{ env.REGISTRY }}/image:buildcache
    cache-to: |
      type=gha,mode=max
      type=registry,ref=${{ env.REGISTRY }}/image:buildcache,mode=max
```

## Deployment Issues

### Issue: Deployment stuck in pending state
**Symptoms:**
- Pods remain in Pending state
- Deployment doesn't progress

**Diagnosis:**
```bash
# Check pod events
kubectl describe pod <pod-name> -n <namespace>

# Check node resources
kubectl top nodes

# Check PVC status
kubectl get pvc -n <namespace>

# Check resource quotas
kubectl describe resourcequota -n <namespace>
```

**Common Solutions:**
1. Insufficient resources - scale cluster
2. PVC not bound - check storage class
3. Image pull errors - check credentials
4. Node selector mismatch - check labels

### Issue: Health checks failing
**Symptoms:**
- Pods constantly restarting
- Readiness probe failed

**Solution:**
```yaml
# Adjust health check timings
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 60  # Increase for slow starting apps
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

## Performance Issues

### Issue: Slow build times
**Symptoms:**
- Builds taking >10 minutes
- No parallelization

**Solution:**
```makefile
# Enable parallel builds
MAKEFLAGS += -j4

# Use BuildKit
export DOCKER_BUILDKIT=1

# Optimize Dockerfile
# - Order layers by change frequency
# - Use multi-stage builds
# - Minimize layer count
```

### Issue: Deployment timeouts
**Symptoms:**
- Helm install/upgrade times out
- Pods take too long to become ready

**Solution:**
```bash
# Increase timeout
helm upgrade --install app ./chart \
  --wait \
  --timeout 15m

# Check for slow init containers
kubectl logs <pod> -c <init-container> -n <namespace>

# Optimize startup time
# - Reduce image size
# - Lazy load non-critical components
# - Implement proper health checks
```

## Security Scanning Issues

### Issue: False positive vulnerabilities
**Symptoms:**
- Build fails due to vulnerabilities
- Known false positives blocking deployment

**Solution:**
```yaml
# Create .trivyignore file
# Ignore specific CVEs
CVE-2021-12345

# Ignore by package
RUSTSEC-2021-0000

# Use time-limited ignores
CVE-2021-54321 exp:2025-01-31
```

### Issue: Secrets exposed in logs
**Symptoms:**
- Sensitive data visible in CI logs
- Secret scanning alerts

**Solution:**
```yaml
# Mask secrets in GitHub Actions
- name: Mask secrets
  run: |
    echo "::add-mask::${{ secrets.API_KEY }}"

# Use secret masking in scripts
set +x  # Disable command echoing
export SECRET_VALUE="${SECRET}"
set -x  # Re-enable if needed
```
```

### 2. Debug Scripts

```bash
#!/bin/bash
# scripts/debug-deployment.sh

set -euo pipefail

NAMESPACE=${1:-ytempire-core}
DEPLOYMENT=${2:-ytempire-api}

echo "=== Debugging deployment: ${DEPLOYMENT} in namespace: ${NAMESPACE} ==="

echo -e "\n--- Deployment Status ---"
kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE}

echo -e "\n--- ReplicaSet Status ---"
kubectl get rs -n ${NAMESPACE} -l app=${DEPLOYMENT}

echo -e "\n--- Pod Status ---"
kubectl get pods -n ${NAMESPACE} -l app=${DEPLOYMENT} -o wide

echo -e "\n--- Recent Events ---"
kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | grep -E "(${DEPLOYMENT}|Warning|Error)" | tail -20

echo -e "\n--- Pod Descriptions ---"
for pod in $(kubectl get pods -n ${NAMESPACE} -l app=${DEPLOYMENT} -o name); do
    echo -e "\n=== ${pod} ==="
    kubectl describe ${pod} -n ${NAMESPACE} | grep -A 10 -E "(Status:|Conditions:|Events:)"
done

echo -e "\n--- Container Logs (last 50 lines) ---"
for pod in $(kubectl get pods -n ${NAMESPACE} -l app=${DEPLOYMENT} -o name | head -3); do
    echo -e "\n=== ${pod} ==="
    kubectl logs ${pod} -n ${NAMESPACE} --tail=50 --all-containers=true
done

echo -e "\n--- Resource Usage ---"
kubectl top pods -n ${NAMESPACE} -l app=${DEPLOYMENT}

echo -e "\n--- Network Endpoints ---"
kubectl get endpoints -n ${NAMESPACE} | grep ${DEPLOYMENT}

echo -e "\n--- ConfigMaps and Secrets ---"
kubectl get configmaps,secrets -n ${NAMESPACE} | grep ${DEPLOYMENT}

echo -e "\n--- Ingress Status ---"
kubectl get ingress -n ${NAMESPACE} | grep ${DEPLOYMENT}

echo -e "\n--- Service Status ---"
kubectl get svc -n ${NAMESPACE} | grep ${DEPLOYMENT}
kubectl describe svc ${DEPLOYMENT} -n ${NAMESPACE} | grep -A 5 "Endpoints:"
```

### 3. Performance Debugging

```python
#!/usr/bin/env python3
# scripts/debug-performance.py

import subprocess
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt

class PerformanceDebugger:
    """Debug performance issues in CI/CD pipeline"""
    
    def __init__(self, repo: str, workflow: str):
        self.repo = repo
        self.workflow = workflow
        self.github_token = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True
        ).stdout.strip()
    
    def analyze_workflow_runs(self, days: int = 7) -> Dict:
        """Analyze workflow run performance"""
        
        # Get workflow runs
        cmd = [
            "gh", "api",
            f"/repos/{self.repo}/actions/workflows/{self.workflow}/runs",
            "--paginate",
            "-q", f".workflow_runs[] | select(.created_at > \"{(datetime.now() - timedelta(days=days)).isoformat()}\")"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        runs = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
        
        # Analyze run times
        run_times = []
        job_times = {}
        
        for run in runs:
            if run['status'] == 'completed':
                start = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
                duration = (end - start).total_seconds()
                run_times.append(duration)
                
                # Get job details
                jobs_cmd = [
                    "gh", "api",
                    f"/repos/{self.repo}/actions/runs/{run['id']}/jobs"
                ]
                jobs_result = subprocess.run(jobs_cmd, capture_output=True, text=True)
                jobs = json.loads(jobs_result.stdout)
                
                for job in jobs['jobs']:
                    job_name = job['name']
                    if job['completed_at']:
                        job_start = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
                        job_end = datetime.fromisoformat(job['completed_at'].replace('Z', '+00:00'))
                        job_duration = (job_end - job_start).total_seconds()
                        
                        if job_name not in job_times:
                            job_times[job_name] = []
                        job_times[job_name].append(job_duration)
        
        # Calculate statistics
        analysis = {
            'total_runs': len(runs),
            'successful_runs': len(run_times),
            'average_duration': statistics.mean(run_times) if run_times else 0,
            'median_duration': statistics.median(run_times) if run_times else 0,
            'min_duration': min(run_times) if run_times else 0,
            'max_duration': max(run_times) if run_times else 0,
            'job_analysis': {}
        }
        
        for job_name, times in job_times.items():
            analysis['job_analysis'][job_name] = {
                'average': statistics.mean(times),
                'median': statistics.median(times),
                'min': min(times),
                'max': max(times)
            }
        
        return analysis
    
    def identify_bottlenecks(self, analysis: Dict) -> List[str]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        # Find slowest jobs
        job_times = [
            (name, stats['average']) 
            for name, stats in analysis['job_analysis'].items()
        ]
        job_times.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 slowest jobs
        for job, avg_time in job_times[:3]:
            if avg_time > 300:  # More than 5 minutes
                bottlenecks.append(f"Job '{job}' averages {avg_time:.1f}s")
        
        # Check for high variance
        for job, stats in analysis['job_analysis'].items():
            variance = stats['max'] - stats['min']
            if variance > stats['average'] * 0.5:
                bottlenecks.append(f"Job '{job}' has high variance: {variance:.1f}s")
        
        return bottlenecks
    
    def generate_report(self, output_file: str = "performance-report.md"):
        """Generate performance report"""
        
        analysis = self.analyze_workflow_runs()
        bottlenecks = self.identify_bottlenecks(analysis)
        
        report = f"""# CI/CD Performance Analysis Report

Generated: {datetime.now().isoformat()}
Repository: {self.repo}
Workflow: {self.workflow}

## Summary

- Total runs analyzed: {analysis['total_runs']}
- Successful runs: {analysis['successful_runs']}
- Average duration: {analysis['average_duration']:.1f}s ({analysis['average_duration']/60:.1f}m)
- Median duration: {analysis['median_duration']:.1f}s
- Min duration: {analysis['min_duration']:.1f}s
- Max duration: {analysis['max_duration']:.1f}s

## Job Analysis

| Job Name | Average | Median | Min | Max |
|----------|---------|--------|-----|-----|
"""
        
        for job, stats in sorted(analysis['job_analysis'].items(), 
                                key=lambda x: x[1]['average'], 
                                reverse=True):
            report += f"| {job} | {stats['average']:.1f}s | {stats['median']:.1f}s | {stats['min']:.1f}s | {stats['max']:.1f}s |\n"
        
        report += f"""

## Identified Bottlenecks

"""
        for bottleneck in bottlenecks:
            report += f"- {bottleneck}\n"
        
        report += """

## Recommendations

1. **Cache Optimization**
   - Review cache hit rates
   - Implement layer caching for Docker builds
   - Use GitHub Actions cache for dependencies

2. **Parallelization**
   - Run independent jobs in parallel
   - Use matrix builds for multiple configurations
   - Split large test suites

3. **Resource Allocation**
   - Use larger runners for resource-intensive jobs
   - Optimize Docker image sizes
   - Remove unnecessary build steps

4. **Monitoring**
   - Set up alerts for long-running workflows
   - Track performance trends over time
   - Regular review of pipeline efficiency
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Performance report generated: {output_file}")
        
        # Generate visualization
        self.plot_performance_trends(analysis)
    
    def plot_performance_trends(self, analysis: Dict):
        """Generate performance visualization"""
        
        # Create job duration chart
        jobs = list(analysis['job_analysis'].keys())
        avg_times = [stats['average'] for stats in analysis['job_analysis'].values()]
        
        plt.figure(figsize=(12, 6))
        plt.bar(jobs, avg_times)
        plt.xlabel('Job Name')
        plt.ylabel('Average Duration (seconds)')
        plt.title('Average Job Duration - CI/CD Pipeline')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('job-duration-chart.png')
        plt.close()
        
        print("Performance chart generated: job-duration-chart.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: debug-performance.py <repo> <workflow>")
        sys.exit(1)
    
    debugger = PerformanceDebugger(sys.argv[1], sys.argv[2])
    debugger.generate_report()
```

---

## Best Practices and Guidelines

### 1. CI/CD Best Practices

```markdown
# YTEMPIRE CI/CD Best Practices

## Pipeline Design

### 1. Keep It Fast
- **Target**: < 10 minutes for standard builds
- **Strategies**:
  - Parallelize independent jobs
  - Use caching aggressively
  - Optimize Docker layers
  - Skip unnecessary steps (conditional execution)

### 2. Fail Fast
- Run quick checks first (linting, security scans)
- Use `fail-fast: true` in matrix builds
- Set reasonable timeouts
- Provide clear error messages

### 3. Make It Reproducible
- Pin all dependency versions
- Use deterministic builds
- Document all environment requirements
- Version control everything

### 4. Security First
- Never commit secrets
- Use least-privilege service accounts
- Scan for vulnerabilities at every stage
- Sign and verify artifacts

## Docker Best Practices

### 1. Image Optimization
```dockerfile
# Good: Multi-stage build
FROM node:18 as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
CMD ["node", "server.js"]
```

### 2. Layer Caching
- Order Dockerfile commands by change frequency
- Separate dependencies from application code
- Use cache mounts for package managers
- Leverage BuildKit features

### 3. Security Hardening
- Use minimal base images
- Run as non-root user
- Scan for vulnerabilities
- Don't include secrets in images
- Use read-only root filesystem

## Testing Strategy

### 1. Test Pyramid
```
        /\
       /  \    E2E Tests (Few)
      /    \
     /      \  Integration Tests (Some)
    /        \
   /          \ Unit Tests (Many)
  /____________\
```

### 2. Test Parallelization
- Split large test suites
- Use test sharding
- Run independent test types concurrently
- Cache test dependencies

### 3. Test Data Management
- Use test containers for databases
- Implement proper test isolation
- Clean up test data
- Use factories for test data generation

## Deployment Safety

### 1. Progressive Rollouts
- Start with canary deployments
- Monitor key metrics during rollout
- Implement automatic rollback triggers
- Use feature flags for gradual enablement

### 2. Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

### 3. Rollback Strategy
- Maintain previous version availability
- Test rollback procedures regularly
- Document rollback triggers
- Automate where possible

## Monitoring and Observability

### 1. Pipeline Metrics
- Build duration
- Success/failure rates
- Queue time
- Resource utilization

### 2. Deployment Metrics
- Deployment frequency
- Lead time for changes
- Mean time to recovery
- Change failure rate

### 3. Alerting
- Failed builds on main branch
- Security vulnerabilities
- Long-running pipelines
- Deployment failures

## Cost Optimization

### 1. Resource Management
- Use appropriate runner sizes
- Implement job concurrency limits
- Clean up old artifacts
- Use spot instances where possible

### 2. Caching Strategy
- Cache Docker layers
- Cache dependencies
- Cache build artifacts
- Monitor cache hit rates

### 3. Artifact Management
- Set retention policies
- Compress artifacts
- Use external storage for large files
- Clean up unused images
```

### 2. Security Guidelines

```yaml
# .github/security-guidelines.yml
security_guidelines:
  secrets_management:
    - Use GitHub Secrets for sensitive data
    - Rotate secrets regularly
    - Use least-privilege principles
    - Implement secret scanning
    - Never log sensitive information
  
  image_security:
    - Scan all images before deployment
    - Use minimal base images
    - Update base images regularly
    - Sign container images
    - Implement admission controllers
  
  dependency_management:
    - Regular dependency updates
    - Automated vulnerability scanning
    - License compliance checks
    - Supply chain security
    - SBOM generation
  
  access_control:
    - Implement RBAC
    - Use service accounts
    - Enable MFA
    - Regular access reviews
    - Audit logging
  
  compliance:
    - GDPR compliance
    - SOC 2 readiness
    - HIPAA considerations
    - Industry standards
    - Regular audits
```

### 3. Operational Runbooks

```markdown
# Operational Runbooks

## Emergency Response

### Production Outage
1. **Assess Impact**
   ```bash
   # Check service status
   kubectl get pods -n ytempire-core
   kubectl get svc -n ytempire-core
   
   # Check recent deployments
   kubectl get deployments -n ytempire-core -o wide
   ```

2. **Immediate Mitigation**
   ```bash
   # Rollback if recent deployment
   ./scripts/rollback.sh production
   
   # Scale up if resource issue
   kubectl scale deployment ytempire-api --replicas=10 -n ytempire-core
   ```

3. **Root Cause Analysis**
   - Collect logs
   - Review metrics
   - Check recent changes
   - Document findings

### Security Incident
1. **Contain**
   - Isolate affected systems
   - Revoke compromised credentials
   - Block suspicious traffic

2. **Investigate**
   - Review audit logs
   - Analyze attack vectors
   - Identify affected data

3. **Remediate**
   - Patch vulnerabilities
   - Update configurations
   - Strengthen defenses

4. **Report**
   - Document incident
   - Notify stakeholders
   - Update procedures

## Maintenance Procedures

### Certificate Renewal
```bash
# Check certificate expiry
kubectl get certificates -A
kubectl describe certificate ytempire-tls -n ytempire-core

# Trigger renewal
kubectl delete certificate ytempire-tls -n ytempire-core
# Cert-manager will recreate
```

### Database Maintenance
```bash
# Backup before maintenance
./scripts/backup-database.sh

# Run maintenance
kubectl exec -it postgres-0 -n database -- psql -U postgres -c "VACUUM ANALYZE;"

# Verify health
kubectl exec -it postgres-0 -n database -- psql -U postgres -c "SELECT version();"
```

### Cluster Upgrades
1. **Pre-upgrade**
   - Review changelog
   - Test in staging
   - Backup configurations
   - Plan maintenance window

2. **Upgrade Process**
   ```bash
   # Cordon nodes
   kubectl cordon <node-name>
   
   # Drain workloads
   kubectl drain <node-name> --ignore-daemonsets
   
   # Upgrade node
   # (Platform specific)
   
   # Uncordon node
   kubectl uncordon <node-name>
   ```

3. **Post-upgrade**
   - Verify functionality
   - Run smoke tests
   - Monitor for issues
   - Update documentation
```

---

## Document Information

**Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: February 2025  
**Owner**: DevOps Engineering Team  
**Approved By**: Platform Operations Lead

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 2025 | DevOps Team | Initial comprehensive guide |

### Contributing

To contribute to this guide:
1. Create a feature branch
2. Make your changes
3. Test all examples
4. Submit a pull request
5. Request review from Platform Ops Lead

### Support

For questions or issues:
- Slack: #platform-ops
- Email: devops@ytempire.com
- Wiki: https://wiki.ytempire.com/devops

---

**Note**: This document contains production configurations and procedures. Handle with appropriate security measures. All examples should be tested in non-production environments before implementation.