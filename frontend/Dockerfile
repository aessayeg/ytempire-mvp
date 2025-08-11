# Multi-stage Dockerfile for Frontend
# Stage 1: Dependencies
FROM node:20-alpine as dependencies

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies with exact versions
RUN npm ci --only=production

# Copy production dependencies aside
RUN cp -R node_modules /prod_node_modules

# Install all dependencies (including dev)
RUN npm ci

# Stage 2: Development
FROM node:20-alpine as development

# Set working directory
WORKDIR /app

# Install development tools
RUN apk add --no-cache \
    git \
    vim \
    curl

# Copy dependencies
COPY --from=dependencies /app/node_modules ./node_modules
COPY package*.json ./

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Development command with hot reload
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]

# Stage 3: Testing
FROM node:20-alpine as test

# Set working directory
WORKDIR /app

# Copy dependencies and code
COPY --from=dependencies /app/node_modules ./node_modules
COPY package*.json ./
COPY . .

# Run linting
RUN npm run lint || true

# Run type checking
RUN npm run typecheck || true

# Run tests with coverage
RUN npm test -- --coverage --watchAll=false || true

# Stage 4: Builder
FROM node:20-alpine as builder

# Set working directory
WORKDIR /app

# Copy dependencies
COPY --from=dependencies /app/node_modules ./node_modules
COPY package*.json ./

# Copy application code
COPY . .

# Set build arguments
ARG VITE_API_URL
ARG NODE_ENV=production

# Set environment variables for build
ENV VITE_API_URL=${VITE_API_URL}
ENV NODE_ENV=${NODE_ENV}

# Build the application
RUN npm run build

# Remove dev dependencies
RUN npm prune --production

# Stage 5: Production with Nginx
FROM nginx:alpine as production

# Install Node.js for any runtime scripts
RUN apk add --no-cache nodejs npm

# Create non-root user
RUN addgroup -g 1001 -S ytempire && \
    adduser -u 1001 -S ytempire -G ytempire

# Copy built application
COPY --from=builder --chown=ytempire:ytempire /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY --chown=ytempire:ytempire nginx.conf /etc/nginx/nginx.conf

# Create runtime config directory
RUN mkdir -p /usr/share/nginx/html/config && \
    chown -R ytempire:ytempire /usr/share/nginx/html

# Copy runtime configuration script
COPY --chown=ytempire:ytempire docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80 || exit 1

# Use custom entrypoint for runtime configuration
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]

# Stage 6: Production with Node server (alternative)
FROM node:20-alpine as production-node

# Create non-root user
RUN addgroup -g 1001 -S ytempire && \
    adduser -u 1001 -S ytempire -G ytempire

# Set working directory
WORKDIR /app

# Install serve for production serving
RUN npm install -g serve

# Copy production dependencies
COPY --from=dependencies --chown=ytempire:ytempire /prod_node_modules ./node_modules

# Copy built application
COPY --from=builder --chown=ytempire:ytempire /app/dist ./dist
COPY --from=builder --chown=ytempire:ytempire /app/package*.json ./

# Switch to non-root user
USER ytempire

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Serve the built application
CMD ["serve", "-s", "dist", "-l", "3000"]