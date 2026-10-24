# -----------------------------------------------------------------------------
# Base Image
# -----------------------------------------------------------------------------
FROM node:20-alpine AS base

# -----------------------------------------------------------------------------
# 1. Install Dependencies
# -----------------------------------------------------------------------------
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

# -----------------------------------------------------------------------------
# 2. Build Next.js Application
# -----------------------------------------------------------------------------
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Pass build-time environment variable for client-side API calls
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# -----------------------------------------------------------------------------
# 3. Production Runner Stage
# -----------------------------------------------------------------------------
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy static assets and public directory
COPY --from=builder /app/public ./public

# Set permissions for Next.js cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# Leverage standalone output traces
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 4000
ENV PORT=4000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]