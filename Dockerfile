
FROM node:20-bookworm-slim

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node.js dependencies
COPY package*.json ./
RUN npm ci

# Install Python dependencies
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application files
COPY . .

# Build the frontend
RUN npm run build

# Configure production environment
ENV NODE_ENV=production
ENV PYTHON_BIN=python3

# Render provides the PORT environment variable
CMD ["npm", "start"]
