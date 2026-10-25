import express from 'express';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn, ChildProcess } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const FLASK_PORT = 5001;

let flaskProcess: ChildProcess | null = null;

function getPythonCommand(): string {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;

  // Auto-detect virtual environments if user created venv or .venv
  const isWin = process.platform === 'win32';
  const candidates = [
    path.resolve(__dirname, 'venv', isWin ? 'Scripts/python.exe' : 'bin/python'),
    path.resolve(__dirname, '.venv', isWin ? 'Scripts/python.exe' : 'bin/python'),
  ];

  for (const cand of candidates) {
    if (fs.existsSync(cand)) {
      console.log(`[AutoML Server] Using virtual environment Python: ${cand}`);
      return cand;
    }
  }

  return isWin ? 'python' : 'python3';
}

// Start Python Flask ML Backend
function startFlaskBackend() {
  const backendScript = path.resolve(__dirname, 'backend', 'app.py');
  const pythonCmd = getPythonCommand();
  console.log(`[AutoML Server] Spawning Python Flask service via '${pythonCmd}' on port ${FLASK_PORT}...`);

  flaskProcess = spawn(pythonCmd, [backendScript, '--port', String(FLASK_PORT)], {
    cwd: path.resolve(__dirname, 'backend'),
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });

  flaskProcess.stdout?.on('data', (data) => {
    console.log(`[Flask stdout]: ${data.toString().trim()}`);
  });

  flaskProcess.stderr?.on('data', (data) => {
    console.error(`[Flask stderr]: ${data.toString().trim()}`);
  });

  flaskProcess.on('exit', (code, signal) => {
    console.log(`[Flask exit]: Exited with code ${code}, signal ${signal}`);
    if (code !== 0 && code !== null) {
      console.warn(`\n[AutoML Studio Tip] Python backend exited. If packages are missing, run in your terminal:`);
      console.warn(`   pip install -r requirements.txt\n`);
    }
  });
}

startFlaskBackend();

// Reverse proxy for /api requests to Python Flask backend
app.use('/api', (req, res) => {
  const proxyReq = http.request(
    {
      hostname: '127.0.0.1',
      port: FLASK_PORT,
      path: req.originalUrl,
      method: req.method,
      headers: {
        ...req.headers,
        host: `127.0.0.1:${FLASK_PORT}`,
      },
    },
    (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
      proxyRes.pipe(res);
    }
  );

  proxyReq.on('error', (err) => {
    console.error(`[Proxy Error] failed connecting to Flask backend:`, err.message);
    res.status(502).json({
      error: 'Python ML Backend is unavailable. Please install Python dependencies using: pip install -r requirements.txt',
      details: err.message,
    });
  });

  req.pipe(proxyReq);
});

// Vite middleware in dev or static files in production
async function setupApp() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n  ➜  Local:   http://localhost:${PORT}/`);
    console.log(`  ➜  Network: http://127.0.0.1:${PORT}/\n`);
  });
}

setupApp().catch((err) => {
  console.error('[AutoML Server] Failed to start:', err);
  process.exit(1);
});

// Clean shutdown
process.on('SIGINT', () => {
  if (flaskProcess) flaskProcess.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  if (flaskProcess) flaskProcess.kill();
  process.exit(0);
});
