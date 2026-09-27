import { spawn, execSync } from 'node:child_process';
import readline from 'node:readline';

console.log('\x1b[1m\x1b[32m%s\x1b[0m', '🚀 Iniciando Ecosistema Completo (MONAI Backend + Astro Frontend)...');

// Limpiar procesos previos si quedaron huérfanos
try {
  execSync('npx astro dev stop 2>/dev/null || true');
} catch (e) {}
try {
  execSync('pkill -f "uvicorn backend.app:app" 2>/dev/null || true');
} catch (e) {}

// 1. Iniciar Servidor FastAPI / MONAI Backend en Puerto 8000
const backendProcess = spawn('uv', [
  'run',
  '--with', 'fastapi,uvicorn,pydantic,numpy,scipy,scikit-image,pillow,torch,open_clip_torch,monai,transformers,einops',
  'python', '-m', 'uvicorn',
  'backend.app:app',
  '--host', '0.0.0.0',
  '--port', '8000'
], {
  stdio: ['inherit', 'pipe', 'pipe']
});

// Prefijo en color magenta para el backend
const rlBackendOut = readline.createInterface({ input: backendProcess.stdout });
rlBackendOut.on('line', (line) => {
  console.log(`\x1b[35m[MONAI API]\x1b[0m ${line}`);
});

const rlBackendErr = readline.createInterface({ input: backendProcess.stderr });
rlBackendErr.on('line', (line) => {
  console.log(`\x1b[35m[MONAI API]\x1b[0m ${line}`);
});

// 2. Iniciar Servidor Astro Frontend en Puerto 4321
const frontendProcess = spawn('npx', ['astro', 'dev', '--ignore-lock'], {
  stdio: ['inherit', 'pipe', 'pipe']
});

// Prefijo en color cian para el frontend
const rlFrontendOut = readline.createInterface({ input: frontendProcess.stdout });
rlFrontendOut.on('line', (line) => {
  console.log(`\x1b[36m[ASTRO WEB]\x1b[0m ${line}`);
});

const rlFrontendErr = readline.createInterface({ input: frontendProcess.stderr });
rlFrontendErr.on('line', (line) => {
  console.log(`\x1b[36m[ASTRO WEB]\x1b[0m ${line}`);
});

// Manejo de apagado limpio al presionar Ctrl+C (SIGINT / SIGTERM)
let isExiting = false;
function shutdown() {
  if (isExiting) return;
  isExiting = true;
  console.log('\n\x1b[33m%s\x1b[0m', '🛑 Deteniendo Backend y Frontend de forma segura...');

  try {
    backendProcess.kill('SIGTERM');
  } catch (e) {}

  try {
    frontendProcess.kill('SIGTERM');
  } catch (e) {}

  try {
    execSync('npx astro dev stop 2>/dev/null || true');
  } catch (e) {}

  try {
    execSync('pkill -f "uvicorn backend.app:app" 2>/dev/null || true');
  } catch (e) {}

  setTimeout(() => {
    try {
      backendProcess.kill('SIGKILL');
      frontendProcess.kill('SIGKILL');
    } catch (e) {}
    process.exit(0);
  }, 600);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
backendProcess.on('exit', (code, sig) => {
  console.log(`[EXIT] Backend process exited: code=${code}, sig=${sig}`);
  if (!isExiting) shutdown();
});
frontendProcess.on('exit', (code, sig) => {
  console.log(`[EXIT] Frontend process exited: code=${code}, sig=${sig}`);
  if (code !== 0 && !isExiting) shutdown();
});
