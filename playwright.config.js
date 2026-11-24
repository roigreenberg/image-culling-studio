// @ts-check
const fs = require('fs');
const path = require('path');
const { defineConfig } = require('@playwright/test');

const PORT = process.env.E2E_PORT || '8799';

function resolvePython() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const isWin = process.platform === 'win32';
  const venvPy = isWin
    ? path.join(__dirname, '.venv', 'Scripts', 'python.exe')
    : path.join(__dirname, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPy)) return venvPy;
  return isWin ? 'python' : 'python3';
}

module.exports = defineConfig({
  testDir: './tests/e2e',
  timeout: 30000,
  expect: {
    timeout: 8000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
  ],
  webServer: {
    command: `"${resolvePython()}" tests/e2e/run_test_server.py`,
    url: `http://127.0.0.1:${PORT}/api/settings`,
    reuseExistingServer: false,
    timeout: 30000,
    env: {
      E2E_PORT: PORT,
    },
  },
});
