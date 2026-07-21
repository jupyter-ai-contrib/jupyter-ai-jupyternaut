/**
 * Configuration for Playwright using default from @jupyterlab/galata.
 *
 * A single test server serves every suite. The server registers a fake LiteLLM
 * provider (see fake_litellm_provider.py) and isolates Jupyternaut's config to a
 * temp dir (see jupyter_server_test_config.py), so suites are deterministic and
 * need no real model or credentials.
 */
const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

// Random port so a run doesn't collide with a dev server (or another run) on a
// fixed port. Playwright re-`require`s this config in each worker, so compute it
// once and pin it into the environment — a fresh random value per reload would
// desync the server's port from the port the test workers connect to.
if (!process.env.JAI_TEST_PORT) {
  process.env.JAI_TEST_PORT = String(8989 + Math.floor(Math.random() * 900));
}
const PORT = Number(process.env.JAI_TEST_PORT);

module.exports = {
  ...baseConfig,
  use: { ...(baseConfig.use || {}), baseURL: `http://localhost:${PORT}` },
  webServer: {
    command: `jlpm start --ServerApp.port=${PORT}`,
    url: `http://localhost:${PORT}/lab`,
    timeout: 120 * 1000,
    // Never reuse an already-running server: reusing an unrelated dev server
    // would leave the fake-provider / config-isolation setup unapplied. Free the
    // port before running locally.
    reuseExistingServer: false
  }
};
