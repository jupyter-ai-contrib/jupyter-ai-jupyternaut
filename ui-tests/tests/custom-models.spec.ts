/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 *
 * End-to-end test of the Jupyternaut custom-model flow:
 *
 *   - open the settings view from the toolbar's Jupyternaut settings button
 *   - create a custom model (name + LiteLLM model ID + a parameter) against the
 *     fake LiteLLM provider, and save it
 *   - see the custom model at the top of the model picker (above "Default")
 *   - add a second custom model and confirm reorder in the picker
 *   - select the custom model, send a message, and get the fake provider's
 *     canned reply
 *
 * The fake LiteLLM provider (registered in the server process; see
 * ../fake_litellm_provider.py) makes this deterministic with no real model or
 * credentials. These need only pass locally for now: CI may be red until the
 * persona-manager awareness API and the migrated persona controls are released.
 */

import { expect, test } from '@jupyterlab/galata';

import {
  CANNED_REPLY,
  FAKE_MODEL_ID,
  resetCustomModels,
  TestHelpers
} from './test-helpers';

test.describe('Jupyternaut custom models', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page, request }) => {
    // Jupyternaut's config is process-global; clear custom models so each test
    // starts from an empty picker.
    await resetCustomModels(request);
    helpers = new TestHelpers({ dir: '', page });
    await helpers.openChat();
    await helpers.selectJupyternaut();
  });

  test('creates a custom model and gets the fake provider reply', async () => {
    // Create a custom model against the fake LiteLLM provider. This opens the
    // Jupyternaut settings view via the toolbar settings button, adds the model,
    // saves, and returns focus to the chat.
    await helpers.createCustomModel({
      name: 'Fake Model',
      modelId: FAKE_MODEL_ID,
      param: { name: 'temperature', value: '0.5', type: 'number' }
    });

    // The custom model appears in the picker, above the built-in "Default"
    // option. (The very first row is the "Default (…)" reset choice; the
    // persona's advertised options follow — custom models first.) Poll, since
    // the model configuration propagates to the frontend over awareness.
    await helpers.waitForModelControl();
    await expect
      .poll(async () => (await helpers.modelOptions()).includes('Fake Model'), {
        timeout: 30000
      })
      .toBe(true);
    const options = await helpers.modelOptions();
    const fakeIdx = options.indexOf('Fake Model');
    const defaultIdx = options.lastIndexOf('Default');
    expect(fakeIdx).toBeGreaterThan(0); // after the reset row
    expect(fakeIdx).toBeLessThan(defaultIdx); // above the built-in Default

    // Select it, send a message, and get the canned reply.
    await helpers.selectModel('Fake Model');
    const reply = await helpers.sendMessage('hello');
    expect(reply).toContain(CANNED_REPLY);
  });

  test('reorders custom models at the top of the picker', async () => {
    await helpers.createCustomModel({
      name: 'Model A',
      modelId: FAKE_MODEL_ID
    });
    await helpers.createCustomModel({
      name: 'Model B',
      modelId: FAKE_MODEL_ID
    });

    await helpers.waitForModelControl();
    await expect
      .poll(async () => (await helpers.modelOptions()).includes('Model B'), {
        timeout: 30000
      })
      .toBe(true);
    let options = await helpers.modelOptions();
    // Both custom models sort above the built-in "Default", in creation order.
    const aBefore = options.indexOf('Model A');
    const bBefore = options.indexOf('Model B');
    const defaultIdx = options.lastIndexOf('Default');
    expect(aBefore).toBeGreaterThan(0);
    expect(bBefore).toBeGreaterThan(aBefore);
    expect(defaultIdx).toBeGreaterThan(bBefore);

    // Reorder: move Model B above Model A in the settings view, then save.
    const view = await helpers.openSettings();
    const bCard = view.locator('.MuiCard-root').nth(1);
    await bCard.getByRole('button', { name: 'Move up' }).click();
    await view.getByRole('button', { name: 'Save custom models' }).click();
    await expect(view.getByText('Saved custom models.')).toBeVisible();
    await helpers.activateChat();

    options = await helpers.modelOptions();
    expect(options.findIndex(o => o.includes('Model B'))).toBeLessThan(
      options.findIndex(o => o.includes('Model A'))
    );
  });
});
