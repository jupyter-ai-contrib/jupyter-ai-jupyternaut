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

    // The custom model appears near the top of the picker. (The very first row
    // is the "Default (…)" reset choice; the persona's advertised options
    // follow — custom models first, then the LiteLLM catalog.) Poll, since the
    // model configuration propagates to the frontend over awareness.
    await helpers.waitForModelControl();
    await expect
      .poll(async () => (await helpers.modelOptions()).includes('Fake Model'), {
        timeout: 30000
      })
      .toBe(true);
    const options = await helpers.modelOptions();
    const fakeIdx = options.indexOf('Fake Model');
    expect(fakeIdx).toBe(1); // right after the reset row, above the catalog

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
    // Both custom models sort to the top (after the reset row, before the
    // catalog), in creation order.
    const aBefore = options.indexOf('Model A');
    const bBefore = options.indexOf('Model B');
    expect(aBefore).toBe(1); // right after the reset row
    expect(bBefore).toBe(2);

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

  test('lists LiteLLM models in the picker, with a custom model on top', async () => {
    // With no custom models yet, the picker still offers the full LiteLLM
    // catalog — a well-known model like `gpt-4o` is present. Poll for the model
    // control (options arrive over awareness).
    await helpers.waitForModelControl();
    await helpers.expectModelOption('gpt-4o');

    // Adding a custom model puts it above the LiteLLM catalog.
    await helpers.createCustomModel({
      name: 'Top Model',
      modelId: FAKE_MODEL_ID
    });
    await expect
      .poll(async () => (await helpers.modelOptions()).includes('Top Model'), {
        timeout: 30000
      })
      .toBe(true);
    const options = await helpers.modelOptions();
    expect(options.indexOf('Top Model')).toBeLessThan(
      options.indexOf('gpt-4o')
    );
  });

  test('shows the settings button to the right of the model selector', async () => {
    await helpers.waitForModelControl();
    await expect(helpers.settingsButtonVisible).toBeVisible();
    await helpers.expectSettingsButtonRightOfModel();
  });

  test('virtualizes the large model menu', async () => {
    // The LiteLLM catalog (~2000 models) far exceeds persona-manager's
    // containment threshold, so the model menu opts into the virtualized path.
    // A model is still directly selectable from the virtualized menu.
    await helpers.waitForModelControl();
    await helpers.expectModelMenuVirtualized();
    await helpers.expectModelOption('gpt-4o');
  });
});
