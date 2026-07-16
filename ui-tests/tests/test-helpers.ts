/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 *
 * Helpers for the Jupyternaut E2E suites.
 *
 * These drive the chat UI (the persona controls come from `@jupyter-ai/acp-client`,
 * which owns the input toolbar) and the Jupyternaut settings view (where custom
 * models are defined). The model backend is a fake LiteLLM provider registered
 * in the server process — see `../fake_litellm_provider.py` — so a custom model
 * created against model ID `fake-litellm/echo` streams a deterministic reply.
 */

import { expect, IJupyterLabPageFixture } from '@jupyterlab/galata';
import { APIRequestContext, Locator } from '@playwright/test';
import { UUID } from '@lumino/coreutils';

/** The LiteLLM model ID of the fake provider (see fake_litellm_provider.py). */
export const FAKE_MODEL_ID = 'fake-litellm/echo';

/** The canned reply the fake provider streams (see fake_litellm_provider.py). */
export const CANNED_REPLY = 'Hello from the fake LiteLLM provider!';

/** Jupyternaut's display name in the persona picker. */
export const JUPYTERNAUT_NAME = 'Jupyternaut';

// --- Selectors -------------------------------------------------------------

// Persona controls (from @jupyter-ai/persona-manager, which owns the input
// toolbar). The settings button is contributed by @jupyter-ai/jupyternaut
// through persona-manager's control registry.
const PICKER = '.jp-jai-personaControls-persona-btn';
const SETTINGS_BTN = '.jp-ai-jupyternaut-settingsButton';
const VISIBLE_CONTROL_BTN =
  '.jp-jai-personaControls-controls > .jp-jai-personaControls-control-btn';
// A control dropdown's MUI menu popover (a page-root portal) and its per-option
// name span. See `menuAnchorProps` / `ChoiceMenuItem` in persona-manager.
const MENU_PAPER = '.jp-jai-controlMenu-paper';
const MENU_NAME = '.jp-jai-controlMenu-name';

const INPUT = '.jp-chat-input-container';
const MESSAGE = '.jp-chat-rendered-message';

// The Jupyternaut settings view (opened by the settings button).
const SETTINGS_VIEW = '.jp-ai-ChatSettings';

const TIMEOUT = 30000;

/**
 * Clear all custom models via the config REST API, so each test starts from an
 * empty picker regardless of what earlier tests created. Jupyternaut's config
 * is process-global (one file), so state persists across tests on the shared
 * server without this reset. Call from `beforeEach` with the `request` fixture.
 */
export async function resetCustomModels(
  request: APIRequestContext
): Promise<void> {
  const response = await request.post('/api/jupyternaut/config', {
    data: { custom_models: [] }
  });
  if (!response.ok()) {
    throw new Error(
      `Failed to reset custom models: ${response.status()} ${await response.text()}`
    );
  }
}

/**
 * Per-test helper bound to one suite directory and one page. Drives the chat UI
 * and the Jupyternaut settings view.
 */
export class TestHelpers {
  readonly dir: string;
  readonly page: IJupyterLabPageFixture;
  private _chat: Locator | null = null;
  private _chatTab: string | null = null;

  constructor(options: { dir: string; page: IJupyterLabPageFixture }) {
    this.dir = options.dir;
    this.page = options.page;
  }

  /** The current chat panel (throws if `openChat` hasn't run). */
  get chat(): Locator {
    if (!this._chat) {
      throw new Error('Call openChat() first.');
    }
    return this._chat;
  }

  /** Create and open a chat under this suite's directory. */
  async openChat(): Promise<Locator> {
    const filepath = `${this.dir}/chat-${UUID.uuid4()}.chat`;
    await this.page.filebrowser.contents.uploadContent('{}', 'text', filepath);
    await this.page.evaluate(async (name: string) => {
      await window.jupyterapp.commands.execute('jupyterlab-chat:open', {
        filepath: name
      });
    }, filepath);
    const tab = filepath.split('/').pop()!;
    await this.page.waitForCondition(async () =>
      this.page.activity.isTabActive(tab)
    );
    this._chatTab = tab;
    this._chat = (await this.page.activity.getPanelLocator(tab)) as Locator;
    return this._chat;
  }

  /**
   * Re-activate the chat tab and wait for it to be foremost. Opening the
   * Jupyternaut settings view adds a main-area widget that backgrounds the chat
   * (hiding its toolbar), so call this after using the settings view before
   * interacting with the toolbar again.
   */
  async activateChat(): Promise<void> {
    if (!this._chatTab) {
      throw new Error('Call openChat() first.');
    }
    await this.page.activity.activateTab(this._chatTab);
    await this.page.waitForCondition(async () =>
      this.page.activity.isTabActive(this._chatTab!)
    );
  }

  /** Select the Jupyternaut persona from the picker and wait for it to take. */
  async selectJupyternaut(): Promise<void> {
    const picker = this.chat.locator(PICKER);
    await expect(picker).toBeVisible({ timeout: TIMEOUT });
    await picker.click();
    await this.page.getByRole('menuitem', { name: JUPYTERNAUT_NAME }).click();
    await expect(picker).toContainText(JUPYTERNAUT_NAME);
  }

  /** The Jupyternaut settings button in the toolbar. */
  get settingsButton(): Locator {
    return this.chat.locator(SETTINGS_BTN);
  }

  /**
   * Click the toolbar's Jupyternaut settings button and wait for the settings
   * view to open. Returns the settings-view root (page-scoped: it opens in the
   * main area / a side panel, not inside the chat).
   */
  async openSettings(): Promise<Locator> {
    await expect(this.settingsButton).toBeVisible({ timeout: TIMEOUT });
    await this.settingsButton.click();
    const view = this.page.locator(SETTINGS_VIEW);
    await expect(view.first()).toBeVisible({ timeout: TIMEOUT });
    return view.first();
  }

  /**
   * Create a custom model and save it: open the settings view, add a card, fill
   * name/model ID (and one parameter if given), click Save, then return focus to
   * the chat. Self-contained so it can be called repeatedly.
   */
  async createCustomModel(options: {
    name: string;
    modelId: string;
    param?: { name: string; value: string; type?: string };
  }): Promise<void> {
    const view = await this.openSettings();
    await view.getByRole('button', { name: 'Add a custom model' }).click();

    const card = view.locator('.MuiCard-root').last();
    await card.getByLabel('Model name').fill(options.name);
    await card.getByLabel('Model ID (LiteLLM)').fill(options.modelId);

    if (options.param) {
      await card.getByRole('button', { name: 'Add parameter' }).click();
      await card.getByLabel('Parameter').last().fill(options.param.name);
      // Set the parameter type (defaults to "string" in the UI) so numeric
      // params are coerced on save; the type is a MUI Select combobox.
      if (options.param.type) {
        await card.getByRole('combobox').last().click();
        await this.page
          .getByRole('option', { name: options.param.type, exact: true })
          .click();
      }
      await card.getByLabel('Value').last().fill(options.param.value);
    }

    await view.getByRole('button', { name: 'Save custom models' }).click();
    await expect(view.getByText('Saved custom models.')).toBeVisible({
      timeout: TIMEOUT
    });
    // Return focus to the chat so its toolbar (and the model picker) is visible.
    await this.activateChat();
  }

  /** The visible model-selector control button in the toolbar. */
  get modelControl(): Locator {
    return this.chat.locator(`${VISIBLE_CONTROL_BTN}[title="Model"]`);
  }

  /** Wait for the model selector to appear in the toolbar. */
  async waitForModelControl(): Promise<void> {
    await expect(this.modelControl).toBeVisible({ timeout: TIMEOUT });
  }

  /**
   * Open the model picker and return the option names, in order. Scoped to the
   * control's MUI menu popover (a page-root portal) and read from the option
   * name spans, so this excludes unrelated menuitems (e.g. the top menu bar).
   *
   * The first entry is always the "Default (…)" reset row (selection = the
   * persona's default); the persona's advertised options — custom models first,
   * then the built-in "Default" entry — follow.
   */
  async modelOptions(): Promise<string[]> {
    await this.modelControl.click();
    const menu = this.page.locator(MENU_PAPER);
    await expect(menu).toBeVisible({ timeout: TIMEOUT });
    const names = await menu.locator(MENU_NAME).allTextContents();
    // Close the menu without changing the selection.
    await this.page.keyboard.press('Escape');
    return names.map(l => l.trim());
  }

  /** Select a model option (by its visible name) from the model picker. */
  async selectModel(name: string): Promise<void> {
    await this.modelControl.click();
    const menu = this.page.locator(MENU_PAPER);
    await expect(menu).toBeVisible({ timeout: TIMEOUT });
    await menu.getByRole('menuitem', { name, exact: true }).click();
  }

  /**
   * Send a message and return the text of the resulting reply (the latest
   * rendered message once the human message + reply have both rendered).
   */
  async sendMessage(text: string): Promise<string> {
    const before = await this.chat.locator(MESSAGE).count();
    await this.chat
      .locator(INPUT)
      .getByRole('combobox')
      .pressSequentially(text);
    await this.chat.locator(`${INPUT} .jp-chat-send-button`).click();
    await expect
      .poll(async () => this.chat.locator(MESSAGE).count(), {
        timeout: TIMEOUT
      })
      .toBeGreaterThanOrEqual(before + 2);
    return (await this.chat.locator(MESSAGE).last().textContent()) ?? '';
  }
}
