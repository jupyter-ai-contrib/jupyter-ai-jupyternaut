import React from 'react';
import { Button } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import { CommandRegistry } from '@lumino/commands';

const SETTINGS_BUTTON_CLASS = 'jp-ai-jupyternaut-settingsButton';

/**
 * A settings button contributed to the persona controls in the chat input
 * toolbar, shown when Jupyternaut is the selected persona. Clicking it opens
 * Jupyternaut's settings view, where users define custom models.
 *
 * The persona controls are select-only, which is a regression for Jupyternaut's
 * precise per-field model configuration, so Jupyternaut surfaces this button as
 * the entry point to its own settings view.
 *
 * Bound to the app command registry via a closure in the plugin, since the
 * persona-manager control props don't carry it.
 */
export function makeSettingsButton(
  commands: CommandRegistry,
  commandId: string
): React.FunctionComponent {
  return function JupyternautSettingsButton(): JSX.Element {
    return (
      <Button
        className={SETTINGS_BUTTON_CLASS}
        size="small"
        variant="text"
        disableRipple
        onClick={() => {
          void commands.execute(commandId);
        }}
        title="Open Jupyternaut settings"
        aria-label="Open Jupyternaut settings"
      >
        <SettingsIcon fontSize="small" />
      </Button>
    );
  };
}
