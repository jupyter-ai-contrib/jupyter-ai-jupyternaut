import { INotebookShell } from '@jupyter-notebook/application';
import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import {
  IWidgetTracker,
  ReactWidget,
  IThemeManager,
  MainAreaWidget,
  ICommandPalette
} from '@jupyterlab/apputils';
import { IMessageFooterRegistry } from '@jupyter/chat';
import { IDocumentWidget } from '@jupyterlab/docregistry';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { SingletonLayout, Widget } from '@lumino/widgets';

import { StopButton } from './components/message-footer/stop-button';
//import { completionPlugin } from './completions';
import { buildErrorWidget } from './widgets/chat-error';
import { buildAiSettings } from './widgets/settings-widget';
import { statusItemPlugin } from './status';
import { IJaiCompletionProvider } from './tokens';
import { requestAPI } from './handler';

export type DocumentTracker = IWidgetTracker<IDocumentWidget>;
export namespace CommandIDs {
  /**
   * Command to open the AI settings.
   */
  export const openAiSettings = '@jupyter-ai/jupyternaut:open-settings';
}

/**
 * Initialization data for the @jupyter-ai/jupyternaut extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/jupyternaut:plugin',
  description:
    'Package providing the default AI persona, Jupyternaut, in Jupyter AI.',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension @jupyter-ai/jupyternaut is activated!');

    requestAPI<any>('api/jupyternaut/get-example')
      .then(data => {
        console.log(data);
      })
      .catch(reason => {
        console.error(
          `The jupyter_ai_jupyternaut server extension appears to be missing.\n${reason}`
        );
      });
  }
};

const jupyternautSettingsPlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/jupyternaut:settings',
  autoStart: true,
  requires: [IRenderMimeRegistry],
  optional: [
    ICommandPalette,
    IThemeManager,
    IJaiCompletionProvider,
    INotebookShell
  ],
  activate: async (
    app: JupyterFrontEnd,
    rmRegistry: IRenderMimeRegistry,
    palette: ICommandPalette | null,
    themeManager: IThemeManager | null,
    completionProvider: IJaiCompletionProvider | null,
    notebookShell: INotebookShell | null
  ) => {
    const openInlineCompleterSettings = () => {
      app.commands.execute('settingeditor:open', {
        query: 'Inline Completer'
      });
    };

    // Create a AI settings widget.
    let aiSettings: Widget;
    let settingsWidget: ReactWidget;
    try {
      settingsWidget = buildAiSettings(
        themeManager,
        rmRegistry,
        completionProvider,
        openInlineCompleterSettings
      );
    } catch (e) {
      // TODO: Do better error handling here.
      console.error(e);
      settingsWidget = buildErrorWidget(themeManager);
    }

    // Add a command to open settings widget in main area.
    app.commands.addCommand(CommandIDs.openAiSettings, {
      execute: () => {
        if (!aiSettings || aiSettings.isDisposed) {
          if (notebookShell) {
            aiSettings = new Widget();
            const layout = new SingletonLayout();
            aiSettings.layout = layout;
            layout.widget = settingsWidget;
          } else {
            aiSettings = new MainAreaWidget({ content: settingsWidget });
          }
          aiSettings.id = '@jupyter-ai/jupyternaut:settings';
          aiSettings.title.label = 'Jupyternaut settings';
          aiSettings.title.caption = 'Jupyternaut settings';
          aiSettings.title.closable = true;
        }
        if (!aiSettings.isAttached) {
          app?.shell.add(aiSettings, notebookShell ? 'right' : 'main');
        }
        app.shell.activateById(aiSettings.id);
      },
      label: 'Jupyternaut settings'
    });

    if (palette) {
      palette.addItem({
        category: 'jupyter-ai',
        command: CommandIDs.openAiSettings
      });
    }
  }
};

const stopButtonPlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/jupyternaut:stop-button',
  autoStart: true,
  requires: [IMessageFooterRegistry],
  activate: (app: JupyterFrontEnd, registry: IMessageFooterRegistry) => {
    registry.addSection({
      component: StopButton,
      position: 'center'
    });
  }
};

export default [
  plugin,
  jupyternautSettingsPlugin,
  // webComponentsPlugin,
  stopButtonPlugin,
  // completionPlugin,
  statusItemPlugin
];
