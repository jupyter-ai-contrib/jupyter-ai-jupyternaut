import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { requestAPI } from './handler';

/**
 * Initialization data for the @jupyter-ai/jupyternaut extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/jupyternaut:plugin',
  description: 'Package providing the default AI persona, Jupyternaut, in Jupyter AI.',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension @jupyter-ai/jupyternaut is activated!');

    requestAPI<any>('get-example')
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

export default plugin;
