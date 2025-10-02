import { LabIcon } from '@jupyterlab/ui-components';

import jupyternautSvg from '../../style/icons/jupyternaut.svg';
import chatSvgStr from '../../style/icons/chat.svg';

export const jupyternautIcon = new LabIcon({
  name: 'jupyter-ai::jupyternaut',
  svgstr: jupyternautSvg
});

export const chatIcon = new LabIcon({
  name: 'jupyter-ai::chat',
  svgstr: chatSvgStr
});

// this icon is only used in the status bar.
// to configure the icon shown on agent replies in the chat UI, please specify a
// custom `Persona`.
export const Jupyternaut = jupyternautIcon.react;
