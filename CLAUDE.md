# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **jupyter_ai_jupyternaut**, a JupyterLab extension that provides the default AI persona "Jupyternaut" for Jupyter AI. It consists of:

- **Python package**: `jupyter_ai_jupyternaut` (server extension)
- **NPM package**: `@jupyter-ai/jupyternaut` (frontend extension)
- **Entry point**: Provides the Jupyternaut persona via the `jupyter_ai.personas` entry point

## Development Commands

### Setup

```bash
micromamba install uv jupyterlab nodejs=22
jlpm
jlpm dev:install
```

### Build Commands

- `jlpm build` - Build TypeScript sources with source maps
- `jlpm build:prod` - Production build (clean + lib + labextension)
- `jlpm build:lib` - Build TypeScript library only
- `jlpm build:labextension` - Build JupyterLab extension

### Development

- `jlpm watch` - Watch source files and auto-rebuild
- `jlpm dev:install` - Full development installation
- `jlpm dev:uninstall` - Remove development installation

### Testing

- **Frontend**: `jlpm test` (Jest)
- **Backend**: `pytest -vv -r ap --cov jupyter_ai_jupyternaut`
- **UI Tests**: See `ui-tests/README.md` (Playwright/Galata)

### Code Quality

- `jlpm lint` - Run all linters (stylelint + prettier + eslint)
- `jlpm lint:check` - Check without fixing
- Individual tools: `jlpm eslint`, `jlpm prettier`, `jlpm stylelint`

### Clean Commands

- `jlpm clean` - Clean lib directory
- `jlpm clean:all` - Clean everything including labextension

## Architecture

### Core Components

1. **JupyternautPersona** (`jupyter_ai_jupyternaut/jupyternaut/jupyternaut.py`):
   - Main persona class extending `BasePersona`
   - Handles chat model configuration and message processing
   - Integrates with `litellm` for LLM completions
   - Manages conversation history and context

2. **Frontend Plugins** (`src/index.ts`):
   - Main plugin: Basic extension activation
   - Settings plugin: AI configuration interface
   - Stop button plugin: Message interaction controls
   - Status plugin: Extension status management

3. **Configuration System**:
   - Uses `jupyter_ai_persona_manager` for persona management
   - Chat model configuration via settings UI
   - Server-side config management and REST APIs

### Key Dependencies

- **Backend**: `jupyter_server`, `pydantic>=2.10.0`, `litellm>=1.73`, `jupyterlab-chat>=0.17.0`
- **Frontend**: `@jupyter/chat`, `@mui/material`, `@emotion/react`, JupyterLab 4.x

### File Structure

- `jupyter_ai_jupyternaut/` - Python package (server extension)
  - `jupyternaut/` - Core persona implementation
  - `config/` - Configuration management
  - `models/` - REST API models
  - `secrets/` - Secret management
- `src/` - TypeScript source (frontend extension)
- `lib/` - Compiled JavaScript output
- `style/` - CSS styling

### Development Notes

- Extension follows JupyterLab 4.x plugin architecture
- Uses `hatch` for Python packaging and `jlpm` (yarn) for Node.js
- Frontend built with React/TypeScript and Material-UI
- Backend provides REST APIs for configuration and model management
- Entry point registration allows automatic persona discovery by Jupyter AI

### Code Style

- TypeScript: ESLint with Prettier, single quotes, no trailing commas
- Python: Standard formatting (details in pyproject.toml)
- Interface naming: Must start with `I` and use PascalCase
- CSS: Stylelint with standard config
