import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  IconButton,
  MenuItem,
  Select,
  TextField,
  Tooltip
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import DeleteIcon from '@mui/icons-material/Delete';
import Save from '@mui/icons-material/Save';

import { AiService } from '../../handler';
import { useStackingAlert } from '../mui-extras/stacking-alert';

const PARAMETER_TYPES = [
  'string',
  'integer',
  'number',
  'boolean',
  'array',
  'object'
] as const;

type ParameterType = (typeof PARAMETER_TYPES)[number];

/**
 * A model parameter as edited in the UI: a name, a type, and a string value.
 * The string value is coerced to its typed value on save (and back to a string
 * when loading an existing custom model).
 */
type EditableParam = {
  name: string;
  type: ParameterType;
  value: string;
};

/**
 * A custom model as edited in the UI. Mirrors `AiService.CustomModel` but holds
 * parameters as an editable list (order-preserving, with per-row type) rather
 * than a plain object.
 */
type EditableModel = {
  id?: string;
  name: string;
  description: string;
  modelId: string;
  params: EditableParam[];
};

function inferParameterType(value: unknown): ParameterType {
  if (typeof value === 'boolean') {
    return 'boolean';
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? 'integer' : 'number';
  }
  if (Array.isArray(value)) {
    return 'array';
  }
  if (typeof value === 'object' && value !== null) {
    return 'object';
  }
  return 'string';
}

function paramsToEditable(params: Record<string, any>): EditableParam[] {
  return Object.entries(params).map(([name, value]) => {
    const type = inferParameterType(value);
    const asString =
      type === 'array' || type === 'object'
        ? JSON.stringify(value)
        : String(value);
    return { name, type, value: asString };
  });
}

function serviceToEditable(model: AiService.CustomModel): EditableModel {
  return {
    id: model.id,
    name: model.name,
    description: model.description ?? '',
    modelId: model.model_id,
    params: paramsToEditable(model.params ?? {})
  };
}

/**
 * Coerce an editable parameter's string value to its typed value. Throws with a
 * user-facing message when the value doesn't parse for its declared type.
 */
function coerceParamValue(param: EditableParam): any {
  const raw = param.value.trim();
  switch (param.type) {
    case 'integer': {
      const n = Number(raw);
      if (!Number.isInteger(n)) {
        throw new Error(`Parameter '${param.name}' must be an integer.`);
      }
      return n;
    }
    case 'number': {
      const n = Number(raw);
      if (Number.isNaN(n)) {
        throw new Error(`Parameter '${param.name}' must be a number.`);
      }
      return n;
    }
    case 'boolean': {
      const lower = raw.toLowerCase();
      if (lower !== 'true' && lower !== 'false') {
        throw new Error(`Parameter '${param.name}' must be 'true' or 'false'.`);
      }
      return lower === 'true';
    }
    case 'array':
    case 'object': {
      try {
        return JSON.parse(raw);
      } catch {
        throw new Error(`Parameter '${param.name}' must be valid JSON.`);
      }
    }
    default:
      return param.value;
  }
}

/**
 * Convert an editable model back to the `AiService.CustomModel` sent to the
 * server. Throws (via `coerceParamValue`) when a parameter value is invalid.
 */
function editableToService(model: EditableModel): AiService.CustomModel {
  const params: Record<string, any> = {};
  for (const param of model.params) {
    if (!param.name.trim()) {
      continue;
    }
    params[param.name.trim()] = coerceParamValue(param);
  }
  return {
    id: model.id,
    name: model.name.trim(),
    description: model.description.trim() || null,
    model_id: model.modelId.trim(),
    params
  };
}

function emptyModel(): EditableModel {
  return { name: '', description: '', modelId: '', params: [] };
}

/**
 * Editor for a single custom model: its name, description, LiteLLM model ID, and
 * parameters, plus reorder/delete controls.
 */
function CustomModelCard(props: {
  model: EditableModel;
  index: number;
  count: number;
  onChange: (model: EditableModel) => void;
  onMove: (direction: -1 | 1) => void;
  onDelete: () => void;
}): JSX.Element {
  const { model, index, count } = props;

  const updateParam = (
    paramIndex: number,
    field: keyof EditableParam,
    value: string
  ) => {
    props.onChange({
      ...model,
      params: model.params.map((param, i) =>
        i === paramIndex ? { ...param, [field]: value } : param
      )
    });
  };

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <TextField
            label="Model name"
            placeholder="e.g. 'My Hermes'"
            value={model.name}
            onChange={e => props.onChange({ ...model, name: e.target.value })}
            size="small"
            sx={{ flex: 1 }}
          />
          <Tooltip title="Move up">
            <span>
              <IconButton
                size="small"
                aria-label="Move up"
                disabled={index === 0}
                onClick={() => props.onMove(-1)}
              >
                <ArrowUpwardIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Move down">
            <span>
              <IconButton
                size="small"
                aria-label="Move down"
                disabled={index === count - 1}
                onClick={() => props.onMove(1)}
              >
                <ArrowDownwardIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Delete custom model">
            <IconButton
              size="small"
              color="error"
              aria-label="Delete custom model"
              onClick={props.onDelete}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        <TextField
          label="Description (optional)"
          value={model.description}
          onChange={e =>
            props.onChange({ ...model, description: e.target.value })
          }
          size="small"
          fullWidth
        />

        <TextField
          label="Model ID (LiteLLM)"
          placeholder="e.g. 'openai/hermes'"
          value={model.modelId}
          onChange={e => props.onChange({ ...model, modelId: e.target.value })}
          size="small"
          fullWidth
        />

        {model.params.map((param, paramIndex) => (
          <Box
            key={paramIndex}
            sx={{ display: 'flex', gap: 1, alignItems: 'center' }}
          >
            <TextField
              label="Parameter"
              placeholder="e.g. 'temperature'"
              value={param.name}
              onChange={e => updateParam(paramIndex, 'name', e.target.value)}
              size="small"
              sx={{ flex: 1 }}
            />
            <Select
              value={param.type}
              onChange={e => updateParam(paramIndex, 'type', e.target.value)}
              size="small"
              sx={{ flex: 1 }}
            >
              {PARAMETER_TYPES.map(type => (
                <MenuItem key={type} value={type}>
                  {type}
                </MenuItem>
              ))}
            </Select>
            <TextField
              label="Value"
              placeholder="e.g. '0.7'"
              value={param.value}
              onChange={e => updateParam(paramIndex, 'value', e.target.value)}
              size="small"
              sx={{ flex: 1 }}
            />
            <IconButton
              size="small"
              color="error"
              onClick={() =>
                props.onChange({
                  ...model,
                  params: model.params.filter((_, i) => i !== paramIndex)
                })
              }
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Box>
        ))}

        <Button
          variant="outlined"
          size="small"
          startIcon={<AddIcon />}
          onClick={() =>
            props.onChange({
              ...model,
              params: [...model.params, { name: '', type: 'string', value: '' }]
            })
          }
          sx={{ alignSelf: 'flex-start' }}
        >
          Add parameter
        </Button>
      </CardContent>
    </Card>
  );
}

/**
 * The "Custom models" manager for the Jupyternaut settings view. Users create,
 * edit, reorder, and delete custom models here; on save, the full list is
 * persisted (in display order) and appears at the top of the model picker.
 */
export function CustomModelsInput(): JSX.Element {
  const [models, setModels] = useState<EditableModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const alert = useStackingAlert();

  useEffect(() => {
    async function load() {
      try {
        const custom = await AiService.getCustomModels();
        setModels(custom.map(serviceToEditable));
      } catch (error) {
        console.error('Failed to load custom models:', error);
        alert.show('error', 'Failed to load custom models.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const updateModel = (index: number, model: EditableModel) => {
    setModels(prev => prev.map((m, i) => (i === index ? model : m)));
  };

  const moveModel = (index: number, direction: -1 | 1) => {
    setModels(prev => {
      const next = [...prev];
      const target = index + direction;
      if (target < 0 || target >= next.length) {
        return prev;
      }
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  };

  const deleteModel = (index: number) => {
    setModels(prev => prev.filter((_, i) => i !== index));
  };

  const handleSave = async () => {
    // Validate names/model IDs and coerce parameter values before saving.
    const invalid = models.find(m => !m.name.trim() || !m.modelId.trim());
    if (invalid) {
      alert.show(
        'error',
        'Every custom model needs a name and a LiteLLM model ID.'
      );
      return;
    }

    let payload: AiService.CustomModel[];
    try {
      payload = models.map(editableToService);
    } catch (error) {
      const msg =
        error instanceof Error ? error.message : 'An unknown error occurred';
      alert.show('error', msg);
      return;
    }

    setSaving(true);
    try {
      await AiService.saveCustomModels(payload);
      alert.show('success', 'Saved custom models.');
    } catch (error) {
      const msg =
        error instanceof Error ? error.message : 'An unknown error occurred';
      alert.show('error', `Failed to save custom models: ${msg}`);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        Loading custom models...
      </Box>
    );
  }

  return (
    <Box>
      {models.map((model, index) => (
        <CustomModelCard
          key={model.id ?? `new-${index}`}
          model={model}
          index={index}
          count={models.length}
          onChange={m => updateModel(index, m)}
          onMove={direction => moveModel(index, direction)}
          onDelete={() => deleteModel(index)}
        />
      ))}

      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
        <Button
          variant="outlined"
          startIcon={<AddIcon />}
          onClick={() => setModels(prev => [...prev, emptyModel()])}
        >
          Add a custom model
        </Button>
        <Button
          variant="contained"
          startIcon={<Save />}
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save custom models'}
        </Button>
      </Box>
      {alert.jsx}
    </Box>
  );
}
