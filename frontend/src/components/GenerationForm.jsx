import { memo, useState } from 'react';
import { motion } from 'framer-motion';

const GenerationForm = memo(({
  register,
  handleSubmit,
  onSubmit,
  errors,
  config,
  isGenerating,
  onClear,
  initImage,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {/* Preset Selector */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-semibold mb-2">Preset</label>
          <select
            {...register('preset')}
            className="field"
            disabled={isGenerating}
          >
            {config?.presets?.map((preset) => (
              <option key={preset.value} value={preset.value}>
                {preset.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-muted mt-1">
            Pick once; you can override parameters below
          </p>
        </div>
      </div>

      {/* Prompt */}
      <div>
        <label className="block text-sm font-semibold mb-2">
          Prompt <span className="text-red-500">*</span>
        </label>
        <textarea
          {...register('prompt', { required: 'Prompt is required' })}
          className="field resize-none"
          rows="3"
          placeholder="A cozy village at dusk with lanterns, whimsical rooftops, soft painterly style..."
          disabled={isGenerating}
        />
        {errors.prompt && (
          <p className="text-red-400 text-xs mt-1">{errors.prompt.message}</p>
        )}
      </div>

      {/* Initial Image Upload */}
      <div>
        <label className="block text-sm font-semibold mb-2">
          Initial image (optional)
        </label>
        <input
          type="file"
          {...register('init_image')}
          accept="image/*"
          className="field"
          disabled={isGenerating}
        />
        {initImage?.[0] && (
          <p className="text-xs text-green-400 mt-1">
            ✓ {initImage[0].name} selected
          </p>
        )}
      </div>

      {/* Advanced Settings Toggle */}
      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="btn-secondary text-sm px-4 py-2 rounded-lg"
        disabled={isGenerating}
      >
        {showAdvanced ? '▼' : '▶'} Advanced Settings
      </button>

      {/* Advanced Settings */}
      {showAdvanced && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="space-y-6"
        >
          {/* Generation Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                Strength (img2img)
              </label>
              <input
                type="number"
                step="0.05"
                min="0.05"
                max="1.0"
                {...register('strength')}
                className="field"
                disabled={isGenerating}
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">
                Guidance
              </label>
              <input
                type="number"
                step="0.1"
                min="1"
                max="20"
                {...register('guidance_scale')}
                className="field"
                disabled={isGenerating}
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">
                Steps
              </label>
              <input
                type="number"
                min="6"
                max="60"
                {...register('num_inference_steps')}
                className="field"
                disabled={isGenerating}
              />
            </div>
          </div>

          {/* Aspect & Seed */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                Aspect (text-to-image)
              </label>
              <select
                {...register('aspect')}
                className="field"
                disabled={isGenerating}
              >
                {config?.aspects?.map((aspect) => (
                  <option key={aspect.value} value={aspect.value}>
                    {aspect.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">
                Seed (optional)
              </label>
              <input
                type="number"
                {...register('seed')}
                className="field"
                placeholder="Random if empty"
                disabled={isGenerating}
              />
            </div>
          </div>

          {/* Negative Prompt & LoRA */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                Negative prompt
              </label>
              <input
                {...register('negative_prompt')}
                className="field"
                placeholder="low quality, text, watermark"
                disabled={isGenerating}
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">
                LoRA (optional)
              </label>
              <input
                {...register('lora')}
                className="field"
                placeholder="local path or HF repo id"
                disabled={isGenerating}
              />
            </div>
          </div>

          {/* Upscale Settings */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                Upscale mode
              </label>
              <select
                {...register('upscale_mode')}
                className="field"
                disabled={isGenerating}
              >
                {config?.upscale_modes?.map((mode) => (
                  <option key={mode.value} value={mode.value}>
                    {mode.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">
                Upscale factor
              </label>
              <select
                {...register('upscale_factor')}
                className="field"
                disabled={isGenerating}
              >
                {config?.upscale_factors?.map((factor) => (
                  <option key={factor.value} value={factor.value}>
                    {factor.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-end">
              <p className="text-xs text-muted pb-3">
                2× is a great speed/quality trade-off
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center gap-3 pt-4">
        <button
          type="submit"
          className="btn"
          disabled={isGenerating}
        >
          {isGenerating ? (
            <span className="flex items-center gap-2">
              <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Generating...
            </span>
          ) : (
            'Generate'
          )}
        </button>
        <button
          type="button"
          onClick={onClear}
          className="btn btn-secondary"
          disabled={isGenerating}
        >
          Clear
        </button>
      </div>
    </form>
  );
});

GenerationForm.displayName = 'GenerationForm';

export default GenerationForm;