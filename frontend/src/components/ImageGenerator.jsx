import { useState, useCallback } from 'react';
import { useForm } from 'react-hook-form';
import { motion, AnimatePresence } from 'framer-motion';
import { useConfig } from '../hooks/useConfig';
import { useImageGeneration } from '../hooks/useImageGeneration';
import LoadingSpinner from './LoadingSpinner';
import GenerationForm from './GenerationForm';
import ImagePreview from './ImagePreview';
import GenerationProgress from './GenerationProgress';

const ImageGenerator = () => {
  const { data: config, isLoading: configLoading } = useConfig();
  const { generate, isGenerating, data: generationResult, reset } = useImageGeneration();
  const [progress, setProgress] = useState({ phase: 'idle', progress: 0 });

  const {
    register,
    handleSubmit,
    watch,
    reset: resetForm,
    formState: { errors },
  } = useForm({
    defaultValues: {
      preset: 'balanced',
      prompt: '',
      negative_prompt: '',
      aspect: '3:2',
      guidance_scale: 7.5,
      num_inference_steps: 18,
      strength: 0.6,
      upscale_mode: 'auto',
      upscale_factor: '2',
      seed: '',
      lora: '',
    },
  });

  const initImage = watch('init_image');

  const onSubmit = useCallback(
    (formData) => {
      setProgress({ phase: 'preparing', progress: 0 });

      // Prepare data for API
      const data = {
        ...formData,
        init_image: formData.init_image?.[0] || null,
      };

      generate(
        {
          data,
          onProgress: (progressData) => {
            setProgress(progressData);
          },
        },
        {
          onSuccess: () => {
            setProgress({ phase: 'complete', progress: 100 });
          },
          onError: () => {
            setProgress({ phase: 'idle', progress: 0 });
          },
        }
      );
    },
    [generate]
  );

  const handleClear = useCallback(() => {
    resetForm();
    reset();
    setProgress({ phase: 'idle', progress: 0 });
  }, [resetForm, reset]);

  if (configLoading) {
    return <LoadingSpinner text="Loading configuration..." />;
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left: Form */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass lg:col-span-2 p-6"
      >
        <h2 className="font-semibold text-lg mb-6">Create Image</h2>
        
        <GenerationForm
          register={register}
          handleSubmit={handleSubmit}
          onSubmit={onSubmit}
          errors={errors}
          config={config}
          isGenerating={isGenerating}
          onClear={handleClear}
          initImage={initImage}
        />

        {/* Progress Indicator */}
        <AnimatePresence>
          {isGenerating && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6"
            >
              <GenerationProgress progress={progress} />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.section>

      {/* Right: Preview */}
      <motion.aside
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="glass p-6"
      >
        <h2 className="font-semibold text-lg mb-4">Preview</h2>
        
        <ImagePreview
          result={generationResult}
          isGenerating={isGenerating}
        />
      </motion.aside>
    </div>
  );
};

export default ImageGenerator;