import { memo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import LoadingSpinner from './LoadingSpinner';

const ImagePreview = memo(({ result, isGenerating }) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const imageUrl = result?.image_url;
  const metadata = result?.metadata;

  return (
    <div className="space-y-4">
      {/* Preview Container */}
      <div className="preview-container">
        <AnimatePresence mode="wait">
          {isGenerating ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <LoadingSpinner size="medium" text="Generating..." />
            </motion.div>
          ) : imageUrl && !imageError ? (
            <motion.img
              key={imageUrl}
              src={imageUrl}
              alt="Generated image"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ 
                opacity: imageLoaded ? 1 : 0, 
                scale: imageLoaded ? 1 : 0.95 
              }}
              transition={{ duration: 0.3 }}
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageError(true)}
              loading="lazy"
            />
          ) : imageError ? (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center p-4"
            >
              <div className="text-4xl mb-2">⚠️</div>
              <p className="text-red-400 text-sm">Failed to load image</p>
            </motion.div>
          ) : (
            <motion.span
              key="placeholder"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-muted text-sm"
            >
              Your image will appear here
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Action Buttons */}
      {imageUrl && !isGenerating && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex gap-3"
        >
          <a
            href={imageUrl}
            className="btn-secondary flex-1 text-center"
            download
          >
            📥 Download
          </a>
          <a
            href={imageUrl}
            className="btn-secondary flex-1 text-center"
            target="_blank"
            rel="noopener noreferrer"
          >
            🔍 Open
          </a>
        </motion.div>
      )}

      {/* Metadata */}
      {metadata && !isGenerating && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-xs text-slate-300 space-y-2 p-4 rounded-lg bg-black/20 border border-white/5"
        >
          <div className="font-semibold text-slate-200 mb-2">Parameters used</div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-muted">Mode:</span> {metadata.mode}
            </div>
            <div>
              <span className="text-muted">Preset:</span> {metadata.preset}
            </div>
            <div>
              <span className="text-muted">Guidance:</span> {metadata.guidance}
            </div>
            <div>
              <span className="text-muted">Steps:</span> {metadata.steps}
            </div>
            <div>
              <span className="text-muted">Aspect:</span> {metadata.aspect}
            </div>
            <div>
              <span className="text-muted">Upscale:</span> {metadata.upscale_mode} ×
              {metadata.upscale_factor}
            </div>
          </div>
          <div>
            <span className="text-muted">Seed:</span>{' '}
            <span className="font-mono text-xs">{metadata.seed}</span>
          </div>
          {metadata.lora !== '—' && (
            <div>
              <span className="text-muted">LoRA:</span> {metadata.lora}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
});

ImagePreview.displayName = 'ImagePreview';

export default ImagePreview;