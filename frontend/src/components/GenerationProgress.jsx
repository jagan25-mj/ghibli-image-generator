import { motion } from 'framer-motion';

const GenerationProgress = ({ progress }) => {
  const getPhaseText = (phase) => {
    switch (phase) {
      case 'preparing':
        return 'Preparing generation...';
      case 'upload':
        return 'Uploading image...';
      case 'generating':
        return 'Generating image...';
      case 'complete':
        return 'Complete!';
      default:
        return 'Processing...';
    }
  };

  const getPhaseColor = (phase) => {
    switch (phase) {
      case 'complete':
        return 'bg-green-500';
      case 'upload':
        return 'bg-blue-500';
      case 'generating':
        return 'bg-purple-500';
      default:
        return 'bg-amber-500';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="p-4 rounded-xl bg-black/20 border border-white/10"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">{getPhaseText(progress.phase)}</span>
        {progress.progress > 0 && (
          <span className="text-xs text-muted">{progress.progress}%</span>
        )}
      </div>
      
      {/* Progress Bar */}
      <div className="h-2 bg-black/30 rounded-full overflow-hidden">
        <motion.div
          className={`h-full ${getPhaseColor(progress.phase)} rounded-full`}
          initial={{ width: 0 }}
          animate={{ width: `${progress.progress}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {/* Animated dots for indeterminate progress */}
      {progress.progress === 0 && progress.phase !== 'idle' && (
        <div className="flex justify-center gap-1 mt-3">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-2 h-2 bg-white/50 rounded-full"
              animate={{
                scale: [1, 1.5, 1],
                opacity: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
                delay: i * 0.2,
              }}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default GenerationProgress;