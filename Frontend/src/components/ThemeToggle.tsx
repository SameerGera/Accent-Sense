import { useTheme } from '../hooks/useTheme';
import { motion, AnimatePresence } from 'framer-motion';

/** Stitch-style bordered toggle: icon + "Dark"/"Light" label. */
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';

  return (
    <motion.button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      className="btn-secondary !h-[30px] !px-2.5 !gap-1.5 !text-label-code overflow-hidden"
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.96 }}
      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={isDark ? 'light' : 'dark'}
          className="material-symbols-outlined !text-[16px]"
          aria-hidden="true"
          initial={{ rotate: -90, opacity: 0 }}
          animate={{ rotate: 0, opacity: 1 }}
          exit={{ rotate: 90, opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {isDark ? 'light_mode' : 'dark_mode'}
        </motion.span>
      </AnimatePresence>
      <span className="hidden sm:inline">{isDark ? 'Light' : 'Dark'}</span>
    </motion.button>
  );
}
