import { createContext, useContext } from 'react';

export type Theme = 'light' | 'dark';

export interface ThemeState {
  theme: Theme;
  toggle: () => void;
}

export const ThemeContext = createContext<ThemeState>({
  theme: 'light',
  toggle: () => {},
});

export function useTheme(): ThemeState {
  return useContext(ThemeContext);
}
