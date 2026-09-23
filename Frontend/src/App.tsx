import { Component, type ErrorInfo, type ReactNode } from 'react';
import LandingPage from '@/components/LandingPage';
import { ThemeProvider } from '@/components/ThemeProvider';


interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error in AccentSense UI:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div
          className="min-h-screen flex flex-col items-center justify-center p-6 text-center bg-surface text-on-surface"
        >
          <div
            className="max-w-md p-8 rounded-lg border border-outline-variant"
          >
            <h2 className="text-lg font-semibold mb-2">Something went wrong</h2>
            <p
              className="text-sm mb-5 leading-relaxed font-mono text-on-surface-variant"
            >
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <button
              className="btn-primary"
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
            >
              Reload
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function App() {
  return (
    <ThemeProvider>
      <ErrorBoundary>
        <LandingPage />
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App;
