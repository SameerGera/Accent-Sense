import { Component, type ErrorInfo, type ReactNode } from 'react';
import LandingPage from '@/components/LandingPage';

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
        <div className="min-h-screen bg-[#070A1F] text-[#F1F0FB] flex flex-col items-center justify-center p-6 text-center">
          <div className="card max-w-md p-8 border border-rose-500/30">
            <div className="w-12 h-12 rounded-xl bg-rose-500/20 text-rose-400 flex items-center justify-center mx-auto mb-4 font-bold text-xl">
              !
            </div>
            <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
            <p className="text-xs text-[#9CA3AF] mb-5 leading-relaxed font-mono">
              {this.state.error?.message || 'An unexpected rendering error occurred in the application.'}
            </p>
            <button
              className="btn-primary px-5 py-2 text-sm mx-auto"
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
            >
              Reload Application
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
    <ErrorBoundary>
      <LandingPage />
    </ErrorBoundary>
  );
}

export default App;
