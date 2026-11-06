import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRootRouteWithContext, Link, Outlet } from "@tanstack/react-router";
import { Toaster } from "sonner";
import { SignalProvider } from "@/hooks/use-satmod";
export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: Root,
  notFoundComponent: () => (
    <div className="p-12 text-center">
      <h1>Workspace not found</h1>
      <Link to="/" className="text-primary">
        Return to Mission Control
      </Link>
    </div>
  ),
  errorComponent: ({ error, reset }) => (
    <div className="p-12">
      <h1>This workspace could not load</h1>
      <p role="alert">{error.message}</p>
      <button onClick={reset}>Try again</button>
    </div>
  ),
});
function Root() {
  const { queryClient } = Route.useRouteContext();
  return (
    <QueryClientProvider client={queryClient}>
      <SignalProvider>
        <Outlet />
        <Toaster richColors position="bottom-right" />
      </SignalProvider>
    </QueryClientProvider>
  );
}
