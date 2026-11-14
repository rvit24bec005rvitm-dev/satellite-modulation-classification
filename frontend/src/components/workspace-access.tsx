import { useCallback, useEffect, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Lock, Satellite } from "lucide-react";
import { apiBase, satmodApi } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import { Field, Message } from "@/components/workspace-ui";

export function WorkspaceAccess({ children }: { children: ReactNode }) {
  const client = useQueryClient();
  const base = apiBase();
  const storageKey = "satmod-access-token:" + base;
  const sessionKey = ["workspace-session", base];
  const [hasToken, setHasToken] = useState(() => Boolean(sessionStorage.getItem(storageKey)));
  const [token, setToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const access = useQuery({
    queryKey: ["workspace-access", base],
    queryFn: ({ signal }) => satmodApi.access(signal),
    retry: false,
    staleTime: 30000,
  });
  const session = useQuery({
    queryKey: sessionKey,
    queryFn: ({ signal }) => satmodApi.session(signal),
    enabled: access.data?.tokenRequired === true && hasToken,
    retry: false,
    staleTime: 30000,
  });
  const unlock = useMutation({
    mutationFn: async (value: string) => {
      const result = await satmodApi.session(undefined, value);
      if (!result.authorized) throw new Error("Workspace access could not be verified.");
      return result;
    },
    onSuccess: (result, value) => {
      sessionStorage.setItem(storageKey, value);
      client.setQueryData(sessionKey, result);
      setHasToken(true);
      setToken("");
    },
  });
  const resetUnlock = unlock.reset;
  const lock = useCallback(() => {
    sessionStorage.removeItem(storageKey);
    setHasToken(false);
    setToken("");
    resetUnlock();
    // Unmounting the children also clears capture/selection context. Keep only
    // public access discovery; private caches must not survive a locked session.
    void client.cancelQueries({ predicate: (query) => query.queryKey[0] !== "workspace-access" });
    client.removeQueries({ predicate: (query) => query.queryKey[0] !== "workspace-access" });
  }, [client, storageKey, resetUnlock]);
  useEffect(() => {
    window.addEventListener("satmod:workspace-unauthorized", lock);
    return () => window.removeEventListener("satmod:workspace-unauthorized", lock);
  }, [lock]);

  if (
    access.isPending ||
    (access.data?.tokenRequired && hasToken && session.isPending && session.fetchStatus !== "idle")
  ) {
    return (
      <main
        id="main-content"
        tabIndex={-1}
        className="flex min-h-screen items-center justify-center bg-background p-6 text-foreground"
        role="status"
      >
        Checking workspace access…
      </main>
    );
  }
  // Backend authorization remains authoritative. The independent educational
  // globe/theory stay usable when access discovery is offline or an older API
  // does not yet support the public discovery endpoint.
  if (!access.data || (access.data.configured && !access.data.tokenRequired))
    return <>{children}</>;

  if (access.data.configured && session.data?.authorized && hasToken) {
    return (
      <>
        <div className="fixed bottom-3 right-3 z-50">
          <Button variant="outline" size="sm" onClick={lock}>
            <Lock className="size-3" />
            Lock workspace
          </Button>
        </div>
        {children}
      </>
    );
  }
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-5 py-12 text-foreground">
      <main
        id="main-content"
        tabIndex={-1}
        className="w-full max-w-md border border-border bg-card p-6 shadow-xl sm:p-9"
        aria-labelledby="workspace-access-title"
      >
        <div className="mb-8 flex items-center gap-3 text-primary">
          <Satellite aria-hidden="true" />
          <span className="font-mono text-sm tracking-widest">SATMODCLASS</span>
        </div>
        <p className="mb-3 font-mono text-[10px] tracking-widest text-muted-foreground">
          PROTECTED RESEARCH WORKSPACE
        </p>
        <h1 id="workspace-access-title" className="mb-4 font-display text-2xl font-semibold">
          {access.data.configured ? "Unlock your workspace" : "Workspace setup required"}
        </h1>
        {access.data.configured ? (
          <>
            <p className="mb-6 text-sm leading-6 text-muted-foreground">
              Enter the access token provided by your project owner. Your access is kept in this
              browser tab.
            </p>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (token && !unlock.isPending) unlock.mutate(token);
              }}
            >
              <Field label="Workspace access token">
                <input
                  className="control"
                  type={showToken ? "text" : "password"}
                  autoComplete="current-password"
                  spellCheck={false}
                  autoCapitalize="none"
                  required
                  maxLength={2048}
                  value={token}
                  disabled={unlock.isPending}
                  onChange={(event) => setToken(event.target.value)}
                />
              </Field>
              <label className="mb-5 flex items-center gap-2 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={showToken}
                  onChange={(event) => setShowToken(event.target.checked)}
                />
                Show token
              </label>
              <Message error={unlock.error ?? (hasToken ? session.error : null)} />
              <Button type="submit" className="w-full" disabled={!token || unlock.isPending}>
                {unlock.isPending ? "Verifying access…" : "Unlock workspace"}
              </Button>
            </form>
            <p className="mt-5 text-xs leading-6 text-muted-foreground">
              This is a shared operator workspace. Recordings and history are visible to operators
              with the same access token.
            </p>
          </>
        ) : (
          <p role="alert" className="text-sm leading-7">
            The server owner must configure a valid workspace access token before this application
            can accept public connections. Check the hosting service’s secret settings, then reload
            this page.
          </p>
        )}
      </main>
    </div>
  );
}
