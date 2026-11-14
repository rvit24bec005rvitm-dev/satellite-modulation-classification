// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkspaceAccess } from "@/components/workspace-access";
import { satmodApi } from "@/lib/satmod-api";
import { operationDownload } from "@/lib/operations-api";
import { researchDownload } from "@/lib/research-api";

const clients: QueryClient[] = [];
const required = { tokenRequired: true, configured: true, publicMode: true, sharedWorkspace: true };
function mount() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <WorkspaceAccess>
        <div>Private research results</div>
      </WorkspaceAccess>
    </QueryClientProvider>,
  );
  return client;
}
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});
afterEach(() => {
  cleanup();
  clients.forEach((client) => client.clear());
  clients.length = 0;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Shared workspace access", () => {
  it.each([
    ["workspace", operationDownload],
    ["research", researchDownload],
  ] as const)(
    "locks private content when a %s export receives HTTP 401",
    async (_kind, download) => {
      sessionStorage.setItem("satmod-access-token:", "revoked-unit-token");
      vi.spyOn(satmodApi, "access").mockResolvedValue(required);
      vi.spyOn(satmodApi, "session").mockResolvedValue({ authorized: true, sharedWorkspace: true });
      vi.stubGlobal(
        "fetch",
        vi
          .fn()
          .mockResolvedValue(
            new Response(JSON.stringify({ detail: "Workspace access token required." }), {
              status: 401,
            }),
          ),
      );
      mount();
      await screen.findByText("Private research results");
      await act(async () => {
        await expect(download("/fixture-export", "fixture.json")).rejects.toThrow();
      });
      await screen.findByRole("heading", { name: "Unlock your workspace" });
      expect(screen.queryByText("Private research results")).not.toBeInTheDocument();
      expect(sessionStorage.getItem("satmod-access-token:")).toBeNull();
    },
  );
  it("keeps trusted local workflows available without requesting a token", async () => {
    vi.spyOn(satmodApi, "access").mockResolvedValue({
      ...required,
      publicMode: false,
      tokenRequired: false,
    });
    const session = vi.spyOn(satmodApi, "session");
    mount();
    await screen.findByText("Private research results");
    expect(session).not.toHaveBeenCalled();
  });

  it("keeps offline educational views available when discovery cannot reach the backend", async () => {
    vi.spyOn(satmodApi, "access").mockRejectedValue(new Error("Backend offline"));
    mount();
    await screen.findByText("Private research results");
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("shows a clear server setup state without accepting a token for an unconfigured public host", async () => {
    vi.spyOn(satmodApi, "access").mockResolvedValue({ ...required, configured: false });
    mount();
    await screen.findByRole("heading", { name: "Workspace setup required" });
    expect(screen.queryByText("Private research results")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Unlock workspace" })).not.toBeInTheDocument();
  });

  it("rejects a wrong token and only mounts protected content after real session verification", async () => {
    vi.spyOn(satmodApi, "access").mockResolvedValue(required);
    const session = vi
      .spyOn(satmodApi, "session")
      .mockRejectedValueOnce(new Error("Workspace access token required."))
      .mockResolvedValue({ authorized: true, sharedWorkspace: true });
    mount();
    await screen.findByRole("heading", { name: "Unlock your workspace" });
    expect(screen.queryByText("Private research results")).not.toBeInTheDocument();
    const input = screen.getByLabelText("Workspace access token", { exact: true });
    fireEvent.change(input, { target: { value: "incorrect-unit-token" } });
    fireEvent.click(screen.getByRole("button", { name: /^Unlock workspace$/ }));
    await screen.findByRole("alert");
    expect(sessionStorage.getItem("satmod-access-token:")).toBeNull();
    fireEvent.change(input, { target: { value: "valid-unit-operator-token" } });
    fireEvent.click(screen.getByRole("button", { name: /^Unlock workspace$/ }));
    await screen.findByText("Private research results");
    expect(session).toHaveBeenCalledTimes(2);
    expect(sessionStorage.getItem("satmod-access-token:")).toBe("valid-unit-operator-token");
    expect(localStorage.getItem("satmod-access-token:")).toBeNull();
  });

  it("validates stored access and clears private queries and content when locked", async () => {
    sessionStorage.setItem("satmod-access-token:", "retained-unit-token");
    vi.spyOn(satmodApi, "access").mockResolvedValue(required);
    vi.spyOn(satmodApi, "session").mockResolvedValue({ authorized: true, sharedWorkspace: true });
    const client = mount();
    await screen.findByText("Private research results");
    client.setQueryData(["private-analysis"], { samples: "private unit fixture" });
    fireEvent.click(screen.getByRole("button", { name: /^Lock workspace$/ }));
    await screen.findByRole("heading", { name: "Unlock your workspace" });
    expect(screen.queryByText("Private research results")).not.toBeInTheDocument();
    expect(client.getQueryData(["private-analysis"])).toBeUndefined();
    expect(sessionStorage.getItem("satmod-access-token:")).toBeNull();
  });

  it("returns to the unlock state when the server revokes a previously valid session", async () => {
    sessionStorage.setItem("satmod-access-token:", "revoked-unit-token");
    vi.spyOn(satmodApi, "access").mockResolvedValue(required);
    vi.spyOn(satmodApi, "session").mockResolvedValue({ authorized: true, sharedWorkspace: true });
    mount();
    await screen.findByText("Private research results");
    await act(async () => window.dispatchEvent(new Event("satmod:workspace-unauthorized")));
    await waitFor(() =>
      expect(screen.queryByText("Private research results")).not.toBeInTheDocument(),
    );
    expect(sessionStorage.getItem("satmod-access-token:")).toBeNull();
  });

  it("keeps known protection authoritative when background access discovery fails", async () => {
    sessionStorage.setItem("satmod-access-token:", "retained-unit-token");
    vi.spyOn(satmodApi, "access")
      .mockResolvedValueOnce(required)
      .mockRejectedValue(new Error("Discovery offline"));
    vi.spyOn(satmodApi, "session").mockResolvedValue({ authorized: true, sharedWorkspace: true });
    const client = mount();
    await screen.findByText("Private research results");
    await act(async () => {
      await client.refetchQueries({ queryKey: ["workspace-access", ""] });
    });
    expect(screen.getByRole("button", { name: /^Lock workspace$/ })).toBeInTheDocument();
    await act(async () => window.dispatchEvent(new Event("satmod:workspace-unauthorized")));
    await screen.findByRole("heading", { name: "Unlock your workspace" });
    expect(screen.queryByText("Private research results")).not.toBeInTheDocument();
  });
});
