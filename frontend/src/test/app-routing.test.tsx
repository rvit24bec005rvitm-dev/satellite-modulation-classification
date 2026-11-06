import { QueryClient } from "@tanstack/react-query";
import { createRouter, createMemoryHistory, rootRouteId } from "@tanstack/react-router";
import { describe, expect, it } from "vitest";

import { routeTree } from "@/routeTree.gen";

// Match routes without running loaders or rendering: loaders may need a server or
// network the test run lacks, and jsdom never loads the stylesheets React waits on.
describe("App routing", () => {
  it.each([
    "/",
    "/classifier",
    "/studio",
    "/evaluation",
    "/arena",
    "/intelligence",
    "/theory",
    "/settings",
  ])("matches workspace %s", (path) => {
    const router = createRouter({
      routeTree,
      history: createMemoryHistory({ initialEntries: [path] }),
      context: { queryClient: new QueryClient() },
    });

    const matches = router.matchRoutes(path);

    expect(matches.at(-1)?.routeId).not.toBe(rootRouteId);
  });
});
