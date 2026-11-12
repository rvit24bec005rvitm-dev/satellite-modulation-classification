import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  BarChart3,
  Bell,
  BookOpen,
  ChevronRight,
  Crosshair,
  FlaskConical,
  Globe2,
  Layers3,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Radio,
  Satellite,
  Settings2,
  UserRound,
  Wifi,
} from "lucide-react";
import { useHealth } from "@/hooks/use-satmod";
import {
  OverviewPage,
  ClassifierPage,
  StudioPage,
  EvaluationPage,
  ArenaPage,
  IntelligencePage,
  TheoryPage,
  SettingsPage,
} from "@/pages/workspaces";
import { PageHeader } from "@/components/workspace-ui";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
const NAV = [
  { key: "overview", title: "Overview", href: "/", icon: Globe2, group: "MISSION" },
  {
    key: "classifier",
    title: "Live classifier",
    href: "/classifier",
    icon: Crosshair,
    group: "MISSION",
  },
  {
    key: "studio",
    title: "Satellite studio",
    href: "/studio",
    icon: FlaskConical,
    group: "MISSION",
  },
  {
    key: "evaluation",
    title: "Model evaluation",
    href: "/evaluation",
    icon: BarChart3,
    group: "INTELLIGENCE",
  },
  { key: "arena", title: "Model arena", href: "/arena", icon: Layers3, group: "INTELLIGENCE" },
  {
    key: "intelligence",
    title: "Signal intelligence",
    href: "/intelligence",
    icon: Activity,
    group: "INTELLIGENCE",
  },
  { key: "theory", title: "Satellite theory", href: "/theory", icon: BookOpen, group: "REFERENCE" },
  { key: "settings", title: "Settings", href: "/settings", icon: Settings2, group: "REFERENCE" },
] as const;
function UtcClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <time dateTime={now.toISOString()} className="header-clock">
      {now.toLocaleTimeString("en-GB", { timeZone: "UTC" })}
      <small>UTC</small>
    </time>
  );
}
function AppShell({ page, children }: { page: string; children: React.ReactNode }) {
  const health = useHealth();
  const online = health.isSuccess && health.data.connected;
  const [collapsed, setCollapsed] = useState(
    () =>
      typeof window !== "undefined" && localStorage.getItem("satmod-sidebar-collapsed") === "true",
  );
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigation = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!mobileOpen) return;
    const opener = document.activeElement as HTMLElement | null;
    const focusables = () =>
      Array.from(
        navigation.current?.querySelectorAll<HTMLElement>("a[href],button:not([disabled])") ?? [],
      );
    focusables()[0]?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setMobileOpen(false);
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusables();
      const first = items[0],
        last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      if (opener?.isConnected) opener.focus();
    };
  }, [mobileOpen]);
  const active = NAV.find((item) => item.key === page) ?? NAV[0];
  const groups = [...new Set(NAV.map((item) => item.group))];
  useEffect(() => {
    localStorage.setItem("satmod-sidebar-collapsed", String(collapsed));
  }, [collapsed]);
  return (
    <div
      className={`app-shell ${collapsed ? "nav-collapsed" : ""}`}
      style={{ "--nav-width": collapsed ? "68px" : "232px" } as CSSProperties}
    >
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:bg-card focus:p-3 focus:text-primary"
      >
        Skip to workspace content
      </a>
      {mobileOpen && (
        <button
          className="navigation-scrim"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside
        ref={navigation}
        role={mobileOpen ? "dialog" : undefined}
        aria-modal={mobileOpen || undefined}
        aria-label={mobileOpen ? "Workspace navigation" : undefined}
        className={`mission-sidebar ${mobileOpen ? "mobile-open" : ""}`}
      >
        <Link to="/" className="mission-brand" aria-label="SatModClass overview">
          <span className="brand-symbol">
            <Satellite size={21} />
          </span>
          {!collapsed && (
            <span>
              <strong>
                SATMOD<span>CLASS</span>
              </strong>
              <small>Signal Intelligence Platform</small>
            </span>
          )}
        </Link>
        <div className="sidebar-section-label">OPERATIONS WORKSPACE</div>
        <div className="mission-navigation">
          {groups.map((group) => (
            <div key={group} className="navigation-group">
              {!collapsed && <div className="navigation-group-title">{group}</div>}
              <nav aria-label={`${group.toLowerCase()} navigation`}>
                {NAV.filter((item) => item.group === group).map(
                  ({ key, title, href, icon: Icon }) => (
                    <Link
                      key={key}
                      to={href}
                      title={collapsed ? title : undefined}
                      aria-label={title}
                      aria-current={page === key ? "page" : undefined}
                      onClick={() => setMobileOpen(false)}
                      className={`mission-nav-link ${page === key ? "active" : ""}`}
                    >
                      <Icon size={16} strokeWidth={1.65} />
                      {!collapsed && <span>{title}</span>}
                      {page === key && !collapsed && <i />}
                    </Link>
                  ),
                )}
              </nav>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          {!collapsed && (
            <div className="sidebar-service">
              <span>
                <i className={online ? "status-dot online" : "status-dot warning"} />
                {online ? "BACKEND CONNECTED" : "BACKEND DISCONNECTED"}
              </span>
              <small>{online ? health.data?.backend : "Check service configuration"}</small>
            </div>
          )}
          <button
            onClick={() => setCollapsed((value) => !value)}
            className="collapse-navigation"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand navigation" : "Collapse navigation"}
          >
            {collapsed ? (
              <PanelLeftOpen size={16} />
            ) : (
              <>
                <PanelLeftClose size={16} />
                <span>Collapse navigation</span>
                <kbd>‹</kbd>
              </>
            )}
          </button>
          {!collapsed && (
            <div className="sidebar-version">
              <Radio size={11} /> SATMODCLASS <span>v1.0.0</span>
            </div>
          )}
        </div>
      </aside>
      <div className="mission-app-content">
        <header className="mission-header">
          <div className="header-breadcrumb">
            <button
              className="header-icon mobile-menu"
              onClick={() => {
                setCollapsed(false);
                setMobileOpen(true);
              }}
              aria-label="Open navigation"
              aria-expanded={mobileOpen}
            >
              <Menu size={18} />
            </button>
            <span>Mission Control</span>
            <ChevronRight size={12} />
            <strong>{active.title}</strong>
          </div>
          <div className="header-operational">
            <span className={online ? "mission-online" : "mission-offline"}>
              <i className={online ? "status-dot online" : "status-dot warning"} />
              {online ? "ONLINE" : health.isPending ? "CONNECTING" : "OFFLINE"}
            </span>
            <UtcClock />
            <span className={`header-backend ${online ? "online" : ""}`}>
              <Wifi size={12} />
              {online ? "API CONNECTED" : "API OFFLINE"}
            </span>
          </div>
          <div className="header-tools">
            <Link
              to="/"
              className="global-view-button"
              onClick={() => {
                window.dispatchEvent(new Event("satmod:global-view"));
                window.scrollTo({ top: 0, behavior: "auto" });
              }}
              title="Return to the global Earth view"
            >
              <Globe2 size={13} />
              <span>Global View</span>
            </Link>
            <Popover>
              <PopoverTrigger asChild>
                <button
                  className="header-icon notification-button"
                  aria-label="Notifications"
                  title="Notifications"
                >
                  <Bell size={16} />
                  {health.isError && <i />}
                </button>
              </PopoverTrigger>
              <PopoverContent align="end" className="mission-popover">
                <strong>System notifications</strong>
                <p>
                  {health.isPending
                    ? "Checking backend connection…"
                    : health.isError
                      ? health.error.message
                      : online
                        ? "Backend connected. Spacecraft are simulated demonstration data."
                        : "Backend unavailable."}
                </p>
                {health.isError && (
                  <button onClick={() => void health.refetch()}>Retry backend connection</button>
                )}
              </PopoverContent>
            </Popover>
            <Link
              to="/settings"
              className="header-icon"
              aria-label="Open settings"
              title="Settings"
            >
              <Settings2 size={16} />
            </Link>
            <Popover>
              <PopoverTrigger asChild>
                <button
                  className="operator-button"
                  aria-label="Operator session"
                  title="Local operator session"
                >
                  <UserRound size={15} />
                </button>
              </PopoverTrigger>
              <PopoverContent align="end" className="mission-popover">
                <strong>Local operator</strong>
                <p>Local workspace · browser preferences</p>
                <Link to="/settings">Workspace settings →</Link>
              </PopoverContent>
            </Popover>
          </div>
        </header>
        <main
          id="main-content"
          tabIndex={-1}
          className={
            page === "overview" ? "mission-main overview-main" : "mission-main workspace-main"
          }
        >
          {children}
        </main>
      </div>
    </div>
  );
}
export function SatmodApp({ page }: { page: string }) {
  const pages: Record<string, React.ComponentType> = {
    overview: OverviewPage,
    classifier: ClassifierPage,
    studio: StudioPage,
    evaluation: EvaluationPage,
    arena: ArenaPage,
    intelligence: IntelligencePage,
    theory: TheoryPage,
    settings: SettingsPage,
  };
  const Page = pages[page] ?? OverviewPage;
  return (
    <AppShell page={page}>
      {page !== "overview" && <PageHeader page={page} />}
      <Page />
    </AppShell>
  );
}
