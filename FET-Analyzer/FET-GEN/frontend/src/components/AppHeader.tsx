import { Download } from "lucide-react";
import { Logo } from "./Logo";
import type { TabName } from "../types";

interface AppHeaderProps {
  activeTab: TabName;
  availableTabs: TabName[];
  apiOnline: boolean;
  onTabChange: (tab: TabName) => void;
  exportHref: string | null;
  exportFilename: string | null;
  exportDisabled: boolean;
}

export function AppHeader({
  activeTab,
  availableTabs,
  apiOnline,
  onTabChange,
  exportHref,
  exportFilename,
  exportDisabled
}: AppHeaderProps) {
  return (
    <header className="app-header">
      <Logo />
      <nav className="top-nav" aria-label="Primary navigation">
        {availableTabs.map((tab) => (
          <button
            className={activeTab === tab ? "top-nav-item active" : "top-nav-item"}
            key={tab}
            aria-current={activeTab === tab ? "page" : undefined}
            onClick={() => onTabChange(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>
      <div className="header-actions">
        <span
          className={apiOnline ? "api-state online" : "api-state offline"}
          role="status"
          aria-live="polite"
        >
          <span className="status-dot" />
          API {apiOnline ? "Online" : "Offline"}
        </span>
        {exportHref && exportFilename && !exportDisabled ? (
          <a
            className="button secondary compact"
            href={exportHref}
            download={exportFilename}
          >
            <Download size={15} />
            Export
          </a>
        ) : (
          <button className="button secondary compact" disabled>
            <Download size={15} />
            Export
          </button>
        )}
      </div>
    </header>
  );
}
