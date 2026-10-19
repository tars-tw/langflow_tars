import { useUtilityStore } from "@/stores/utilityStore";
import { NAV_ITEMS, type SidebarNavItem } from "./sidebar-nav-items";

// The Agent (A2A) tab follows the server-wide LANGFLOW_A2A_ENABLED so
// deployments that don't serve A2A don't expose a tab that can only 404.
export function useVisibleNavItems(): SidebarNavItem[] {
  const a2aEnabled = useUtilityStore((state) => state.a2aEnabled);
  return a2aEnabled
    ? NAV_ITEMS
    : NAV_ITEMS.filter((item) => item.id !== "agent");
}
