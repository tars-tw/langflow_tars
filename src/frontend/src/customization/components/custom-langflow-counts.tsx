import { LangflowCounts } from "@/components/core/appHeaderComponent/components/langflow-counts";
import { ENABLE_SOCIAL_LINKS } from "@/customization/feature-flags";

export function CustomLangflowCounts() {
  if (!ENABLE_SOCIAL_LINKS) return null;
  return <LangflowCounts />;
}

export default CustomLangflowCounts;
