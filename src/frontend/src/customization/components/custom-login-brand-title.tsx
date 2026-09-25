import { APP_NAME } from "@/customization/config-constants";
// OSS default product name on the login page. Downstream overlays replace this
// with an edition-specific brand without changing layout.
export default function CustomLoginBrandTitle() {
  return <>{APP_NAME}</>;
}
