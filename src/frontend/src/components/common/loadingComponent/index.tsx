import { useTranslation } from "react-i18next";
import TarsLogo from "@/components/common/tarsLogoComponent";
import type { LoadingComponentProps } from "../../../types/components";

export default function LoadingComponent({
  remSize,
}: LoadingComponentProps): JSX.Element {
  const { t } = useTranslation();
  return (
    <div role="status" className="flex flex-col items-center justify-center">
      <TarsLogo
        aria-hidden="true"
        className="animate-pulse"
        style={{
          width: `${remSize * 0.25}rem`,
          height: `${remSize * 0.25}rem`,
        }}
      />
      <br></br>
      <span className="animate-pulse text-lg text-primary">
        {t("loading.loading")}
      </span>
    </div>
  );
}
