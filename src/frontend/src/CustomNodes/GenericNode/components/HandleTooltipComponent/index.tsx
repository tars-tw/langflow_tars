import type React from "react";
import {
  Trans as TransComponent,
  type TransProps,
  useTranslation,
} from "react-i18next";
import { convertTestName } from "@/components/common/storeCardComponent/utils/convert-test-name";
import { Badge } from "@/components/ui/badge";
import { nodeColorsName } from "@/utils/styleUtils";

const Trans = TransComponent as unknown as React.FC<TransProps<string>>;

export default function HandleTooltipComponent({
  isInput,
  tooltipTitle,
  isConnecting,
  isCompatible,
  isSameNode,
  left,
}: {
  isInput: boolean;
  tooltipTitle: string;
  isConnecting: boolean;
  isCompatible: boolean;
  isSameNode: boolean;
  left: boolean;
}) {
  const { t } = useTranslation();
  const tooltips = tooltipTitle.split("\n");

  return (
    <div className="font-medium">
      {isSameNode ? (
        t("handle.cannotConnectSameNode")
      ) : (
        <div className="flex items-center gap-1.5">
          {isConnecting ? (
            isCompatible ? (
              <span>
                <Trans
                  i18nKey="handle.connectTo"
                  components={{ 1: <span className="font-semibold" /> }}
                />
              </span>
            ) : (
              <span>{t("node.incompatibleWith")}</span>
            )
          ) : (
            <span className="text-xs">
              {isInput
                ? t("handle.inputTypes", { count: tooltips.length })
                : t("handle.outputTypes", { count: tooltips.length })}
              :{" "}
            </span>
          )}
          {tooltips.map((word, index) => (
            <Badge
              className="h-6 rounded-md p-1"
              key={`${index}-${word.toLowerCase()}`}
              style={{
                backgroundColor: left
                  ? `hsl(var(--datatype-${nodeColorsName[word]}))`
                  : `hsl(var(--datatype-${nodeColorsName[word]}-foreground))`,
                color: left
                  ? `hsl(var(--datatype-${nodeColorsName[word]}-foreground))`
                  : `hsl(var(--datatype-${nodeColorsName[word]}))`,
              }}
              data-testid={`${isInput ? "input" : "output"}-tooltip-${convertTestName(word)}`}
            >
              {word}
            </Badge>
          ))}
          {isConnecting && (
            <span>
              {isInput ? t("handle.inputLabel") : t("handle.outputLabel")}
            </span>
          )}
        </div>
      )}
      {!isConnecting && (
        <div className="mt-2 flex flex-col gap-0.5 text-xs leading-6">
          <div>
            <Trans
              i18nKey={
                !isInput
                  ? "handle.dragToConnectInputs"
                  : "handle.dragToConnectOutputs"
              }
              components={{ 1: <b /> }}
            />
          </div>
          <div>
            <Trans
              i18nKey={
                !isInput
                  ? "handle.clickToFilterInputs"
                  : "handle.clickToFilterOutputs"
              }
              components={{ 1: <b /> }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
