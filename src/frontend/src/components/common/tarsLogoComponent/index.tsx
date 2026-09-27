import type { ImgHTMLAttributes } from "react";
import tarsLogo from "@/assets/tars_ai_logo.png";
import { cn } from "@/utils/utils";

type TarsLogoProps = Omit<ImgHTMLAttributes<HTMLImageElement>, "src">;

/**
 * The product logo. Every place that shows the brand mark renders this, so
 * swapping the image means replacing `assets/tars_ai_logo.png` only.
 * `title` doubles as the accessible name when no `alt` is given.
 */
export default function TarsLogo({
  alt,
  title,
  className,
  ...props
}: TarsLogoProps): JSX.Element {
  return (
    <img
      src={tarsLogo}
      alt={alt ?? title ?? ""}
      title={title}
      draggable={false}
      className={cn(
        "pointer-events-none select-none object-contain",
        className,
      )}
      {...props}
    />
  );
}
