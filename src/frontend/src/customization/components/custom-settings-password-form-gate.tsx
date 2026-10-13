import type { ReactNode } from "react";

export interface CustomSettingsPasswordFormGateProps {
  children: ReactNode;
}

// TARS: 身份由 LibreChat > pwc_tars 取得（SSO 帳號密碼為系統隨機產生），
// 不讓使用者在 Settings > General 修改密碼，一律隱藏密碼表單。
export default function CustomSettingsPasswordFormGate(
  _props: CustomSettingsPasswordFormGateProps,
) {
  return null;
}
