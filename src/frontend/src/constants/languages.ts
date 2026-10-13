export const DEFAULT_LANGUAGE = "zh-Hans";

// TARS: 只提供繁體中文與 English（"zh-Hans" 代碼沿用，內容為繁體中文）
export const SUPPORTED_LANGUAGES = [
  { code: "zh-Hans", label: "繁體中文" },
  { code: "en", label: "English" },
] as const;
