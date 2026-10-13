// TARS: 預設語言判斷 — 使用者選過且仍支援的語言優先；否則英文系統用 English，
// 其餘一律用繁體中文（"zh-Hans"）。
jest.unmock("react-i18next");

const loadDetectedLang = (
  browserLanguage: string,
  storedPreference?: string,
): string => {
  localStorage.clear();
  if (storedPreference) {
    localStorage.setItem("languagePreference", storedPreference);
  }
  jest.spyOn(navigator, "language", "get").mockReturnValue(browserLanguage);

  let detected = "";
  jest.isolateModules(() => {
    detected = require("../i18n").detectedLang;
  });
  return detected;
};

describe("detectedLang", () => {
  afterEach(() => {
    jest.restoreAllMocks();
    localStorage.clear();
  });

  it.each(["en", "en-US", "en-GB"])(
    "uses English for an English browser (%s)",
    (browserLanguage) => {
      expect(loadDetectedLang(browserLanguage)).toBe("en");
    },
  );

  it.each(["zh-TW", "zh-CN", "ja-JP", "fr-FR", ""])(
    "uses Traditional Chinese for any other browser language (%s)",
    (browserLanguage) => {
      expect(loadDetectedLang(browserLanguage)).toBe("zh-Hans");
    },
  );

  it("keeps a stored supported preference over the browser language", () => {
    expect(loadDetectedLang("en-US", "zh-Hans")).toBe("zh-Hans");
    expect(loadDetectedLang("zh-TW", "en")).toBe("en");
  });

  it("ignores a stored preference that is no longer offered", () => {
    expect(loadDetectedLang("en-US", "ja")).toBe("en");
    expect(loadDetectedLang("zh-TW", "fr")).toBe("zh-Hans");
  });
});
