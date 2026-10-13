import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SUPPORTED_LANGUAGES } from "@/constants/languages";

const mockChangeLanguage = jest.fn();
const mockInvalidateQueries = jest.fn();
const mockSetTypes = jest.fn();
const mockLoadLanguage = jest.fn().mockResolvedValue(undefined);

jest.mock("@/i18n", () => ({
  loadLanguage: mockLoadLanguage,
}));

jest.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { changeLanguage: mockChangeLanguage, language: "en" },
  }),
}));

jest.mock("@tanstack/react-query", () => ({
  ...jest.requireActual("@tanstack/react-query"),
  useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
}));

jest.mock("@/stores/typesStore", () => ({
  useTypesStore: (
    selector: (s: { setTypes: typeof mockSetTypes }) => unknown,
  ) => selector({ setTypes: mockSetTypes }),
}));

jest.mock("@/components/ui/select", () => ({
  Select: ({
    children,
    value,
    onValueChange,
  }: {
    children: React.ReactNode;
    value?: string;
    onValueChange?: (v: string) => void;
  }) => (
    <select value={value} onChange={(e) => onValueChange?.(e.target.value)}>
      {children}
    </select>
  ),
  SelectTrigger: () => null,
  SelectValue: () => null,
  SelectContent: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  SelectItem: ({
    children,
    value,
  }: {
    children: React.ReactNode;
    value: string;
  }) => <option value={value}>{children}</option>,
}));

jest.mock("@/components/ui/card", () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  CardTitle: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  CardDescription: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  CardContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

import LanguageFormComponent from "../index";

describe("LanguageFormComponent", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
  });

  it("renders all supported language options", () => {
    render(<LanguageFormComponent />);
    const select = screen.getByRole("combobox");
    const options = select.querySelectorAll("option");
    expect(options).toHaveLength(SUPPORTED_LANGUAGES.length);
    SUPPORTED_LANGUAGES.forEach((lang) => {
      expect(
        screen.getByRole("option", { name: new RegExp(lang.label) }),
      ).toBeInTheDocument();
    });
  });

  it("calls i18n.changeLanguage with the selected language code", async () => {
    const user = userEvent.setup();
    render(<LanguageFormComponent />);
    await user.selectOptions(screen.getByRole("combobox"), "zh-Hans");
    expect(mockChangeLanguage).toHaveBeenCalledWith("zh-Hans");
  });

  it("saves selected language to localStorage", async () => {
    const user = userEvent.setup();
    render(<LanguageFormComponent />);
    await user.selectOptions(screen.getByRole("combobox"), "zh-Hans");
    expect(localStorage.getItem("languagePreference")).toBe("zh-Hans");
  });

  it("calls setTypes with empty object on language change", async () => {
    const user = userEvent.setup();
    render(<LanguageFormComponent />);
    await user.selectOptions(screen.getByRole("combobox"), "zh-Hans");
    expect(mockSetTypes).toHaveBeenCalledWith({});
  });

  it("invalidates useGetTypes query on language change", async () => {
    const user = userEvent.setup();
    render(<LanguageFormComponent />);
    await user.selectOptions(screen.getByRole("combobox"), "zh-Hans");
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["useGetTypes"],
    });
  });

  it("lists Traditional Chinese first", () => {
    render(<LanguageFormComponent />);
    const options = screen.getByRole("combobox").querySelectorAll("option");
    expect(options[0]).toHaveValue("zh-Hans");
  });

  it("shows recommended label for Traditional Chinese option", () => {
    render(<LanguageFormComponent />);
    const zhOption = screen.getByRole("option", { name: /繁體中文/ });
    expect(zhOption.textContent).toContain("settings.languageRecommended");
  });

  it("does not show recommended label for other options", () => {
    render(<LanguageFormComponent />);
    const option = screen.getByRole("option", { name: /English/ });
    expect(option.textContent).not.toContain("settings.languageRecommended");
  });

  // TARS: 只提供繁體中文與 English
  it("offers only Traditional Chinese and English", () => {
    render(<LanguageFormComponent />);
    const options = screen.getByRole("combobox").querySelectorAll("option");
    expect(Array.from(options).map((o) => o.getAttribute("value"))).toEqual([
      "zh-Hans",
      "en",
    ]);
  });

  it("calls loadLanguage before changeLanguage when switching languages", async () => {
    const user = userEvent.setup();
    render(<LanguageFormComponent />);
    await user.selectOptions(screen.getByRole("combobox"), "zh-Hans");
    expect(mockLoadLanguage).toHaveBeenCalledWith("zh-Hans");
    expect(mockChangeLanguage).toHaveBeenCalledWith("zh-Hans");
  });
});
