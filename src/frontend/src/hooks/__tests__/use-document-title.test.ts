import { renderHook } from "@testing-library/react";
import { APP_NAME } from "@/customization/config-constants";
import { formatDocumentTitle, useDocumentTitle } from "../use-document-title";

describe("formatDocumentTitle", () => {
  it("brands the page title", () => {
    expect(formatDocumentTitle("Flows")).toBe(`Flows | ${APP_NAME}`);
  });

  it("does not double-brand a title that already names the product", () => {
    expect(formatDocumentTitle(`${APP_NAME} API Keys`)).toBe(
      `${APP_NAME} API Keys`,
    );
  });

  it("falls back to the product name for an empty title", () => {
    expect(formatDocumentTitle(undefined)).toBe(APP_NAME);
    expect(formatDocumentTitle(null)).toBe(APP_NAME);
    expect(formatDocumentTitle("   ")).toBe(APP_NAME);
  });
});

describe("useDocumentTitle", () => {
  it("sets the document title while mounted and resets it on unmount", () => {
    const { unmount } = renderHook(() => useDocumentTitle("Global Variables"));
    expect(document.title).toBe(`Global Variables | ${APP_NAME}`);

    unmount();
    expect(document.title).toBe(APP_NAME);
  });

  it("follows a title that resolves after the first render", () => {
    const { rerender } = renderHook(
      ({ title }: { title?: string }) => useDocumentTitle(title),
      { initialProps: { title: undefined } },
    );
    expect(document.title).toBe(APP_NAME);

    rerender({ title: "My Flow" });
    expect(document.title).toBe(`My Flow | ${APP_NAME}`);
  });
});
