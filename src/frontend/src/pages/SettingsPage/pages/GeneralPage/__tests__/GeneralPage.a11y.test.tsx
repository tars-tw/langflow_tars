import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { AuthContext } from "@/contexts/authContext";
import useAlertStore from "@/stores/alertStore";
import useAuthStore from "@/stores/authStore";
import type { AuthContextType } from "@/types/contexts/auth";
import { axe } from "@/utils/a11y-test";

const mockResetPasswordMutate = jest.fn();
const mockUpdateUserMutate = jest.fn();
const mockAddApiKeyMutate = jest.fn();

jest.mock("@radix-ui/react-form", () => {
  const React = require("react");
  return {
    __esModule: true,
    Root: ({ children, ...props }) =>
      React.createElement("form", props, children),
    Field: ({ children, name }) =>
      React.createElement("div", { "data-field": name }, children),
    Label: ({ children, ...props }) =>
      React.createElement("label", props, children),
    Control: ({ children }) => children,
    Message: ({ children, ...props }) =>
      React.createElement("p", props, children),
    Submit: ({ children }) => children,
  };
});

jest.mock("@/controllers/API/queries/api-keys", () => ({
  usePostAddApiKey: () => ({ mutate: mockAddApiKeyMutate }),
}));

jest.mock("@/controllers/API/queries/auth", () => ({
  useResetPassword: () => ({ mutate: mockResetPasswordMutate }),
  useUpdateUser: () => ({ mutate: mockUpdateUserMutate }),
}));

const PROFILE_PICTURES = {
  files: [],
  People: ["avatar-01.svg"],
};

jest.mock("@/controllers/API/queries/files", () => ({
  useGetProfilePicturesQuery: () => ({
    isLoading: false,
    isFetching: false,
    data: PROFILE_PICTURES,
  }),
}));

jest.mock(
  "@/pages/SettingsPage/pages/GeneralPage/components/ProfilePictureForm/components/profilePictureChooserComponent/hooks/use-preload-images",
  () => {
    const { useEffect } = require("react");
    return {
      __esModule: true,
      default: (setImagesLoaded: (v: boolean) => void) => {
        useEffect(() => {
          setImagesLoaded(true);
        }, []);
      },
    };
  },
);

jest.mock("@/customization/utils/custom-pre-load-image-url", () => ({
  customPreLoadImageUrl: (path: string) =>
    `/api/v1/files/profile_pictures/${path}`,
}));

jest.mock("@/customization/components/custom-terms-links", () => ({
  CustomTermsLinks: () => <div>Terms links</div>,
}));

import GeneralPage from "../index";

const AUTH_CONTEXT_VALUE: AuthContextType = {
  accessToken: "token",
  login: jest.fn(),
  userData: { id: "1" } as AuthContextType["userData"],
  setUserData: jest.fn(),
  authenticationErrorCount: 0,
  apiKey: null,
  setApiKey: jest.fn(),
  storeApiKey: jest.fn(),
  getUser: jest.fn(),
  clearAuthSession: jest.fn(),
};

function renderGeneralPage() {
  const queryClient = new QueryClient();
  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <AuthContext.Provider value={AUTH_CONTEXT_VALUE}>
          <GeneralPage />
        </AuthContext.Provider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe("GeneralPage accessibility", () => {
  beforeAll(() => {
    if (!Element.prototype.hasPointerCapture) {
      Element.prototype.hasPointerCapture = jest.fn(() => false);
    }
    if (!Element.prototype.releasePointerCapture) {
      Element.prototype.releasePointerCapture = jest.fn();
    }
    if (!Element.prototype.scrollIntoView) {
      Element.prototype.scrollIntoView = jest.fn();
    }
  });

  beforeEach(() => {
    jest.clearAllMocks();
    useAlertStore.setState({
      notificationList: [],
      tempNotificationList: [],
    });
    useAuthStore.setState({ autoLogin: false });
  });

  it("should_have_no_axe_violations_on_initial_render", async () => {
    const { container } = renderGeneralPage();

    expect(await axe(container)).toHaveNoViolations();
  });

  it("should_expose_a_single_page_heading_for_the_page_title", () => {
    renderGeneralPage();

    expect(
      screen.getByRole("heading", { name: /general/i }),
    ).toBeInTheDocument();
  });

  it("renders_the_language_and_password_sections", () => {
    renderGeneralPage();

    expect(
      screen.getByRole("combobox", { name: "Select language" }),
    ).toBeInTheDocument();
    // Only the password section has a "Save" submit button.
    expect(screen.getAllByRole("button", { name: /save/i })).toHaveLength(1);
    expect(screen.getByRole("form", { name: "Password" })).toBeInTheDocument();
  });

  it("does_not_render_the_profile_picture_section", () => {
    // ENABLE_PROFILE_ICONS is off: the avatar chooser is not offered.
    renderGeneralPage();

    expect(
      screen.queryByRole("form", { name: "Profile Picture" }),
    ).not.toBeInTheDocument();
  });

  it("should_have_no_axe_violations_with_the_language_popover_open", async () => {
    const user = userEvent.setup();
    renderGeneralPage();

    await user.click(screen.getByRole("combobox", { name: "Select language" }));
    await screen.findByRole("listbox");

    // Radix portals the popover to document.body, outside the render
    // container, and the region rule is a page-level landmark concern that
    // a bare unit render cannot satisfy.
    expect(
      await axe(document.body, { rules: { region: { enabled: false } } }),
    ).toHaveNoViolations();
  });

  it("does_not_render_the_password_form_when_autologin_is_enabled", () => {
    useAuthStore.setState({ autoLogin: true });
    renderGeneralPage();

    expect(
      screen.queryByPlaceholderText("settings.passwordPlaceholder"),
    ).not.toBeInTheDocument();
  });
});
