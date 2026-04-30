import React, { createContext, useCallback, useContext, useState } from "react";
import * as api from "../lib/api";

interface AuthState {
  token: string | null;
  email: string | null;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Token lives in memory only — no localStorage to protect against XSS
  const [state, setState] = useState<AuthState>({ token: null, email: null });

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.login(email, password);
    setState({ token: res.access_token, email });
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    await api.register(email, password);
    // Auto-login after registration
    const res = await api.login(email, password);
    setState({ token: res.access_token, email });
  }, []);

  const logout = useCallback(() => {
    setState({ token: null, email: null });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        ...state,
        login,
        register,
        logout,
        isAuthenticated: state.token !== null,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
