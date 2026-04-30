import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../lib/api";
import { Globe, Loader2 } from "lucide-react";
import clsx from "clsx";

type Mode = "login" | "register";

function FieldError({ msg }: { msg?: string }) {
  if (!msg) return null;
  return <p className="mt-1 text-xs text-red-400">{msg}</p>;
}

export default function LoginPage() {
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);

  const { login, register } = useAuth();
  const navigate = useNavigate();

  function validate() {
    const errs: Record<string, string> = {};
    if (!email.includes("@")) errs.email = "Enter a valid email address.";
    if (password.length < 8) errs.password = "Password must be at least 8 characters.";
    return errs;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setApiError("");
    const errs = validate();
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;

    setLoading(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        await register(email, password);
      }
      navigate("/");
    } catch (err) {
      if (err instanceof ApiError) {
        setApiError(err.detail);
      } else {
        setApiError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#0f1117] flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-indigo-600 flex items-center justify-center mb-4 shadow-lg shadow-indigo-500/30">
            <Globe className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Smart Travel Planner</h1>
          <p className="text-slate-400 text-sm mt-1">AI-powered trip recommendations</p>
        </div>

        {/* Card */}
        <div className="bg-[#161b2e] border border-[#1e2540] rounded-2xl p-8 shadow-2xl">
          {/* Mode toggle */}
          <div className="flex rounded-xl bg-[#0f1117] p-1 mb-6">
            {(["login", "register"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => { setMode(m); setApiError(""); setErrors({}); }}
                className={clsx(
                  "flex-1 py-2 text-sm font-medium rounded-lg transition-all duration-200",
                  mode === m
                    ? "bg-indigo-600 text-white shadow"
                    : "text-slate-400 hover:text-slate-200"
                )}
              >
                {m === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Email
              </label>
              <input
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className={clsx(
                  "w-full px-4 py-2.5 rounded-xl bg-[#0f1117] border text-slate-100 placeholder-slate-500",
                  "focus:outline-none focus:ring-2 focus:ring-indigo-500 transition",
                  errors.email ? "border-red-500" : "border-[#2a3050]"
                )}
              />
              <FieldError msg={errors.email} />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Password
              </label>
              <input
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className={clsx(
                  "w-full px-4 py-2.5 rounded-xl bg-[#0f1117] border text-slate-100 placeholder-slate-500",
                  "focus:outline-none focus:ring-2 focus:ring-indigo-500 transition",
                  errors.password ? "border-red-500" : "border-[#2a3050]"
                )}
              />
              <FieldError msg={errors.password} />
            </div>

            {apiError && (
              <div className="rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-400">
                {apiError}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50
                         text-white font-semibold transition-all duration-200 flex items-center justify-center gap-2"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
