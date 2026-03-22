import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useApp } from "../context/AppContext";
import { signupUser } from "../lib/api";
import { Mail, Lock, User, Sparkles, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";

function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}

// Password strength: returns { score 0–4, label, color }
function passwordStrength(pw) {
  if (!pw) return null;
  let score = 0;
  if (pw.length >= 6)                        score++;
  if (pw.length >= 10)                       score++;
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++;
  if (/[0-9]/.test(pw))                      score++;
  if (/[^A-Za-z0-9]/.test(pw))              score++;
  const map = [
    { label: "Too short",  color: "bg-destructive" },
    { label: "Weak",       color: "bg-red-400" },
    { label: "Fair",       color: "bg-amber-400" },
    { label: "Good",       color: "bg-emerald-400" },
    { label: "Strong",     color: "bg-emerald-500" },
  ];
  return { score, ...map[Math.min(score, 4)] };
}

export default function Signup() {
  const [name,     setName]     = useState("");
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [touched,  setTouched]  = useState({ name: false, email: false, password: false });
  const [apiError, setApiError] = useState("");
  const [loading,  setLoading]  = useState(false);

  const { login } = useApp();
  const navigate  = useNavigate();

  // ── inline validation ─────────────────────────────────────────────────────
  const errors = {
    name:
      !name.trim()                              ? "Full name is required." :
      name.trim().length < 2                    ? "Name must be at least 2 characters." :
      !/[a-zA-Z]/.test(name)                   ? "Name must contain letters." :
      name.trim().split(/\s+/).length < 2       ? "Please enter your full name (first and last)." : "",
    email:
      !email.trim()                             ? "Email is required." :
      !isValidEmail(email)                      ? "Enter a valid email address." : "",
    password:
      !password                                 ? "Password is required." :
      password.length < 6                       ? "Password must be at least 6 characters." : "",
  };

  const isFormValid = !errors.name && !errors.email && !errors.password;
  const pwStrength  = passwordStrength(password);
  const blur = (field) => setTouched((t) => ({ ...t, [field]: true }));

  // ── submit ────────────────────────────────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    setTouched({ name: true, email: true, password: true });
    if (!isFormValid) return;

    setApiError("");
    setLoading(true);
    try {
      const data = await signupUser(name.trim(), email.trim(), password);
      login(data.user);
      navigate("/dashboard");
    } catch (err) {
      setApiError(err.message || "Signup failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <div className="w-full max-w-md">

        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2 mb-6">
            <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-bold text-lg text-foreground">MultiPDF AI</span>
          </Link>
          <h1 className="text-2xl font-bold text-foreground">Create your account</h1>
          <p className="text-muted-foreground mt-1">Start querying your PDFs with AI</p>
        </div>

        <form onSubmit={handleSubmit} noValidate
          className="bg-card border border-border rounded-xl p-6 shadow-card space-y-4"
        >
          {/* API error */}
          {apiError && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {apiError}
            </div>
          )}

          {/* Full name */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Full Name</label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                value={name}
                onChange={(e) => { setName(e.target.value); setApiError(""); }}
                onBlur={() => blur("name")}
                placeholder="John Doe"
                className={`w-full pl-10 pr-4 py-2.5 rounded-lg border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring text-sm transition-colors ${
                  touched.name && errors.name ? "border-destructive" : "border-input"
                }`}
              />
            </div>
            {touched.name && errors.name && (
              <p className="mt-1 text-xs text-destructive flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />{errors.name}
              </p>
            )}
            {touched.name && !errors.name && (
              <p className="mt-1 text-xs text-emerald-500 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />Looks good
              </p>
            )}
          </div>

          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Email</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="email"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setApiError(""); }}
                onBlur={() => blur("email")}
                placeholder="you@example.com"
                className={`w-full pl-10 pr-4 py-2.5 rounded-lg border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring text-sm transition-colors ${
                  touched.email && errors.email ? "border-destructive" : "border-input"
                }`}
              />
            </div>
            {touched.email && errors.email && (
              <p className="mt-1 text-xs text-destructive flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />{errors.email}
              </p>
            )}
            {touched.email && !errors.email && (
              <p className="mt-1 text-xs text-emerald-500 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />Valid email
              </p>
            )}
          </div>

          {/* Password + strength meter */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="password"
                value={password}
                onChange={(e) => { setPassword(e.target.value); setApiError(""); }}
                onBlur={() => blur("password")}
                placeholder="••••••••"
                className={`w-full pl-10 pr-4 py-2.5 rounded-lg border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring text-sm transition-colors ${
                  touched.password && errors.password ? "border-destructive" : "border-input"
                }`}
              />
            </div>
            {/* Strength bar — shows once user starts typing */}
            {password && (
              <div className="mt-2 space-y-1">
                <div className="flex gap-1 h-1">
                  {[1,2,3,4].map((bar) => (
                    <div key={bar} className={`flex-1 rounded-full transition-all duration-200 ${
                      pwStrength && pwStrength.score >= bar ? pwStrength.color : "bg-muted"
                    }`} />
                  ))}
                </div>
                <p className={`text-xs ${
                  pwStrength?.score >= 3 ? "text-emerald-500" :
                  pwStrength?.score >= 2 ? "text-amber-500" : "text-red-400"
                }`}>{pwStrength?.label}</p>
              </div>
            )}
            {touched.password && errors.password && (
              <p className="mt-1 text-xs text-destructive flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />{errors.password}
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg gradient-primary text-primary-foreground font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            Create Account
          </button>
        </form>

        <p className="text-center text-sm text-muted-foreground mt-4">
          Already have an account?{" "}
          <Link to="/login" className="text-primary font-medium hover:underline">Sign in</Link>
        </p>
      </div>
    </div>
  );
}