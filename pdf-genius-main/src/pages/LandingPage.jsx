import { Link } from "react-router-dom";
import { FileText, MessageSquare, Brain, BookOpen, ArrowRight, Sparkles } from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";

const features = [
  { icon: FileText, title: "Upload PDFs", desc: "Drag & drop multiple PDF documents for instant processing" },
  { icon: MessageSquare, title: "Ask Questions", desc: "Natural language queries across all your uploaded documents" },
  { icon: Brain, title: "AI Answers", desc: "Get precise, contextual answers powered by advanced AI" },
  { icon: BookOpen, title: "Source Citations", desc: "Every answer includes verifiable source references" },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Nav */}
      <nav className="flex items-center justify-between px-6 py-4 max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-primary-foreground" />
          </div>
          <span className="font-bold text-lg text-foreground">MultiPDF AI</span>
        </div>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link to="/login" className="px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
            Login
          </Link>
          <Link to="/signup" className="px-5 py-2 text-sm font-semibold rounded-lg gradient-primary text-primary-foreground hover:opacity-90 transition-opacity">
            Sign Up
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-20 pb-16 px-6 text-center max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6 animate-fade-in-up">
          <Sparkles className="w-3.5 h-3.5" /> Powered by AI
        </div>
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-foreground leading-tight mb-5 animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
          MultiPDF AI{" "}
          <span className="bg-clip-text text-transparent" style={{ backgroundImage: "var(--gradient-primary)" }}>
            Assistant
          </span>
        </h1>
        <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          Ask questions across multiple PDFs instantly. Get AI-powered answers with precise source citations.
        </p>
        <div className="flex justify-center gap-4 animate-fade-in-up" style={{ animationDelay: "0.3s" }}>
          <Link to="/signup" className="inline-flex items-center gap-2 px-6 py-3 rounded-lg gradient-primary text-primary-foreground font-semibold hover:opacity-90 transition-opacity shadow-card">
            Get Started <ArrowRight className="w-4 h-4" />
          </Link>
          <Link to="/login" className="inline-flex items-center gap-2 px-6 py-3 rounded-lg border border-border text-foreground font-semibold hover:bg-muted transition-colors">
            Login
          </Link>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-6 max-w-6xl mx-auto">
        <h2 className="text-2xl sm:text-3xl font-bold text-center text-foreground mb-12">
          Everything you need to query your documents
        </h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((f, i) => (
            <div
              key={f.title}
              className="p-6 rounded-xl bg-card border border-border shadow-soft hover:shadow-card transition-shadow animate-fade-in-up"
              style={{ animationDelay: `${0.1 * i}s` }}
            >
              <div className="w-10 h-10 rounded-lg gradient-primary flex items-center justify-center mb-4">
                <f.icon className="w-5 h-5 text-primary-foreground" />
              </div>
              <h3 className="font-semibold text-foreground mb-2">{f.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 text-center text-sm text-muted-foreground border-t border-border">
        © 2026 MultiPDF AI Assistant. All rights reserved.
      </footer>
    </div>
  );
}
