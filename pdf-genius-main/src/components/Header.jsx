import { Link } from "react-router-dom";
import { Menu, User, Sparkles } from "lucide-react";
import ThemeToggle from "./ThemeToggle";

export default function Header({ onMenuClick }) {
  return (
    <header className="h-14 border-b border-border bg-card flex items-center justify-between px-4 shrink-0">
      <div className="flex items-center gap-3">
        <button onClick={onMenuClick} className="p-1.5 rounded-lg hover:bg-muted transition-colors lg:hidden">
          <Menu className="w-5 h-5 text-foreground" />
        </button>
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md gradient-primary flex items-center justify-center">
            <Sparkles className="w-3 h-3 text-primary-foreground" />
          </div>
          <span className="font-semibold text-foreground text-sm">MultiPDF AI</span>
        </div>
      </div>
      <div className="flex items-center gap-1">
        <ThemeToggle />
        <Link to="/profile" className="p-1.5 rounded-lg hover:bg-muted transition-colors">
          <User className="w-5 h-5 text-muted-foreground" />
        </Link>
      </div>
    </header>
  );
}