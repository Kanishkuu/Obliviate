import { Navigate } from "react-router-dom";
import { useAuth } from "./authcontext.jsx";

export default function ProtectedRoute({ children }) {
  const { user, checked } = useAuth();

  // Wait until /auth/me finishes to avoid flicker back to login
  if (!checked) {
    return (
      <div className="min-h-screen flex items-center justify-center text-white">
        <div className="glass px-6 py-3">Checking session…</div>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  return children;
}
