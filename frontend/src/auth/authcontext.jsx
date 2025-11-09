import { createContext, useContext, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const AuthCtx = createContext(null);
export const useAuth = () => useContext(AuthCtx);

export default function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [checked, setChecked] = useState(false); // cookie checked?
  const navigate = useNavigate();

  // hydrate from httpOnly cookie on first load
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("http://localhost:5000/auth/me", {
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          if (data?.ok) {
            setUser(data.user);
            localStorage.setItem("obl_user", JSON.stringify(data.user)); // optional
          } else {
            setUser(null);
            localStorage.removeItem("obl_user");
          }
        } else {
          setUser(null);
          localStorage.removeItem("obl_user");
        }
      } catch {
        setUser(null);
        localStorage.removeItem("obl_user");
      } finally {
        setChecked(true); // ✅ finished checking cookie
      }
    })();
  }, []);

  const loginWithGoogleCredential = async (credential) => {
    const res = await fetch("http://localhost:5000/auth/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ credential }),
    });
    const data = await res.json();
    if (data.ok) {
      setUser(data.user);
      localStorage.setItem("obl_user", JSON.stringify(data.user));
      navigate("/");
    } else {
      alert("Login failed");
    }
  };

  const logout = async () => {
    try {
      await fetch("http://localhost:5000/auth/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch {}
    localStorage.removeItem("obl_user");
    setUser(null);
    navigate("/login");
  };

  return (
    <AuthCtx.Provider value={{ user, checked, loginWithGoogleCredential, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}
