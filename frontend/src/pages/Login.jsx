import React from "react";
import { GoogleLogin } from "@react-oauth/google";
import { useAuth } from "../auth/authcontext.jsx";

export default function Login() {
  const { loginWithGoogleCredential } = useAuth();

  return (
    <div className="min-h-screen w-full bg-black text-white relative overflow-hidden flex items-center justify-center">

      {/* Floating bubbles */}
      <div className="bubbles-container">
        {[...Array(30)].map((_, i) => (
          <div
            key={i}
            className="bubble"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 20}s`,
              animationDuration: `${10 + Math.random() * 15}s`,
              width: `${2 + Math.random() * 5}px`,
              height: `${2 + Math.random() * 5}px`,
            }}
          />
        ))}
      </div>

      {/* Login Card */}
      <div className="relative z-10 w-full max-w-md px-6">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <span className="processing-dot"></span>
            <span className="text-xs tracking-[0.3em] text-gray-500">MACHINE UNLEARNING</span>
          </div>
          <h1 className="text-6xl font-light mb-4 tracking-tight">Obliviate</h1>
          <p className="text-gray-400 text-lg">Privacy Reinvented</p>
        </div>

        <div className="card">
          <div className="mb-8">
            <h2 className="text-2xl font-light mb-2 text-center">Sign In</h2>
            <p className="text-gray-400 text-sm text-center">Continue with Google</p>
          </div>

          {/* ✅ Real Google Login Button (keeps UI same) */}
          <div className="flex justify-center mb-6">
            <GoogleLogin
              onSuccess={(res) => loginWithGoogleCredential(res.credential)}
              onError={() => alert("Google sign-in failed")}
              theme="filled_black"
              width="300"
              shape="pill"
              text="continue_with"
            />
          </div>

          <div className="mt-8 pt-8 border-t border-gray-800">
            <p className="text-xs text-gray-500 text-center">
              Secure authentication powered by Google OAuth 2.0
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-xs text-gray-600">
            By signing in, you agree to our Terms of Service and Privacy Policy
          </p>
        </div>
      </div>

      {/* Styles (unchanged) */}
      <style>{`
        .bubbles-container { position:absolute; width:100%; height:100%; overflow:hidden; pointer-events:none; }
        .bubble { position:absolute; bottom:-100px; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.15), rgba(255,255,255,0.05)); border-radius:50%; animation:bubbleRise linear infinite; box-shadow:0 0 10px rgba(255,255,255,0.1); }
        @keyframes bubbleRise { 0% { bottom:-50px; opacity:0; transform:translateX(0); } 10% { opacity:0.6; } 50% { transform:translateX(20px); } 90% { opacity:0.6; } 100% { bottom:110vh; opacity:0; transform:translateX(-15px); } }
        .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(12px); border-radius: 16px; padding: 32px; animation: fadeIn 0.6s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .processing-dot { width:8px; height:8px; background:#fff; border-radius:50%; animation: processingPulse 1.5s ease-in-out infinite; box-shadow:0 0 15px rgba(255,255,255,0.5); }
        @keyframes processingPulse { 0%, 100% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.4); opacity: 0.3; } }
      `}</style>
    </div>
  );
}
