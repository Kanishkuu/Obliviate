import React from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../auth/authcontext.jsx";

const Dashboard = () => {
  const { logout } = useAuth();

  const cards = [
    { title: "PrivacyPatch", desc: "Selective unlearning for complete data removal without sacrificing model accuracy.", link: "/unlearning", icon: "●" },
    { title: "Hallucination Auditor", desc: "Confidence-based gating to prevent unreliable AI outputs in production.", link: "/auditor", icon: "◆" },
    { title: "ModelForge", desc: "Fine-tune and deploy optimized models with automated evaluation.", link: "/finetune", icon: "▲" }
  ];

  return (
    <div className="min-h-screen w-full bg-black text-white relative overflow-hidden">
      {/* Flowing bubbles across entire screen */}
      <div className="bubbles-container">
        {[...Array(50)].map((_, i) => (
          <div
            key={i}
            className="bubble"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 20}s`,
              animationDuration: `${10 + Math.random() * 15}s`,
              width: `${3 + Math.random() * 8}px`,
              height: `${3 + Math.random() * 8}px`,
            }}
          />
        ))}
      </div>

      <button onClick={logout} className="absolute top-8 right-8 z-20 text-sm text-gray-300 hover:text-white transition-all backdrop-blur-md px-5 py-2.5 rounded-lg border border-cyan-500/30 hover:border-cyan-400 hover:shadow-lg hover:shadow-cyan-500/20">
        Logout
      </button>

      <div className="relative z-10 px-8 py-24 max-w-5xl mx-auto">
        <div className="mb-20">
          <div className="text-xs tracking-[0.4em] text-cyan-400 mb-6 flex items-center gap-3">
            <span className="processing-dot"></span>
            AI GOVERNANCE SYSTEM
          </div>
          <h1 className="text-7xl font-light mb-6 tracking-tight leading-tight">
            Control Your<br />
            <span className="llm-gradient">Large Language Models</span>
          </h1>
          <p className="text-gray-300 text-lg max-w-xl leading-relaxed">
            Enterprise AI governance with machine unlearning, hallucination prevention, and fine-tuning capabilities.
          </p>
        </div>

        <div className="space-y-5">
          {cards.map((card, i) => (
            <Link key={i} to={card.link} className="card-wrapper group block border-2 border-cyan-500/20 hover:border-cyan-400/60 transition-all duration-500 rounded-2xl overflow-hidden backdrop-blur-md bg-gradient-to-r from-cyan-950/10 to-blue-950/10 hover:from-cyan-900/20 hover:to-blue-900/20 relative">
              <div className="card-glow"></div>
              <div className="flex items-center p-8 relative">
                <div className="text-5xl text-cyan-400/60 group-hover:text-cyan-300 transition-all duration-500 mr-8 group-hover:scale-125 group-hover:rotate-12">
                  {card.icon}
                </div>
                <div className="flex-1">
                  <h2 className="text-2xl font-light mb-2 text-white group-hover:text-cyan-100 transition-colors">
                    {card.title}
                  </h2>
                  <p className="text-gray-400 text-sm group-hover:text-gray-200 transition-colors leading-relaxed">
                    {card.desc}
                  </p>
                </div>
                <div className="text-cyan-400/60 group-hover:text-cyan-300 transition-all duration-500 group-hover:translate-x-3 text-3xl">
                  →
                </div>
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-16 pt-8 border-t border-cyan-900/30 text-sm text-gray-400 flex items-center gap-4">
          <span className="status-dot"></span>
          System Active · Neural Networks Optimized · Ready for Deployment
        </div>
      </div>

      <style>{`
        .bubbles-container {
          position: absolute;
          width: 100%;
          height: 100%;
          overflow: hidden;
          pointer-events: none;
        }
        
        .bubble {
          position: absolute;
          bottom: -100px;
          background: radial-gradient(circle at 30% 30%, rgba(6, 182, 212, 0.5), rgba(6, 182, 212, 0.15));
          border-radius: 50%;
          animation: bubbleRise linear infinite;
          box-shadow: 0 0 15px rgba(6, 182, 212, 0.3);
        }
        
        @keyframes bubbleRise {
          0% {
            bottom: -50px;
            opacity: 0;
            transform: translateX(0);
          }
          10% {
            opacity: 0.8;
          }
          50% {
            transform: translateX(20px);
          }
          90% {
            opacity: 0.8;
          }
          100% {
            bottom: 110vh;
            opacity: 0;
            transform: translateX(-15px);
          }
        }
        
        .llm-gradient {
          background: linear-gradient(120deg, #06b6d4, #3b82f6, #8b5cf6, #06b6d4);
          background-size: 300% 100%;
          -webkit-background-clip: text;
          background-clip: text;
          -webkit-text-fill-color: transparent;
          animation: llmShift 4s ease infinite;
          font-weight: 300;
        }
        
        @keyframes llmShift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        .card-glow {
          position: absolute;
          inset: -2px;
          background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.5), transparent);
          opacity: 0;
          filter: blur(10px);
          transition: opacity 0.5s;
          animation: cardGlowMove 3s linear infinite;
        }
        
        .card-wrapper:hover .card-glow {
          opacity: 1;
        }
        
        @keyframes cardGlowMove {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        
        .processing-dot {
          width: 8px;
          height: 8px;
          background: #06b6d4;
          border-radius: 50%;
          animation: processingPulse 1.5s ease-in-out infinite;
          box-shadow: 0 0 20px #06b6d4, 0 0 40px rgba(6, 182, 212, 0.5);
        }
        
        @keyframes processingPulse {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.6); opacity: 0.5; }
        }
        
        .status-dot {
          width: 8px;
          height: 8px;
          background: #22c55e;
          border-radius: 50%;
          animation: statusPulse 2s ease-in-out infinite;
          box-shadow: 0 0 20px #22c55e, 0 0 40px rgba(34, 197, 94, 0.5);
        }
        
        @keyframes statusPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(1.4); }
        }
      `}</style>
    </div>
  );
};

export default Dashboard;