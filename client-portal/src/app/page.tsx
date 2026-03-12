"use client";

import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 selection:bg-indigo-500/30 font-sans">
      {/* Background Glow Effects */}
      <div className="absolute top-0 inset-x-0 h-64 bg-indigo-500/10 blur-[100px] -z-10" />
      <div className="absolute top-40 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-purple-600/10 blur-[120px] rounded-full -z-10" />
      
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto backdrop-blur-sm border-b border-white/5 sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-lg shadow-[0_0_15px_rgba(99,102,241,0.5)]">
            A
          </div>
          <span className="font-semibold text-xl tracking-tight">Antigravity<span className="text-indigo-400">Agent</span></span>
        </div>
        <nav className="flex items-center gap-6 text-sm font-medium text-slate-300">
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="#pricing" className="hover:text-white transition-colors">Pricing</a>
          <button className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-full transition-all">Sign In</button>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-8 pt-24 pb-32 flex flex-col items-center text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm font-medium mb-8">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
          </span>
          MCP SaaS Now Available
        </div>
        
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-8 leading-tight">
          Supercharge your IDE with <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">
            100% Private AI
          </span>
        </h1>
        
        <p className="text-lg md:text-xl text-slate-400 max-w-2xl mb-12">
          Connect our hosted Mistral NeMo 12B agent directly to your IDE via the Model Context Protocol. Zero data retention. Infinite possibilities.
        </p>

        <div className="flex flex-col sm:flex-row gap-4">
          <a href="#pricing" className="px-8 py-4 bg-white text-slate-950 font-semibold rounded-full hover:scale-105 transition-transform shadow-[0_0_30px_rgba(255,255,255,0.2)]">
            Get your License Key
          </a>
          <a href="https://github.com/modelcontextprotocol" target="_blank" className="px-8 py-4 bg-white/5 border border-white/10 font-semibold rounded-full hover:bg-white/10 transition-colors">
            Read Documentation
          </a>
        </div>
      </main>

      {/* Pricing Section (Stripe Integration Target) */}
      <section id="pricing" className="max-w-7xl mx-auto px-8 py-24 border-t border-white/5 relative">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-500/10 blur-[150px] -z-10 rounded-full" />
        
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-4">Transparent Pricing</h2>
          <p className="text-slate-400 max-w-xl mx-auto">Pay securely with Stripe. Instant access to your MCP endpoint. Cancel anytime.</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {/* Pro Plan */}
          <div className="relative group rounded-3xl bg-slate-900/50 border border-white/10 p-8 backdrop-blur-sm overflow-hidden transition-all hover:border-indigo-500/50">
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="relative z-10">
              <h3 className="text-xl font-semibold mb-2">Pro Developer</h3>
              <p className="text-slate-400 text-sm mb-6">For individual developers requiring ultimate privacy.</p>
              <div className="mb-6">
                <span className="text-5xl font-bold">$29</span>
                <span className="text-slate-400">/mo</span>
              </div>
              <ul className="space-y-3 mb-8 text-slate-300 text-sm">
                <li className="flex items-center gap-2">✓ Unlimited Agent Interactions</li>
                <li className="flex items-center gap-2">✓ Dedicated Mistral NeMo 12B Process</li>
                <li className="flex items-center gap-2">✓ FastMCP Endpoint</li>
                <li className="flex items-center gap-2">✓ Standard Support</li>
              </ul>
              <button 
                className="w-full py-3 rounded-xl bg-white/5 border border-white/10 font-medium hover:bg-white/10 transition-colors"
                onClick={() => alert("Redirects to Stripe Checkout...")}
              >
                Subscribe with Stripe
              </button>
            </div>
          </div>

          {/* Enterprise Plan */}
          <div className="relative group rounded-3xl bg-gradient-to-b from-indigo-900/40 to-slate-900/50 border border-indigo-500/30 p-8 backdrop-blur-sm overflow-hidden shadow-[0_0_40px_rgba(99,102,241,0.1)]">
            <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-indigo-500 to-purple-500" />
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/10 to-transparent opacity-100" />
            
            <div className="relative z-10">
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-xl font-semibold text-white">Enterprise Teams</h3>
                <span className="px-2 py-1 text-xs font-semibold bg-indigo-500 text-white rounded-full">POPULAR</span>
              </div>
              <p className="text-slate-400 text-sm mb-6">Fully managed Hosted DevContainers included.</p>
              <div className="mb-6">
                <span className="text-5xl font-bold">$149</span>
                <span className="text-slate-400">/user/mo</span>
              </div>
              <ul className="space-y-3 mb-8 text-slate-300 text-sm">
                <li className="flex items-center gap-2 text-white font-medium">✨ Everything in Pro, plus:</li>
                <li className="flex items-center gap-2">✓ Hosted Workspace (.devcontainer)</li>
                <li className="flex items-center gap-2">✓ Custom API Integrations</li>
                <li className="flex items-center gap-2">✓ SLA 99.9% Uptime</li>
                <li className="flex items-center gap-2">✓ Priority 24/7 Support</li>
              </ul>
              <button 
                className="w-full py-3 rounded-xl bg-indigo-500 text-white font-medium hover:bg-indigo-600 transition-colors shadow-lg shadow-indigo-500/25"
                onClick={() => alert("Redirects to Stripe Checkout...")}
              >
                Contact Sales
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 mt-24 py-12 px-8 text-center text-slate-500 text-sm">
        <p>© 2026 Antigravity MCP Agent SaaS. Built with Next.js, FastMCP, and TailwindCSS.</p>
      </footer>
    </div>
  );
}
