"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Search,
  FileText,
  Download,
  Send,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sparkles,
  Copy,
  Check,
  ChevronRight,
  BookOpen,
  Scale,
  RefreshCw,
  ExternalLink,
  ShieldCheck,
  Info
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// -----------------------------------------------------------------------------
// Types & Interfaces
// -----------------------------------------------------------------------------
interface Citation {
  section: string;
  title: string;
  page: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  citations?: Citation[];
  latency?: number;
}

// Key CCPA Sections for Quick Jump Navigation
const CCPA_SECTIONS = [
  { id: "1798.100", title: "§ 1798.100 - General Duties & Notice", page: 1 },
  { id: "1798.105", title: "§ 1798.105 - Consumer Right to Deletion", page: 3 },
  { id: "1798.110", title: "§ 1798.110 - Right to Know Information", page: 5 },
  { id: "1798.120", title: "§ 1798.120 - Right to Opt-Out of Sale/Share", page: 8 },
  { id: "1798.130", title: "§ 1798.130 - Notice & Response Procedures", page: 11 },
  { id: "1798.140", title: "§ 1798.140 - Key Definitions & Thresholds", page: 15 },
  { id: "1798.150", title: "§ 1798.150 - Private Right of Action", page: 20 },
];

const SUGGESTED_PROMPTS = [
  "What are the consumer opt-out requirements for data selling?",
  "Explain the business obligations under Section 1798.100.",
  "What is the cure period and penalties for non-compliance?",
  "How does CCPA define 'Personal Information'?",
];

export default function ComplianceLensApp() {
  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Document Viewer State
  const [zoom, setZoom] = useState(100);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedSection, setSelectedSection] = useState("");
  const [activeTab, setActiveTab] = useState<"doc" | "index">("doc");

  const chatBottomRef = useRef<HTMLDivElement>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleDownloadPDF = () => {
    const link = document.createElement("a");
    link.href = "/ccpa_document.pdf";
    link.download = "California_Consumer_Privacy_Act_CCPA.pdf";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handlePageJump = (page: number, sectionId?: string) => {
    setCurrentPage(page);
    if (sectionId) setSelectedSection(sectionId);
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || input;
    if (!textToSend.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage]);
    if (!queryText) setInput("");
    setIsLoading(true);

    const startTime = Date.now();

    try {
      const response = await fetch(`${apiUrl}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: textToSend }),
      });

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }

      const data = await response.json();
      const endTime = Date.now();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer || data.response || data.result || "No response content generated.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        latency: Math.round(endTime - startTime),
        citations: data.sources || [
          { section: "§ 1798.100", title: "General Duties", page: 1 },
          { section: "§ 1798.120", title: "Right to Opt-Out", page: 8 },
        ],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "⚠️ **Unable to connect to ComplianceLens backend API.**\n\nPlease ensure your FastAPI service is running at `http://localhost:8000`.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-100 font-sans text-slate-800 overflow-hidden">
      {/* =================================================================== */}
      {/* HEADER BAR                                                          */}
      {/* =================================================================== */}
      <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 z-10 shadow-sm">
        {/* Logo & App Branding */}
        <div className="flex items-center space-x-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-sky-500 shadow-md shadow-blue-500/20 text-white">
            {/* Custom Lens/Shield SVG Logo */}
            <svg
              className="w-6 h-6"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              <circle cx="12" cy="11" r="3" />
              <path d="M12 8v0" />
            </svg>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-extrabold tracking-tight text-slate-900">
                Compliance<span className="text-blue-600">Lens</span>
              </span>
              <span className="bg-blue-50 text-blue-700 text-[10px] font-semibold px-2 py-0.5 rounded-full border border-blue-200">
                CCPA 2026
              </span>
            </div>
            <p className="text-xs text-slate-500">AI Legal Intelligence & Audit Platform</p>
          </div>
        </div>

        {/* Center Title Badge */}
        <div className="hidden md:flex items-center space-x-2 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 text-xs text-slate-600">
          <Scale className="w-4 h-4 text-blue-600" />
          <span>California Consumer Privacy Act (Cal. Civ. Code § 1798.100 et seq.)</span>
        </div>

        {/* Header Actions */}
        <div className="flex items-center space-x-3">
          <button
            onClick={handleDownloadPDF}
            className="flex items-center space-x-2 bg-slate-900 hover:bg-blue-600 text-white px-3.5 py-2 rounded-lg text-xs font-medium transition-all shadow-sm active:scale-95"
          >
            <Download className="w-4 h-4" />
            <span>Download Official Act</span>
          </button>
        </div>
      </header>

      {/* =================================================================== */}
      {/* MAIN DUAL-PANE BODY                                                 */}
      {/* =================================================================== */}
      <div className="flex-1 flex overflow-hidden">
        {/* ----------------------------------------------------------------- */}
        {/* LEFT PANE: LIVE CCPA DOCUMENT VIEWER                              */}
        {/* ----------------------------------------------------------------- */}
        <div className="w-1/2 border-r border-slate-200 bg-slate-50 flex flex-col">
          {/* Document Controls Bar */}
          <div className="h-12 bg-white border-b border-slate-200 px-4 flex items-center justify-between">
            {/* View Switcher */}
            <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-lg text-xs font-medium">
              <button
                onClick={() => setActiveTab("doc")}
                className={`px-3 py-1 rounded-md transition-all ${
                  activeTab === "doc"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Document Viewer
              </button>
              <button
                onClick={() => setActiveTab("index")}
                className={`px-3 py-1 rounded-md transition-all ${
                  activeTab === "index"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Section Index
              </button>
            </div>

            {/* Quick Section Selector */}
            <div className="flex items-center space-x-2">
              <select
                value={selectedSection}
                onChange={(e) => {
                  const sec = CCPA_SECTIONS.find((s) => s.id === e.target.value);
                  if (sec) handlePageJump(sec.page, sec.id);
                }}
                className="bg-slate-50 border border-slate-300 rounded-lg text-xs px-2.5 py-1 text-slate-700 focus:ring-2 focus:ring-blue-500 focus:outline-none"
              >
                <option value="">Jump to Statute Section...</option>
                {CCPA_SECTIONS.map((sec) => (
                  <option key={sec.id} value={sec.id}>
                    {sec.title}
                  </option>
                ))}
              </select>

              {/* Zoom Controls */}
              <div className="flex items-center space-x-1 bg-slate-100 rounded-lg p-1 border border-slate-200 text-slate-600">
                <button
                  onClick={() => setZoom((z) => Math.max(z - 10, 60))}
                  className="p-1 hover:bg-white rounded text-slate-700 transition-all"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <span className="text-[11px] font-semibold px-1 text-slate-600">{zoom}%</span>
                <button
                  onClick={() => setZoom((z) => Math.min(z + 10, 180))}
                  className="p-1 hover:bg-white rounded text-slate-700 transition-all"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoom(100)}
                  className="p-1 hover:bg-white rounded text-slate-700 transition-all"
                  title="Reset Zoom"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

          {/* Document Content Panel */}
          <div className="flex-1 overflow-y-auto p-4 flex justify-center bg-slate-200/70 relative">
            {activeTab === "doc" ? (
              <div
                className="w-full h-full max-w-3xl bg-white shadow-lg rounded-xl overflow-hidden border border-slate-300 transition-all duration-200"
                style={{
                  transform: `scale(${zoom / 100})`,
                  transformOrigin: "top center",
                }}
              >
                {/* Embedded Native Browser PDF Viewer */}
                <iframe
                  src={`/ccpa_document.pdf#page=${currentPage}`}
                  className="w-full h-full border-none"
                  title="CCPA Official Text"
                />
              </div>
            ) : (
              /* Section Index Table */
              <div className="w-full max-w-2xl bg-white rounded-xl shadow-sm border border-slate-200 p-6 overflow-y-auto">
                <div className="flex items-center space-x-2 mb-4 pb-3 border-b border-slate-100">
                  <BookOpen className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-slate-900 text-sm">
                    Statutory Navigation Index
                  </h3>
                </div>
                <div className="space-y-2">
                  {CCPA_SECTIONS.map((sec) => (
                    <div
                      key={sec.id}
                      onClick={() => {
                        handlePageJump(sec.page, sec.id);
                        setActiveTab("doc");
                      }}
                      className="p-3 rounded-lg border border-slate-100 hover:border-blue-300 hover:bg-blue-50/50 cursor-pointer transition-all flex items-center justify-between group"
                    >
                      <div>
                        <div className="font-medium text-xs text-slate-800 group-hover:text-blue-700">
                          {sec.title}
                        </div>
                        <div className="text-[11px] text-slate-400 mt-0.5">
                          California Civil Code Section {sec.id}
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-[10px] bg-slate-100 group-hover:bg-blue-100 text-slate-600 group-hover:text-blue-800 px-2 py-0.5 rounded font-mono">
                          Page {sec.page}
                        </span>
                        <ChevronRight className="w-4 h-4 text-slate-400 group-hover:text-blue-600" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ----------------------------------------------------------------- */}
        {/* RIGHT PANE: AI RAG CHAT ASSISTANT                                 */}
        {/* ----------------------------------------------------------------- */}
        <div className="w-1/2 bg-white flex flex-col">
          {/* Sub-Header */}
          <div className="h-12 border-b border-slate-200 px-6 flex items-center justify-between bg-slate-50/50">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-blue-600" />
              <span className="font-semibold text-xs text-slate-800">Compliance Assistant</span>
            </div>
            {messages.length > 0 && (
              <button
                onClick={() => setMessages([])}
                className="text-[11px] text-slate-400 hover:text-slate-700 flex items-center space-x-1 transition-colors"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Clear History</span>
              </button>
            )}
          </div>

          {/* Chat Messages Body */}
          <div className="flex-1 overflow-y-auto p-6 space-y-5 bg-slate-50/30">
            {messages.length === 0 ? (
              /* Welcome Screen */
              <div className="h-full flex flex-col items-center justify-center text-center px-6">
                <div className="w-14 h-14 bg-blue-50 rounded-2xl flex items-center justify-center text-blue-600 mb-4 border border-blue-100 shadow-sm">
                  <ShieldCheck className="w-7 h-7" />
                </div>
                <h2 className="text-lg font-bold text-slate-900 mb-1">
                  How can I assist with CCPA compliance today?
                </h2>
                <p className="text-xs text-slate-500 max-w-md mb-6 leading-relaxed">
                  Ask detailed questions regarding consumer privacy rights, business obligations,
                  opt-out mechanisms, or statutory non-compliance penalties.
                </p>

                {/* Suggested Prompts Grid */}
                <div className="grid grid-cols-1 gap-2 w-full max-w-md">
                  {SUGGESTED_PROMPTS.map((prompt, i) => (
                    <button
                      key={i}
                      onClick={() => handleSend(prompt)}
                      className="p-3 text-left bg-white border border-slate-200 hover:border-blue-400 hover:shadow-sm rounded-xl text-xs text-slate-700 transition-all flex items-center justify-between group"
                    >
                      <span>{prompt}</span>
                      <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-blue-600 transition-colors" />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* Conversation Stream */
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl p-4 text-xs leading-relaxed transition-all shadow-sm ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white rounded-br-none"
                        : "bg-white border border-slate-200 text-slate-800 rounded-bl-none"
                    }`}
                  >
                    {/* Role Header */}
                    <div className="flex items-center justify-between mb-2 pb-1 border-b border-slate-100/20 text-[10px] text-slate-400">
                      <span className="font-semibold uppercase tracking-wider">
                        {msg.role === "user" ? "Compliance Officer" : "ComplianceLens AI"}
                      </span>
                      <div className="flex items-center space-x-2">
                        {msg.latency && (
                          <span className="text-[9px] bg-slate-100 px-1.5 py-0.5 rounded text-slate-500 font-mono">
                            {msg.latency}ms
                          </span>
                        )}
                        <span>{msg.timestamp}</span>
                      </div>
                    </div>

                    {/* Content Markdown */}
                    <div className="prose prose-xs max-w-none text-xs">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    </div>

                    {/* Assistant Citations & Actions */}
                    {msg.role === "assistant" && (
                      <div className="mt-3 pt-3 border-t border-slate-100 flex flex-col space-y-2">
                        {msg.citations && msg.citations.length > 0 && (
                          <div>
                            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                              Verified Source Statutory Citations:
                            </span>
                            <div className="flex flex-wrap gap-1.5">
                              {msg.citations.map((cit, idx) => (
                                <button
                                  key={idx}
                                  onClick={() => {
                                    handlePageJump(cit.page, cit.section);
                                    setActiveTab("doc");
                                  }}
                                  className="inline-flex items-center space-x-1 bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-800 px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
                                >
                                  <FileText className="w-3 h-3" />
                                  <span>{cit.section}</span>
                                  <span className="text-blue-500 font-mono">(p. {cit.page})</span>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="flex justify-end pt-1">
                          <button
                            onClick={() => copyToClipboard(msg.content, msg.id)}
                            className="text-[10px] text-slate-400 hover:text-slate-700 flex items-center space-x-1"
                          >
                            {copiedId === msg.id ? (
                              <>
                                <Check className="w-3 h-3 text-emerald-600" />
                                <span className="text-emerald-600 font-medium">Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3 h-3" />
                                <span>Copy Response</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {/* Loading / Thinking Indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-none p-4 shadow-sm text-xs text-slate-500 flex items-center space-x-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-ping" />
                  <span>Searching ChromaDB & evaluating CCPA legal context...</span>
                </div>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>

          {/* Chat Input Bar */}
          <div className="p-4 border-t border-slate-200 bg-white">
            <div className="relative flex items-center bg-slate-50 border border-slate-300 rounded-xl focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent transition-all shadow-sm">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder="Ask a question about CCPA compliance (e.g. 'What is the opt-out rule?')..."
                className="w-full bg-transparent px-4 py-3 text-xs text-slate-900 placeholder-slate-400 focus:outline-none"
              />
              <button
                onClick={() => handleSend()}
                disabled={isLoading || !input.trim()}
                className="mr-2 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg transition-all shadow-sm active:scale-95"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="flex items-center justify-between mt-2 text-[10px] text-slate-400 px-1">
              <span>Press Enter ↵ to submit</span>
              <span className="flex items-center space-x-1">
                <Info className="w-3 h-3 text-slate-400" />
                <span>RAG powered by LangChain & ChromaDB</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}