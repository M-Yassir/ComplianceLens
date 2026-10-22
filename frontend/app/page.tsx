"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  FileText,
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
  ShieldCheck,
  Info,
  PanelLeftClose,
  PanelLeftOpen,
  GripVertical,
  X,
  GitCompare,
  FileSearch,
  CheckCircle2
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ComplianceLensLogo from "./components/logo";

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
  "What changed between the 2023 CCPA baseline and the 2026 rules?",
  "What are the consumer opt-out requirements for data selling?",
  "Explain the business obligations under Section 1798.100.",
  "What is the cure period and penalties for non-compliance?",
];

export default function ComplianceLensApp() {
  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Pane Visibility, Resizing & Modal State
  const [isDocPaneOpen, setIsDocPaneOpen] = useState(true);
  const [leftWidth, setLeftWidth] = useState(50); // percentage (20% to 75%)
  const [isResizing, setIsResizing] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);

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

  // Keyboard shortcut (Escape to close About Modal)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isAboutOpen) {
        setIsAboutOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isAboutOpen]);

  // Handle Drag Resizing of Dual Panes
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = (e.clientX / window.innerWidth) * 100;
      if (newWidth >= 20 && newWidth <= 75) {
        setLeftWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handlePageJump = (page: number, sectionId?: string) => {
    if (!isDocPaneOpen) setIsDocPaneOpen(true);
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
    <div
      className={`flex flex-col h-screen w-screen bg-slate-100 font-sans text-slate-800 overflow-hidden ${
        isResizing ? "select-none cursor-col-resize" : ""
      }`}
    >
      {/* =================================================================== */}
      {/* HEADER BAR                                                          */}
      {/* =================================================================== */}
      <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 z-10 shadow-sm shrink-0">
        {/* Logo & App Branding */}
        <div className="flex items-center space-x-3">
          <ComplianceLensLogo className="w-10 h-10 shrink-0 drop-shadow-sm" />
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

        {/* Center Statute Badge */}
        <div className="hidden lg:flex items-center space-x-2 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 text-xs text-slate-600">
          <Scale className="w-4 h-4 text-blue-600" />
          <span>California Consumer Privacy Act (Cal. Civ. Code § 1798.100 et seq.)</span>
        </div>

        {/* Header Actions */}
        <div className="flex items-center space-x-2.5">
          {/* About / Info Button */}
          <button
            onClick={() => setIsAboutOpen(true)}
            className="flex items-center space-x-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 px-3 py-2 rounded-lg text-xs font-medium transition-all shadow-sm active:scale-95"
            title="Learn how ComplianceLens works"
          >
            <Info className="w-4 h-4 text-blue-600" />
            <span>About</span>
          </button>

          {/* Document Viewer Toggle */}
          <button
            onClick={() => setIsDocPaneOpen((prev) => !prev)}
            className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all shadow-sm active:scale-95 ${
              isDocPaneOpen
                ? "bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300"
                : "bg-blue-600 hover:bg-blue-700 text-white"
            }`}
          >
            {isDocPaneOpen ? (
              <>
                <PanelLeftClose className="w-4 h-4" />
                <span className="hidden sm:inline">Hide Document</span>
              </>
            ) : (
              <>
                <PanelLeftOpen className="w-4 h-4" />
                <span className="hidden sm:inline">Show Document Viewer</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* =================================================================== */}
      {/* MAIN DUAL-PANE BODY                                                 */}
      {/* =================================================================== */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* ----------------------------------------------------------------- */}
        {/* LEFT PANE: LIVE CCPA DOCUMENT VIEWER                              */}
        {/* ----------------------------------------------------------------- */}
        {isDocPaneOpen && (
          <div
            style={{ width: `${leftWidth}%` }}
            className="bg-slate-50 flex flex-col h-full border-r border-slate-200 shrink-0"
          >
            {/* Document Controls Bar */}
            <div className="h-12 bg-white border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
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

              {/* Quick Section Selector & Zoom */}
              <div className="flex items-center space-x-2">
                <select
                  value={selectedSection}
                  onChange={(e) => {
                    const sec = CCPA_SECTIONS.find((s) => s.id === e.target.value);
                    if (sec) handlePageJump(sec.page, sec.id);
                  }}
                  className="bg-slate-50 border border-slate-300 rounded-lg text-xs px-2 py-1 text-slate-700 focus:ring-2 focus:ring-blue-500 focus:outline-none max-w-[170px] truncate"
                >
                  <option value="">Jump to Section...</option>
                  {CCPA_SECTIONS.map((sec) => (
                    <option key={sec.id} value={sec.id}>
                      {sec.title}
                    </option>
                  ))}
                </select>

                {/* Zoom Controls */}
                <div className="hidden sm:flex items-center space-x-1 bg-slate-100 rounded-lg p-1 border border-slate-200 text-slate-600">
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
                  className="w-full h-full bg-white shadow-lg rounded-xl overflow-hidden border border-slate-300 transition-all duration-200"
                  style={{
                    transform: `scale(${zoom / 100})`,
                    transformOrigin: "top center",
                  }}
                >
                  <iframe
                    src={`/ccpa_document.pdf#page=${currentPage}`}
                    className={`w-full h-full border-none ${
                      isResizing ? "pointer-events-none" : ""
                    }`}
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
        )}

        {/* ----------------------------------------------------------------- */}
        {/* DRAG RESIZE HANDLE                                                */}
        {/* ----------------------------------------------------------------- */}
        {isDocPaneOpen && (
          <div
            onMouseDown={() => setIsResizing(true)}
            className="w-2 hover:w-3 bg-slate-200 hover:bg-blue-500 cursor-col-resize flex items-center justify-center group transition-all shrink-0 z-20 border-x border-slate-300/50"
            title="Drag to resize windows"
          >
            <GripVertical className="w-3 h-3 text-slate-400 group-hover:text-white" />
          </div>
        )}

        {/* ----------------------------------------------------------------- */}
        {/* RIGHT PANE: AI RAG CHAT ASSISTANT                                 */}
        {/* ----------------------------------------------------------------- */}
        <div
          style={{ width: isDocPaneOpen ? `${100 - leftWidth}%` : "100%" }}
          className="bg-white flex flex-col h-full flex-1 transition-all duration-75"
        >
          {/* Sub-Header */}
          <div className="h-12 border-b border-slate-200 px-6 flex items-center justify-between bg-slate-50/50 shrink-0">
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
                  Ask questions regarding consumer rights, opt-out policies, statutory exemptions,
                  or compare current rules against earlier baselines.
                </p>

                {/* Suggested Prompts */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 w-full max-w-xl">
                  {SUGGESTED_PROMPTS.map((prompt, i) => (
                    <button
                      key={i}
                      onClick={() => handleSend(prompt)}
                      className="p-3 text-left bg-white border border-slate-200 hover:border-blue-400 hover:shadow-sm rounded-xl text-xs text-slate-700 transition-all flex items-center justify-between group"
                    >
                      <span>{prompt}</span>
                      <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-blue-600 transition-colors shrink-0 ml-2" />
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

                    {/* Assistant Source Citations */}
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
                                  onClick={() => handlePageJump(cit.page, cit.section)}
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

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-none p-4 shadow-sm text-xs text-slate-500 flex items-center space-x-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-ping" />
                  <span>Searching & evaluating CCPA legal context...</span>
                </div>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>

          {/* Chat Input Bar */}
          <div className="p-4 border-t border-slate-200 bg-white shrink-0">
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
              
            </div>
          </div>
        </div>
      </div>

      {/* =================================================================== */}
      {/* ABOUT COMPLIANCE LENS MODAL OVERLAY                                 */}
      {/* =================================================================== */}
      {isAboutOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-white border border-slate-200 rounded-2xl shadow-2xl max-w-xl w-full p-6 space-y-6 relative overflow-hidden"
          >
            {/* Close Button */}
            <button
              onClick={() => setIsAboutOpen(false)}
              className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-all"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Modal Header */}
            <div className="flex items-center space-x-3 pb-4 border-b border-slate-100">
              <ComplianceLensLogo className="w-10 h-10 shrink-0" />
              <div>
                <h3 className="text-lg font-bold text-slate-900">About ComplianceLens</h3>
                <p className="text-xs text-slate-500">
                  Interactive Legal Intelligence & Policy Audit Platform
                </p>
              </div>
            </div>

            {/* Platform Purpose Description */}
            <p className="text-xs text-slate-600 leading-relaxed">
              <strong>ComplianceLens</strong> simplifies complex legal documents by pairing a live,
              searchable copy of the California Consumer Privacy Act (CCPA) with an intelligent
              conversational assistant.
            </p>

            {/* Core Features Grid */}
            <div className="space-y-3">
              {/* Pillar 1 */}
              <div className="flex items-start space-x-3 p-3 bg-blue-50/60 rounded-xl border border-blue-100">
                <FileSearch className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-slate-900">
                    Interactive Policy Queries
                  </h4>
                  <p className="text-[11px] text-slate-600 mt-0.5 leading-normal">
                    Ask plain-English questions to search the CCPA for specific rules on consumer rights,
                    business liability, opt-out mechanisms, and enforcement thresholds.
                  </p>
                </div>
              </div>

              {/* Pillar 2 */}
              <div className="flex items-start space-x-3 p-3 bg-indigo-50/60 rounded-xl border border-indigo-100">
                <GitCompare className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-slate-900">
                    Version Comparison (2023 vs. 2026)
                  </h4>
                  <p className="text-[11px] text-slate-600 mt-0.5 leading-normal">
                    Track statutory evolution over time. Instantly identify what changed between the 2023
                    enforcement rules and updated 2026 standards, including expanded consumer protections and audit requirements.
                  </p>
                </div>
              </div>

              {/* Pillar 3 */}
              <div className="flex items-start space-x-3 p-3 bg-sky-50/60 rounded-xl border border-sky-100">
                <BookOpen className="w-5 h-5 text-sky-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-slate-900">
                    Live Section & Subsection References
                  </h4>
                  <p className="text-[11px] text-slate-600 mt-0.5 leading-normal">
                    Verify every answer against official legal text. When the chatbot cites a provision, click the
                    citation badge to view the exact page, section, and subsection in the live document window.
                  </p>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
              <span className="text-[10px] text-slate-400">
                Cal. Civ. Code § 1798.100 et seq.
              </span>
              <button
                onClick={() => setIsAboutOpen(false)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-xs font-medium transition-all shadow-sm active:scale-95"
              >
                Get Started
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}