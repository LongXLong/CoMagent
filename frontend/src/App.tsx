import { useState, useRef, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { Bot, Sparkles, BrainCircuit, Play, CheckCircle2, AlertCircle, Loader2, MessageSquareText, PanelRightClose, PanelRightOpen, ChevronDown, PanelLeftClose, PanelLeftOpen, MessageSquare, Settings2, Moon, Sun, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types
interface Agent {
  id: string;
  label: string;
}

interface SuggestResponse {
  initial_answer: string;
  question_type: string;
  candidate_question_types: string[];
  suggested_agents: string[];
  all_agents: string[];
  all_agents_with_labels?: Agent[];
}

interface Feedback {
  agent: string;
  comment: string;
}

interface RunResponse {
  final_answer: string;
  expert_feedbacks: Feedback[];
  general_feedbacks: Feedback[];
  success: boolean;
  update_ltm: boolean;
  update_mk: boolean;
}

const MAX_GENERAL_AGENTS = 3;

const GENERAL_AGENTS: Agent[] = [
  { id: 'logic_checker', label: 'Logic Checker' },
  { id: 'clarity_editor', label: 'Clarity Editor' },
  { id: 'completeness_checker', label: 'Completeness Checker' },
  { id: 'evidence_checker', label: 'Evidence Checker' },
  { id: 'brevity_advisor', label: 'Brevity Advisor' },
  { id: 'consistency_checker', label: 'Consistency Checker' },
  { id: 'relevancy_checker', label: 'Relevancy Checker' },
  { id: 'harmlessness_checker', label: 'Harmlessness Checker' },
  { id: 'compliance_checker', label: 'Compliance Checker' },
  { id: 'fluency_editor', label: 'Fluency Editor' },
];

const GENERAL_AGENT_LABEL_BY_ID: Record<string, string> = Object.fromEntries(
  GENERAL_AGENTS.map((a) => [a.id, a.label]),
);

/** Expert `agent` ids from backend — same readable English style as General Agents. */
const EXPERT_AGENT_LABEL_BY_ID: Record<string, string> = {
  math_computation_expert: 'Math Computation Expert',
  knowledge_qa_expert: 'Knowledge Q&A Expert',
  logical_reasoning_expert: 'Logical Reasoning Expert',
  code_development_expert: 'Code Development Expert',
  text_writing_expert: 'Text Writing Expert',
  translation_localization_expert: 'Translation & Localization Expert',
  summarization_expert: 'Summarization Expert',
  creative_ideation_expert: 'Creative Ideation expert',
  marketing_copywriting_expert: 'Marketing & Copywriting Expert',
  career_business_expert: 'Career & Business Expert',
  educational_tutoring_expert: 'Educational Tutoring Expert',
  life_emotional_expert: 'Life & Emotional Expert',
  role_playing_expert: 'Role-playing Expert',
  multimodal_expert: 'Multimodal Expert',
  data_processing_expert: 'Data Processing Expert',
  other_general_q_expert: 'General Q&A Expert',
  base_expert: 'Expert',
};

function displayFeedbackAgentName(raw: string): string {
  const id = raw.trim();
  if (!id) return 'Unknown';
  if (GENERAL_AGENT_LABEL_BY_ID[id]) return GENERAL_AGENT_LABEL_BY_ID[id];
  if (EXPERT_AGENT_LABEL_BY_ID[id]) return EXPERT_AGENT_LABEL_BY_ID[id];
  return id
    .split('_')
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

function App() {
  const [question, setQuestion] = useState('');
  const [loadingSuggest, setLoadingSuggest] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [suggestData, setSuggestData] = useState<SuggestResponse | null>(null);
  const [selectedAgents, setSelectedAgents] = useState<Set<string>>(new Set());
  const [updateLtm, setUpdateLtm] = useState(true);
  const [updateMk, setUpdateMk] = useState(true);

  const [runData, setRunData] = useState<RunResponse | null>(null);

  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(true);
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(true);
  const [activePage, setActivePage] = useState<'chat' | 'model_settings'>('chat');
  const [isSettingsExpanded, setIsSettingsExpanded] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  useLayoutEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
    } else {
      root.classList.remove('dark');
      root.classList.add('light');
    }
  }, [theme]);

  // Model settings state
  const [apiKey, setApiKey] = useState('');
  const [temperature, setTemperature] = useState(0.7);

  const toggleTheme = () => {
    setTheme((t) => (t === 'light' ? 'dark' : 'light'));
  };

  const [models] = useState([
    { id: 'gpt-5', label: 'GPT-5' },
    { id: 'gpt-5-mini', label: 'GPT-5 Mini' },
    { id: 'gpt-4.1', label: 'GPT-4.1' },
    { id: 'gpt-4.1-mini', label: 'GPT-4.1 Mini' },
    { id: 'gpt-4', label: 'GPT-4' },
    { id: 'gpt-4o', label: 'GPT-4o' },
    { id: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
  ]);
  const [selectedModel, setSelectedModel] = useState(models[0].id);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const modelBtnRef = useRef<HTMLButtonElement>(null);
  const [modelMenuPos, setModelMenuPos] = useState<{ top: number; left: number; width: number } | null>(null);

  const updateModelMenuPos = () => {
    const el = modelBtnRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const menuWidth = Math.max(192, rect.width);
    const maxH = Math.min(280, window.innerHeight * 0.45);
    let top = rect.bottom + 8;
    if (top + maxH > window.innerHeight - 8) {
      top = rect.top - 8 - maxH;
    }
    if (top < 8) top = 8;
    setModelMenuPos({ top, left: rect.left, width: menuWidth });
  };

  useLayoutEffect(() => {
    if (!isModelDropdownOpen) return;
    updateModelMenuPos();
    const onScroll = () => updateModelMenuPos();
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', updateModelMenuPos);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', updateModelMenuPos);
    };
  }, [isModelDropdownOpen]);

  const [availableAgents] = useState<Agent[]>(GENERAL_AGENTS);

  const API_KEY_REQUIRED_MESSAGE =
    'Please first go to the "Model Settings" page in the sidebar on the left, enter your API key, and try again.';

  const handleSuggest = async () => {
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }
    if (!apiKey.trim()) {
      setError(API_KEY_REQUIRED_MESSAGE);
      return;
    }
    setError(null);
    setLoadingSuggest(true);
    setRunData(null);
    setSelectedAgents(new Set()); // 清空之前勾选的agents
    
    try {
      const res = await fetch('/api/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          api_key: apiKey.trim() || null,
          temperature,
          model: selectedModel,
        }),
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to get suggestions');
      
      setSuggestData(data);
      const suggested = (data.suggested_agents || []).slice(0, MAX_GENERAL_AGENTS);
      setSelectedAgents(new Set(suggested));
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoadingSuggest(false);
    }
  };

  const handleRun = async () => {
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }
    if (!apiKey.trim()) {
      setError(API_KEY_REQUIRED_MESSAGE);
      return;
    }
    if (selectedAgents.size === 0) {
      setError("Please select at least one agent.");
      return;
    }

    setError(null);
    setLoadingRun(true);
    setRunData(null); // 立即清空右侧栏
    
    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          selected_agents: Array.from(selectedAgents),
          update_ltm: updateLtm,
          update_mk: updateMk,
          api_key: apiKey.trim() || null,
          temperature,
          model: selectedModel,
          ...(suggestData
            ? {
                initial_answer: suggestData.initial_answer ?? null,
                question_type: suggestData.question_type ?? null,
                candidate_question_types:
                  suggestData.candidate_question_types ?? null,
              }
            : {}),
        }),
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Run failed');
      
      setRunData(data);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoadingRun(false);
    }
  };

  const toggleAgent = (id: string) => {
    const next = new Set(selectedAgents);
    if (next.has(id)) next.delete(id);
    else if (next.size < MAX_GENERAL_AGENTS) next.add(id);
    setSelectedAgents(next);
  };

  return (
    <div className="h-screen bg-background text-text selection:bg-accent/30 font-sans flex flex-col overflow-hidden">
      {/* Header */}
      <header className="sticky top-0 z-50 glass border-b border-border/50">
        <div className="w-full px-4 lg:px-6 h-16 flex items-center gap-3">
          <button
            onClick={() => setIsLeftSidebarOpen(!isLeftSidebarOpen)}
            className="p-2 rounded-lg hover:bg-surface transition-colors text-muted hover:text-text flex items-center justify-center mr-2 hidden lg:flex"
            title={isLeftSidebarOpen ? "Hide Navigation" : "Show Navigation"}
          >
            {isLeftSidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
          </button>

          <div className="w-8 h-8 rounded-xl bg-accent flex items-center justify-center text-white shadow-sm">
            <BrainCircuit size={18} />
          </div>
          <h1 className="text-lg font-semibold tracking-tight">Multi-Agent Reflection System</h1>
          <div className="ml-auto">
            <button
              onClick={() => setIsRightSidebarOpen(!isRightSidebarOpen)}
              className="p-2 rounded-lg hover:bg-surface transition-colors text-muted hover:text-text flex items-center justify-center"
              title={isRightSidebarOpen ? "Hide Feedbacks" : "Show Feedbacks"}
            >
              {isRightSidebarOpen ? <PanelRightClose size={20} /> : <PanelRightOpen size={20} />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full px-4 lg:px-6 pt-8 pb-0 flex gap-8 items-start overflow-hidden">
        
        {/* Navigation Sidebar */}
        <AnimatePresence initial={false}>
          {isLeftSidebarOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 260, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="shrink-0 flex flex-col gap-4 overflow-hidden hidden lg:flex"
            >
              <div className="w-[260px] flex flex-col h-full border-r border-border/50 pr-6 mr-2 pb-8">
                
                {/* Main Nav Items */}
                <div className="flex flex-col gap-2">
                  <button 
                    onClick={() => setActivePage('chat')}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors text-sm font-medium ${activePage === 'chat' ? 'bg-accent/10 text-accent' : 'text-text hover:bg-surface'}`}
                  >
                    <MessageSquare size={18} />
                    Chat
                  </button>
                  <button 
                    onClick={() => setActivePage('model_settings')}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors text-sm font-medium ${activePage === 'model_settings' ? 'bg-accent/10 text-accent' : 'text-text hover:bg-surface'}`}
                  >
                    <Settings2 size={18} />
                    Model Settings
                  </button>
                </div>

                <div className="mt-auto flex flex-col gap-2 pt-4 border-t border-border/50">
                  <button
                    onClick={() => setIsSettingsExpanded(!isSettingsExpanded)}
                    className="flex items-center justify-between px-4 py-3 rounded-xl transition-colors text-sm font-medium text-text hover:bg-surface"
                  >
                    <div className="flex items-center gap-3">
                      <Settings2 size={18} />
                      Settings
                    </div>
                    <motion.div animate={{ rotate: isSettingsExpanded ? 90 : 0 }}>
                      <ChevronRight size={16} className="text-muted" />
                    </motion.div>
                  </button>

                  <AnimatePresence>
                    {isSettingsExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden flex flex-col gap-1 px-2"
                      >
                        <button
                          onClick={toggleTheme}
                          className="flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors text-sm text-muted hover:text-text hover:bg-surface"
                        >
                          {theme === 'dark' ? <Moon size={16} /> : <Sun size={16} />}
                          Change Theme ({theme === 'dark' ? 'Dark' : 'Light'})
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {activePage === 'chat' ? (
          <>
            {/* Left Sidebar: Agent Selection (Always visible) */}
            <aside className="w-full lg:w-64 shrink-0 flex flex-col gap-4 h-full pb-8">
          <div className="bg-surface/50 border border-border/50 rounded-[1.5rem] p-5 shadow-sm h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-muted flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-success/80"></span>
                General Agents
              </h2>
            </div>
            <div className="text-xs text-muted mb-4 pb-4 border-b border-border/30">
              Select up to {MAX_GENERAL_AGENTS} agents for reflection. Recommended agents are highlighted.
            </div>
            <div className="flex flex-col gap-2 flex-1 overflow-y-auto pr-2 custom-scrollbar">
              {availableAgents.map((agent) => {
                const isSuggested = suggestData?.suggested_agents?.includes(agent.id);
                const isSelected = selectedAgents.has(agent.id);
                const atLimit = selectedAgents.size >= MAX_GENERAL_AGENTS;
                const disableSelect = !isSelected && atLimit;
                return (
                  <button
                    key={agent.id}
                    type="button"
                    disabled={disableSelect}
                    title={
                      disableSelect
                        ? `最多选择 ${MAX_GENERAL_AGENTS} 个 General Agents，请先取消勾选再选其他项`
                        : undefined
                    }
                    onClick={() => toggleAgent(agent.id)}
                    className={`flex flex-col items-start gap-1 p-3 rounded-xl border text-left transition-all ${
                      isSelected
                        ? 'bg-accent/5 border-accent text-accent'
                        : disableSelect
                          ? 'bg-background border-border opacity-50 cursor-not-allowed'
                          : 'bg-background border-border hover:border-accent/40 hover:bg-surface'
                    }`}
                  >
                    <div className="flex items-center gap-3 w-full">
                      <div className={`shrink-0 w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                        isSelected ? 'bg-accent border-accent text-white' : 'border-muted'
                      }`}>
                        {isSelected && <CheckCircle2 size={12} strokeWidth={3} />}
                      </div>
                      <div className={`text-sm font-medium truncate flex-1 ${isSelected ? 'text-accent' : 'text-text'}`}>
                        {agent.label}
                      </div>
                    </div>
                    {isSuggested && (
                      <div className="text-[10px] text-success font-medium ml-7 uppercase tracking-wider flex items-center gap-1">
                        <Sparkles size={10} /> Suggested
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        {/* Center Main Content */}
        <section className="flex-1 flex flex-col gap-8 min-w-0 w-full h-full overflow-y-auto px-4 pt-4 pb-12 custom-scrollbar">
          {/* Error Alert */}
          <AnimatePresence>
            {error && (
              <motion.div 
                key="error"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="p-4 rounded-2xl bg-error/10 border border-error/20 flex items-start gap-3 text-error"
              >
                <AlertCircle size={20} className="mt-0.5 shrink-0" />
                <div className="text-sm font-medium">{error}</div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Input Section */}
          <div className="relative group shrink-0 rounded-[1.5rem]">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-accent to-accent/50 rounded-[1.5rem] blur opacity-20 group-focus-within:opacity-40 transition duration-500 pointer-events-none"></div>
            <div className="relative bg-surface border border-border/50 rounded-[1.5rem] p-4 shadow-sm overflow-visible flex flex-col">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask anything, e.g. What is machine learning?"
                className="w-full bg-transparent resize-none outline-none text-base sm:text-lg min-h-[120px] placeholder:text-muted"
              />
              <div className="flex justify-between items-center mt-2 pt-2 border-t border-border/30 relative">
                <div className="flex items-center gap-4">
                  {/* Model Selector */}
                  <div className="relative">
                    <button
                      ref={modelBtnRef}
                      type="button"
                      onClick={() => {
                        setIsModelDropdownOpen((open) => {
                          if (!open && modelBtnRef.current) {
                            updateModelMenuPos();
                          }
                          return !open;
                        });
                      }}
                      className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-black/5 dark:hover:bg-white/5 transition-colors text-xs font-medium text-text border border-border/50 bg-background/50"
                    >
                      <span>{models.find(m => m.id === selectedModel)?.label}</span>
                      <ChevronDown size={14} className="text-muted" />
                    </button>

                    {typeof document !== 'undefined' &&
                      createPortal(
                        <>
                          {isModelDropdownOpen && (
                            <div
                              className="fixed inset-0 z-[9998]"
                              aria-hidden
                              onClick={() => setIsModelDropdownOpen(false)}
                            />
                          )}
                          <AnimatePresence>
                            {isModelDropdownOpen && modelMenuPos && (
                              <motion.div
                                key="model-menu"
                                role="listbox"
                                initial={{ opacity: 0, y: -6 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -6 }}
                                transition={{ duration: 0.15 }}
                                style={{
                                  position: 'fixed',
                                  top: modelMenuPos.top,
                                  left: modelMenuPos.left,
                                  width: modelMenuPos.width,
                                  zIndex: 9999,
                                }}
                                className="max-h-[min(280px,45vh)] overflow-y-auto custom-scrollbar bg-surface border border-border/50 rounded-xl shadow-lg py-1"
                              >
                                {models.map((model) => (
                                  <button
                                    key={model.id}
                                    type="button"
                                    role="option"
                                    onClick={() => {
                                      setSelectedModel(model.id);
                                      setIsModelDropdownOpen(false);
                                    }}
                                    className={`w-full text-left px-3 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/5 transition-colors flex items-center justify-between shrink-0 ${selectedModel === model.id ? 'text-accent font-medium' : 'text-text'}`}
                                  >
                                    {model.label}
                                    {selectedModel === model.id && <CheckCircle2 size={14} />}
                                  </button>
                                ))}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </>,
                        document.body
                      )}
                  </div>

                  <div className="text-xs text-muted hidden sm:flex items-center gap-1.5">
                    <Sparkles size={14} className="text-accent" />
                    MetaKnowledge will suggest the best agents
                  </div>
                </div>
                <button
                  onClick={handleSuggest}
                  disabled={loadingSuggest || !question.trim()}
                  className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white px-5 py-2 rounded-full font-medium text-sm transition-all shadow-sm hover:shadow active:scale-95 disabled:opacity-50 disabled:pointer-events-none"
                >
                  {loadingSuggest ? <Loader2 size={16} className="animate-spin" /> : <Bot size={16} />}
                  Get Suggestion
                </button>
              </div>
            </div>
          </div>

          <AnimatePresence mode="wait">
            {suggestData && (
              <motion.div 
                key="suggest-details"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Initial Answer */}
                <div className="bg-surface/50 border border-border/50 rounded-[1.5rem] p-6 shadow-sm">
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-muted mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-accent/80"></span>
                    Initial Answer
                  </h2>
                  <div className="prose dark:prose-invert max-w-none text-[0.95rem] leading-relaxed whitespace-pre-wrap">
                    {suggestData.initial_answer || 'No initial answer provided.'}
                  </div>
                </div>

                {/* Run Section */}
                <div className="bg-surface/50 border border-border/50 rounded-[1.5rem] p-6 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex flex-col gap-2">
                    <div className="text-sm font-medium">Ready to reflect?</div>
                    <div className="flex items-center gap-6">
                      <label className="flex items-center gap-2 cursor-pointer group text-sm font-medium">
                        <input 
                          type="checkbox" 
                          checked={updateLtm}
                          onChange={e => setUpdateLtm(e.target.checked)}
                          className="w-4 h-4 rounded border-border text-accent focus:ring-accent accent-accent"
                        />
                        <span className="group-hover:text-accent transition-colors text-muted">Update LTM</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer group text-sm font-medium">
                        <input 
                          type="checkbox" 
                          checked={updateMk}
                          onChange={e => setUpdateMk(e.target.checked)}
                          className="w-4 h-4 rounded border-border text-accent focus:ring-accent accent-accent"
                        />
                        <span className="group-hover:text-accent transition-colors text-muted">Update MK</span>
                      </label>
                    </div>
                  </div>
                  <button
                    onClick={handleRun}
                    disabled={loadingRun || selectedAgents.size === 0}
                    className="flex items-center justify-center gap-2 bg-text text-background hover:opacity-90 px-6 py-2.5 rounded-full font-semibold text-sm transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:pointer-events-none whitespace-nowrap"
                  >
                    {loadingRun ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} fill="currentColor" />}
                    Run Reflection
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence mode="wait">
            {runData && (
              <motion.div 
                key="result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="relative group mt-2"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-success/50 to-success/20 rounded-[1.5rem] blur opacity-20 transition duration-500"></div>
                <div className="relative bg-surface/80 backdrop-blur-sm border border-success/30 rounded-[1.5rem] p-6 shadow-lg">
                  <div className="flex flex-wrap items-center justify-between mb-4 gap-4">
                    <h2 className="text-sm font-bold uppercase tracking-wider text-success flex items-center gap-2">
                      <CheckCircle2 size={16} />
                      Final Reflected Answer
                    </h2>
                    <div className="flex gap-2 text-xs">
                      {runData.update_ltm && <span className="px-2 py-1 rounded bg-success/10 text-success font-medium">LTM Updated</span>}
                      {runData.update_mk && <span className="px-2 py-1 rounded bg-success/10 text-success font-medium">MK Updated</span>}
                    </div>
                  </div>
                  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed whitespace-pre-wrap">
                    {runData.final_answer}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Spacer to ensure padding and shadow/blur are respected in scroll container */}
          <div className="h-8 shrink-0 w-full pointer-events-none"></div>
        </section>

        {/* Right Sidebar: Feedbacks */}
        <AnimatePresence initial={false}>
          {isRightSidebarOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 320, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="shrink-0 flex flex-col gap-6 overflow-hidden hidden lg:flex h-full"
            >
              <div className="w-80 h-full">
                <div className="flex flex-col gap-6 h-full overflow-y-auto pr-2 pb-8 custom-scrollbar">
                  
                  <AnimatePresence mode="popLayout">
                    {(!runData || (runData.expert_feedbacks.length === 0 && runData.general_feedbacks.length === 0)) ? (
                      <motion.div
                        key="empty-feedback"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="bg-surface/30 border border-border/50 border-dashed rounded-[1.25rem] p-8 flex flex-col items-center justify-center text-center text-muted h-64"
                      >
                        <MessageSquareText size={32} className="mb-4 opacity-50" />
                        <p className="text-sm">Agent feedbacks will appear here after reflection.</p>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="feedbacks"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex flex-col gap-6"
                      >
                        {/* Expert Feedbacks */}
                        {runData.expert_feedbacks.length > 0 && (
                          <div className="bg-surface/40 border border-accent/20 rounded-[1.25rem] p-5 shadow-sm">
                            <h3 className="text-sm font-bold text-accent uppercase tracking-wider mb-4 flex items-center gap-2">
                              <Bot size={16} /> Expert Feedback
                            </h3>
                            <div className="flex flex-col gap-4">
                              {runData.expert_feedbacks.map((fb, idx) => (
                                <div key={idx} className="bg-background rounded-xl p-4 border border-border/50 shadow-sm">
                                  <div className="text-xs font-bold text-accent mb-2">
                                    {displayFeedbackAgentName(fb.agent)}
                                  </div>
                                  <div className="text-sm leading-relaxed text-text whitespace-pre-wrap">
                                    {fb.comment}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* General Feedbacks */}
                        {runData.general_feedbacks.length > 0 && (
                          <div className="bg-surface/40 border border-border/50 rounded-[1.25rem] p-5 shadow-sm">
                            <h3 className="text-sm font-bold text-muted uppercase tracking-wider mb-4 flex items-center gap-2">
                              <MessageSquareText size={16} /> General Feedback
                            </h3>
                            <div className="flex flex-col gap-4">
                              {runData.general_feedbacks.map((fb, idx) => (
                                <div key={idx} className="bg-background rounded-xl p-4 border border-border/50 shadow-sm">
                                  <div className="text-xs font-bold text-text mb-2 opacity-80">
                                    {displayFeedbackAgentName(fb.agent)}
                                  </div>
                                  <div className="text-sm leading-relaxed text-text whitespace-pre-wrap">
                                    {fb.comment}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>

                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>
          </>
        ) : (
          <section className="flex-1 flex flex-col gap-8 min-w-0 w-full max-w-3xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-surface/50 border border-border/50 rounded-[1.5rem] p-8 shadow-sm"
            >
              <h2 className="text-xl font-bold mb-6 flex items-center gap-3">
                <Settings2 className="text-accent" />
                Model Settings
              </h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium mb-2">API Key</label>
                  <input 
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-..."
                    className="w-full bg-background border border-border/50 rounded-xl px-4 py-2.5 outline-none focus:border-accent/50 transition-colors text-sm"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2 flex items-center justify-between">
                    <span>Temperature</span>
                    <span className="text-accent">{temperature.toFixed(1)}</span>
                  </label>
                  <input 
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-accent"
                  />
                  <div className="flex justify-between text-xs text-muted mt-2">
                    <span>Precise</span>
                    <span>Creative</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </section>
        )}

      </main>
    </div>
  );
}

export default App;
