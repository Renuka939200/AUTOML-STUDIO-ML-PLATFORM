import React from 'react';
import {
  LayoutDashboard,
  UploadCloud,
  SlidersHorizontal,
  Cpu,
  BarChart3,
  Sparkles,
  Database,
  BookOpen,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

export type NavTab =
  | 'overview'
  | 'upload'
  | 'configure'
  | 'train'
  | 'evaluate'
  | 'predict'
  | 'models'
  | 'report';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  hasDataset: boolean;
  hasModel: boolean;
  activeDatasetName?: string;
  activeModelName?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  hasDataset,
  hasModel,
  activeDatasetName,
  activeModelName,
}) => {
  const navItems = [
    {
      id: 'overview' as NavTab,
      label: 'Dashboard Overview',
      icon: LayoutDashboard,
      badge: null,
      enabled: true,
    },
    {
      id: 'upload' as NavTab,
      label: '1. Dataset Upload',
      icon: UploadCloud,
      badge: hasDataset ? 'Loaded' : null,
      enabled: true,
    },
    {
      id: 'configure' as NavTab,
      label: '2. Configuration',
      icon: SlidersHorizontal,
      badge: hasDataset ? 'Ready' : 'Upload first',
      enabled: hasDataset,
    },
    {
      id: 'train' as NavTab,
      label: '3. Model Training',
      icon: Cpu,
      badge: hasModel ? 'Trained' : (hasDataset ? 'Ready' : null),
      enabled: hasDataset,
    },
    {
      id: 'evaluate' as NavTab,
      label: '4. Evaluation & Plots',
      icon: BarChart3,
      badge: hasModel ? 'Results' : 'Train first',
      enabled: hasModel,
    },
    {
      id: 'predict' as NavTab,
      label: '5. Inference & Predict',
      icon: Sparkles,
      badge: hasModel ? 'Active' : 'Train first',
      enabled: hasModel,
    },
    {
      id: 'models' as NavTab,
      label: '6. Saved Models',
      icon: Database,
      badge: null,
      enabled: true,
    },
    {
      id: 'report' as NavTab,
      label: 'Documentation & Report',
      icon: BookOpen,
      badge: 'Academic',
      enabled: true,
    },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-screen shrink-0 text-slate-300">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-indigo-500/20">
          <Cpu className="w-5 h-5" />
        </div>
        <div>
          <h1 className="font-bold text-base text-white tracking-tight flex items-center gap-1.5">
            AutoML Studio
          </h1>
          <p className="text-xs text-slate-400 font-medium">Scikit-learn Platform</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        <div className="px-3 py-1.5 text-[11px] font-semibold tracking-wider text-slate-500 uppercase">
          Workflow Steps
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          const isEnabled = item.enabled;

          return (
            <button
              key={item.id}
              disabled={!isEnabled}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                  : isEnabled
                  ? 'text-slate-300 hover:bg-slate-800/80 hover:text-white'
                  : 'text-slate-600 cursor-not-allowed opacity-50'
              }`}
            >
              <div className="flex items-center space-x-3 truncate">
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span className="truncate">{item.label}</span>
              </div>
              {item.badge && (
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full font-medium shrink-0 ml-1.5 ${
                    isActive
                      ? 'bg-white/20 text-white'
                      : item.badge === 'Loaded' || item.badge === 'Trained' || item.badge === 'Active'
                      ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/60'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Pipeline Status Summary Card */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/50">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
          Session Status
        </div>
        <div className="space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Dataset:</span>
            {hasDataset ? (
              <span className="text-emerald-400 font-medium flex items-center gap-1 truncate max-w-[110px]" title={activeDatasetName}>
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">{activeDatasetName || 'Loaded'}</span>
              </span>
            ) : (
              <span className="text-amber-500 font-medium flex items-center gap-1">
                <AlertCircle className="w-3.5 h-3.5" /> None
              </span>
            )}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Active Model:</span>
            {hasModel ? (
              <span className="text-blue-400 font-medium flex items-center gap-1 truncate max-w-[110px]" title={activeModelName}>
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">{activeModelName || 'Fitted'}</span>
              </span>
            ) : (
              <span className="text-slate-500">Not trained</span>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
};
