import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { 
  ChevronRight, ChevronLeft, Plus, FolderOpen, Clock, Cpu, Zap, 
  X, Folder, MoreHorizontal, Trash2, ExternalLink, CheckCircle2, 
  AlertCircle, Check, Loader2, RefreshCw, Settings,
  Terminal, HardDrive, Upload, Search, AlertTriangle, HelpCircle,
  BookOpen, Rocket, MousePointer, Layout, Database, Play, Info,
  Copy, Edit, FolderPlus, Sun, Moon, ArrowUpDown, GripVertical, StopCircle, Power
} from 'lucide-react';
import api from './api';
import { AppProvider, useApp } from './context/AppContext';

// ==================== 配置数据 ====================
const SOFTWARE_CARDS = [
  { id: 'classification', name: '图像分类', subtitle: 'Image Classification', icon: '🎯', description: '基于TIMM的图像分类训练，支持数百种预训练模型', features: ['ResNet/EfficientNet/ViT', '迁移学习', '多GPU训练'], gradient: 'from-emerald-500 via-green-500 to-teal-500', metric: { name: 'Top-1 Acc', value: '97.5%' }, stats: { models: '400+', speed: '1.2ms' } },
  { id: 'detection', name: '目标检测', subtitle: 'Object Detection', icon: '🔍', description: 'YOLO系列目标检测，从训练到部署一站式解决', features: ['YOLOv5/v8/v11/v26', '实时检测', 'TensorRT加速'], gradient: 'from-blue-500 via-cyan-500 to-sky-500', metric: { name: 'mAP@50', value: '92.3%' }, stats: { models: '20+', speed: '2.1ms' } },
  { id: 'segmentation', name: '语义分割', subtitle: 'Semantic Segmentation', icon: '🎨', description: 'SegFormer语义分割，像素级精确分类', features: ['SegFormer B0-B5', 'MMSegmentation', '多尺度融合'], gradient: 'from-purple-500 via-violet-500 to-fuchsia-500', metric: { name: 'mIoU', value: '84.2%' }, stats: { models: '30+', speed: '3.5ms' } },
  { id: 'anomaly', name: '异常检测', subtitle: 'Anomaly Detection', icon: '🔬', description: 'PatchCore无监督异常检测，仅需良品数据', features: ['无监督学习', '热力图可视化', '高精度定位'], gradient: 'from-orange-500 via-amber-500 to-yellow-500', metric: { name: 'AUROC', value: '98.7%' }, stats: { models: '15+', speed: '5.2ms' } },
  { id: 'ocr', name: 'OCR识别', subtitle: 'Text Recognition', icon: '📝', description: 'PaddleOCR文字检测识别，支持多语言', features: ['文字检测', '文字识别', '版面分析'], gradient: 'from-rose-500 via-pink-500 to-red-500', metric: { name: 'Accuracy', value: '99.1%' }, stats: { models: '20+', speed: '8.3ms' } },
  { id: 'sevseg', name: '工业缺陷检测', subtitle: 'Defect Scoring', icon: '🏭', description: 'SevSeg-YOLO缺陷检测+严重程度评分+零标注近似分割', features: ['检测+评分+分割', 'YOLO26-Score', 'MaskGenerator'], gradient: 'from-gray-500 via-blue-600 to-purple-500', metric: { name: 'mAP@50', value: '62.3%' }, stats: { models: '5', speed: '6.4ms' } },
  { id: 'developing', name: '待开发', subtitle: 'Coming Soon', icon: '🚧', description: '更多功能正在开发中，敬请期待...', features: ['模型转换', '边缘部署', '更多算法'], gradient: 'from-gray-500 via-gray-600 to-gray-700', metric: { name: 'Status', value: 'DEV' }, stats: { models: '...', speed: '...' }, disabled: true }
];

// ==================== 安全的存储工具 [Bug-6修复] ====================
const safeStorage = {
  getItem: (key, defaultValue = null) => {
    try {
      const value = localStorage.getItem(key);
      return value !== null ? value : defaultValue;
    } catch (e) {
      // Safari隐私模式或localStorage不可用
      console.warn('[Storage] localStorage不可用:', e.message);
      return defaultValue;
    }
  },
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value);
      return true;
    } catch (e) {
      console.warn('[Storage] 无法写入localStorage:', e.message);
      return false;
    }
  },
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (e) {
      return false;
    }
  }
};

// ==================== 主题Hook [Bug-6修复] ====================
const useTheme = () => {
  const [isDark, setIsDark] = useState(() => {
    // [Bug-6修复] 使用安全的存储访问
    const saved = safeStorage.getItem('dlhub_theme');
    return saved ? saved === 'dark' : true;  // 默认深色主题
  });

  useEffect(() => {
    // [Bug-6修复] 使用安全的存储写入
    safeStorage.setItem('dlhub_theme', isDark ? 'dark' : 'light');
    document.documentElement.classList.toggle('light-theme', !isDark);
  }, [isDark]);

  return { isDark, toggleTheme: () => setIsDark(!isDark) };
};

// ==================== 全局通知组件 ====================
const GlobalNotification = () => {
  const { state, actions } = useApp();
  const { notification } = state;
  if (!notification) return null;
  const typeConfig = {
    success: { bg: 'bg-green-500/20', border: 'border-green-500/30', text: 'text-green-300', icon: CheckCircle2 },
    error: { bg: 'bg-red-500/20', border: 'border-red-500/30', text: 'text-red-300', icon: AlertCircle },
    warning: { bg: 'bg-yellow-500/20', border: 'border-yellow-500/30', text: 'text-yellow-300', icon: AlertTriangle },
    info: { bg: 'bg-blue-500/20', border: 'border-blue-500/30', text: 'text-blue-300', icon: Info },
  };
  const config = typeConfig[notification.type] || typeConfig.info;
  const Icon = config.icon;
  return (
    <div className="fixed top-4 right-4 z-[100] animate-in slide-in-from-top-4">
      <div className={`${config.bg} ${config.border} border ${config.text} rounded-xl p-4 flex items-center gap-3 shadow-lg max-w-md`}>
        <Icon size={20} /><span className="flex-1">{notification.message}</span>
        <button onClick={actions.hideNotification} className="p-1 hover:bg-white/10 rounded-lg"><X size={16} /></button>
      </div>
    </div>
  );
};

// ==================== 骨架屏组件（优化-1） ====================
const Skeleton = ({ className = '' }) => (
  <div className={`animate-pulse bg-white/10 rounded ${className}`} />
);

const TaskCardSkeleton = () => (
  <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5">
    <div className="flex items-start gap-4 mb-4">
      <Skeleton className="w-12 h-12 rounded-xl" />
      <div className="flex-1">
        <Skeleton className="h-5 w-32 mb-2" />
        <Skeleton className="h-4 w-48" />
      </div>
    </div>
    <Skeleton className="h-4 w-24 mb-4" />
    <div className="pt-4 border-t border-white/5 flex justify-between">
      <Skeleton className="h-4 w-20" />
      <Skeleton className="h-6 w-16" />
    </div>
  </div>
);

// ==================== [UI优化] 系统资源监控组件 ====================
const SystemResourceMonitor = ({ collapsed = false }) => {
  const [resources, setResources] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResources = async () => {
      try {
        const data = await api.getSystemResources();
        setResources(data);
      } catch (err) {
        console.error('获取系统资源失败:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchResources();
    const interval = setInterval(fetchResources, 3000); // 每3秒更新
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-4 px-4 py-2 bg-white/5 rounded-xl">
        <Skeleton className="w-20 h-4" />
        <Skeleton className="w-20 h-4" />
        <Skeleton className="w-20 h-4" />
      </div>
    );
  }

  if (!resources) return null;

  const getColorByPercent = (percent) => {
    if (percent >= 90) return 'text-red-400';
    if (percent >= 70) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getBarColor = (percent) => {
    if (percent >= 90) return 'bg-red-500';
    if (percent >= 70) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  if (collapsed) {
    // 紧凑模式
    return (
      <div className="flex items-center gap-3 px-3 py-1.5 bg-white/5 rounded-lg text-xs">
        <div className="flex items-center gap-1.5">
          <Cpu size={12} className="text-gray-500" />
          <span className={getColorByPercent(resources.cpu.percent)}>{resources.cpu.percent.toFixed(0)}%</span>
        </div>
        <div className="flex items-center gap-1.5">
          <HardDrive size={12} className="text-gray-500" />
          <span className={getColorByPercent(resources.memory.percent)}>{resources.memory.percent.toFixed(0)}%</span>
        </div>
        {resources.gpu && resources.gpu.length > 0 && (
          <div className="flex items-center gap-1.5">
            <Zap size={12} className="text-gray-500" />
            <span className={getColorByPercent(resources.gpu[0].utilization)}>{resources.gpu[0].utilization}%</span>
            {resources.gpu[0].temperature && (
              <span className="text-gray-500">{resources.gpu[0].temperature}°C</span>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white/[0.03] border border-white/10 rounded-xl p-4">
      <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
        <Settings size={14} className="animate-spin-slow" />
        系统资源
      </h4>
      
      <div className="space-y-3">
        {/* CPU */}
        <div className="flex items-center gap-3">
          <Cpu size={16} className="text-blue-400" />
          <div className="flex-1">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">CPU</span>
              <span className={getColorByPercent(resources.cpu.percent)}>{resources.cpu.percent.toFixed(1)}%</span>
            </div>
            <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div 
                className={`h-full ${getBarColor(resources.cpu.percent)} transition-all duration-300`}
                style={{ width: `${resources.cpu.percent}%` }}
              />
            </div>
          </div>
        </div>

        {/* 内存 */}
        <div className="flex items-center gap-3">
          <HardDrive size={16} className="text-purple-400" />
          <div className="flex-1">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">内存</span>
              <span className={getColorByPercent(resources.memory.percent)}>
                {resources.memory.used.toFixed(1)}/{resources.memory.total.toFixed(1)} GB
              </span>
            </div>
            <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div 
                className={`h-full ${getBarColor(resources.memory.percent)} transition-all duration-300`}
                style={{ width: `${resources.memory.percent}%` }}
              />
            </div>
          </div>
        </div>

        {/* GPU */}
        {resources.gpu && resources.gpu.map((gpu, idx) => {
          const gpuMemPercent = (gpu.memory_used / gpu.memory_total) * 100;
          return (
            <div key={idx} className="flex items-center gap-3">
              <Zap size={16} className="text-green-400" />
              <div className="flex-1">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-400 truncate max-w-[100px]" title={gpu.name}>
                    GPU{gpu.index}
                  </span>
                  <span className="flex items-center gap-2">
                    <span className={getColorByPercent(gpu.utilization)}>{gpu.utilization}%</span>
                    {gpu.temperature && (
                      <span className="text-gray-500">{gpu.temperature}°C</span>
                    )}
                  </span>
                </div>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${getBarColor(gpu.utilization)} transition-all duration-300`}
                    style={{ width: `${gpu.utilization}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-0.5">
                  显存: {gpu.memory_used}/{gpu.memory_total} MB ({gpuMemPercent.toFixed(0)}%)
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ==================== [UI优化] 任务状态徽章 ====================
const TaskStatusBadge = ({ status, isRunning, size = 'md' }) => {
  const configs = {
    running: { 
      bg: 'bg-green-500/20', 
      border: 'border-green-500/50', 
      text: 'text-green-400',
      dot: 'bg-green-500',
      label: '运行中',
      animate: true
    },
    completed: { 
      bg: 'bg-blue-500/20', 
      border: 'border-blue-500/50', 
      text: 'text-blue-400',
      dot: 'bg-blue-500',
      label: '已完成',
      animate: false
    },
    error: { 
      bg: 'bg-red-500/20', 
      border: 'border-red-500/50', 
      text: 'text-red-400',
      dot: 'bg-red-500',
      label: '出错',
      animate: false
    },
    idle: { 
      bg: 'bg-gray-500/20', 
      border: 'border-gray-500/50', 
      text: 'text-gray-400',
      dot: 'bg-gray-500',
      label: '未开始',
      animate: false
    },
  };

  // 如果isRunning为true，强制显示运行中状态
  const effectiveStatus = isRunning ? 'running' : (status || 'idle');
  const config = configs[effectiveStatus] || configs.idle;
  
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs',
    lg: 'px-3 py-1.5 text-sm'
  };

  return (
    <span className={`
      inline-flex items-center gap-1.5 rounded-full border
      ${config.bg} ${config.border} ${config.text} ${sizeClasses[size]}
    `}>
      <span className={`
        w-1.5 h-1.5 rounded-full ${config.dot}
        ${config.animate ? 'animate-pulse' : ''}
      `} />
      {config.label}
    </span>
  );
};

// ==================== [UI优化] 快捷键Hook ====================
const useKeyboardShortcuts = (shortcuts) => {
  useEffect(() => {
    const handleKeyDown = (e) => {
      // 忽略输入框内的快捷键
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      for (const shortcut of shortcuts) {
        const { key, ctrl, shift, alt, action } = shortcut;
        
        if (
          e.key.toLowerCase() === key.toLowerCase() &&
          !!e.ctrlKey === !!ctrl &&
          !!e.shiftKey === !!shift &&
          !!e.altKey === !!alt
        ) {
          e.preventDefault();
          action();
          break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
};

// ==================== [UI优化] 快捷键提示组件 ====================
const KeyboardShortcutHint = ({ show, onClose }) => {
  if (!show) return null;

  const shortcuts = [
    { keys: ['Ctrl', 'N'], desc: '新建任务' },
    { keys: ['Ctrl', 'I'], desc: '导入任务' },
    { keys: ['Ctrl', 'F'], desc: '搜索任务' },
    { keys: ['F5'], desc: '刷新列表' },
    { keys: ['Esc'], desc: '关闭弹窗/返回' },
    { keys: ['?'], desc: '显示快捷键' },
  ];

  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-[#1a1f2e] rounded-2xl w-full max-w-md border border-white/10 p-6">
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          ⌨️ 快捷键
        </h3>
        <div className="space-y-2">
          {shortcuts.map((s, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
              <span className="text-gray-400">{s.desc}</span>
              <div className="flex gap-1">
                {s.keys.map((k, j) => (
                  <kbd key={j} className="px-2 py-1 bg-white/10 rounded text-xs text-white font-mono">
                    {k}
                  </kbd>
                ))}
              </div>
            </div>
          ))}
        </div>
        <button 
          onClick={onClose}
          className="mt-4 w-full py-2 bg-white/10 hover:bg-white/20 rounded-xl text-white text-sm"
        >
          关闭
        </button>
      </div>
    </div>
  );
};

// ==================== 帮助引导弹窗 ====================
const HelpGuideModal = ({ isOpen, onClose }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const { actions } = useApp();
  const steps = [
    { title: '欢迎使用 DL-Hub', icon: <Rocket className="w-16 h-16 text-blue-400" />, content: (<div className="space-y-4"><p className="text-gray-300">DL-Hub 是一个统一的深度学习任务管理平台，支持多种计算机视觉任务：</p><div className="grid grid-cols-2 gap-3">{['图像分类', '目标检测', '语义分割', '异常检测', 'OCR识别', '工业缺陷检测'].map((item, i) => (<div key={i} className="flex items-center gap-2 text-gray-400"><CheckCircle2 size={16} className="text-green-400" /><span>{item}</span></div>))}</div></div>) },
    { title: '选择任务类型', icon: <Layout className="w-16 h-16 text-purple-400" />, content: (<div className="space-y-4"><p className="text-gray-300">在主页选择您需要的任务类型卡片，进入对应的任务管理页面。</p></div>) },
    { title: '创建新任务', icon: <Plus className="w-16 h-16 text-green-400" />, content: (<div className="space-y-4"><p className="text-gray-300">创建任务需要：任务名称、工作目录、Conda环境。</p></div>) },
    { title: '管理和训练', icon: <Play className="w-16 h-16 text-orange-400" />, content: (<div className="space-y-4"><p className="text-gray-300">点击任务卡片启动训练界面，支持导入、编辑、复制、删除任务。</p></div>) },
    { title: '准备开始', icon: <BookOpen className="w-16 h-16 text-cyan-400" />, content: (<div className="space-y-4"><p className="text-gray-300">确保已安装Conda并创建好训练环境。</p><div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl"><p className="text-blue-300 text-sm">💡 点击右上角 <HelpCircle size={14} className="inline" /> 可随时查看帮助</p></div></div>) }
  ];
  if (!isOpen) return null;
  const handleComplete = () => { actions.markVisited(); onClose(); };
  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-[#1a1f2e] rounded-2xl w-full max-w-2xl border border-white/10 overflow-hidden">
        <div className="flex gap-1 p-4 bg-white/5">{steps.map((_, i) => (<div key={i} className={`flex-1 h-1 rounded-full transition-colors ${i <= currentStep ? 'bg-blue-500' : 'bg-white/10'}`} />))}</div>
        <div className="p-8"><div className="flex flex-col items-center text-center mb-8">{steps[currentStep].icon}<h2 className="text-2xl font-bold text-white mt-4">{steps[currentStep].title}</h2></div><div className="min-h-[200px]">{steps[currentStep].content}</div></div>
        <div className="p-6 border-t border-white/10 flex justify-between">
          <button onClick={onClose} className="px-5 py-2.5 text-gray-400 hover:text-white">跳过引导</button>
          <div className="flex gap-3">
            {currentStep > 0 && <button onClick={() => setCurrentStep(currentStep - 1)} className="px-5 py-2.5 bg-white/10 hover:bg-white/20 rounded-xl text-white">上一步</button>}
            {currentStep < steps.length - 1 ? <button onClick={() => setCurrentStep(currentStep + 1)} className="px-6 py-2.5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl text-white font-medium">下一步</button> : <button onClick={handleComplete} className="px-6 py-2.5 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl text-white font-medium">开始使用</button>}
          </div>
        </div>
      </div>
    </div>
  );
};

// ==================== 粒子背景 ====================
const ParticleBackground = () => {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current; const ctx = canvas.getContext('2d'); let animationFrameId; let particles = [];
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }; resize(); window.addEventListener('resize', resize);
    for (let i = 0; i < 50; i++) { particles.push({ x: Math.random() * canvas.width, y: Math.random() * canvas.height, vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5, size: Math.random() * 2 + 1, opacity: Math.random() * 0.5 + 0.2 }); }
    const animate = () => { ctx.clearRect(0, 0, canvas.width, canvas.height); particles.forEach((p, i) => { p.x += p.vx; p.y += p.vy; if (p.x < 0 || p.x > canvas.width) p.vx *= -1; if (p.y < 0 || p.y > canvas.height) p.vy *= -1; ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fillStyle = `rgba(99, 102, 241, ${p.opacity})`; ctx.fill(); particles.slice(i + 1).forEach(p2 => { const dx = p.x - p2.x, dy = p.y - p2.y, dist = Math.sqrt(dx * dx + dy * dy); if (dist < 150) { ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p2.x, p2.y); ctx.strokeStyle = `rgba(99, 102, 241, ${0.1 * (1 - dist / 150)})`; ctx.stroke(); } }); }); animationFrameId = requestAnimationFrame(animate); }; animate();
    return () => { window.removeEventListener('resize', resize); cancelAnimationFrame(animationFrameId); };
  }, []);
  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none" style={{ zIndex: 0 }} />;
};

// ==================== 确认弹窗 ====================
const ConfirmModal = ({ isOpen, onClose, onConfirm, title, message, confirmText = "确定", cancelText = "取消", danger = false }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-[#1a1f2e] rounded-2xl w-full max-w-md border border-white/10 overflow-hidden">
        <div className="p-6"><div className="flex items-start gap-4"><div className={`w-12 h-12 rounded-full flex items-center justify-center ${danger ? 'bg-red-500/20' : 'bg-blue-500/20'}`}>{danger ? <AlertTriangle className="w-6 h-6 text-red-400" /> : <AlertCircle className="w-6 h-6 text-blue-400" />}</div><div className="flex-1"><h3 className="text-lg font-semibold text-white mb-2">{title}</h3><p className="text-gray-400 text-sm">{message}</p></div></div></div>
        <div className="p-4 border-t border-white/10 flex justify-end gap-3 bg-white/5"><button onClick={onClose} className="px-5 py-2.5 text-gray-400 hover:text-white">{cancelText}</button><button onClick={() => { onConfirm(); onClose(); }} className={`px-5 py-2.5 rounded-xl font-medium ${danger ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-blue-500 hover:bg-blue-600 text-white'}`}>{confirmText}</button></div>
      </div>
    </div>
  );
};

// ==================== 目录浏览器（问题-4、问题-8：新建文件夹、磁盘空间） ====================
const DirectoryBrowser = ({ isOpen, onClose, onSelect, title = "选择目录", selectButtonText = "选择此目录" }) => {
  const [currentPath, setCurrentPath] = useState(''); const [inputPath, setInputPath] = useState(''); const [items, setItems] = useState([]); const [loading, setLoading] = useState(false); const [history, setHistory] = useState([]); const [error, setError] = useState(''); const [showInput, setShowInput] = useState(false);
  const [newFolderName, setNewFolderName] = useState(''); const [showNewFolder, setShowNewFolder] = useState(false); const [creating, setCreating] = useState(false);
  const [diskSpace, setDiskSpace] = useState(null);

  useEffect(() => { if (isOpen) { setHistory([]); setInputPath(''); setShowInput(false); setShowNewFolder(false); browsePath(''); } }, [isOpen]);

  const browsePath = async (path) => { 
    setLoading(true); setError(''); 
    try { 
      const data = await api.browseDirectory(path); 
      setItems(data.items.filter(item => item.type === 'directory' || item.type === 'drive')); 
      setCurrentPath(data.current || ''); 
      setInputPath(data.current || ''); 
      // 问题-8：获取磁盘空间
      if (data.current) {
        try {
          const space = await api.getDiskSpace(data.current);
          setDiskSpace(space);
        } catch { setDiskSpace(null); }
      }
    } catch (err) { setError(err.message); } 
    finally { setLoading(false); } 
  };

  const handleNavigate = (path) => { setHistory([...history, currentPath]); browsePath(path); };
  const handleBack = () => { if (history.length > 0) { const prev = history[history.length - 1]; setHistory(history.slice(0, -1)); browsePath(prev); } };
  const handleInputSubmit = (e) => { e.preventDefault(); if (inputPath.trim()) { setHistory([...history, currentPath]); browsePath(inputPath.trim()); } };
  const handleSelect = () => { if (currentPath) { onSelect(currentPath); onClose(); } };

  // 问题-4：新建文件夹
  const handleCreateFolder = async () => {
    if (!newFolderName.trim() || !currentPath) return;
    setCreating(true);
    try {
      const result = await api.createDirectory(currentPath, newFolderName.trim());
      if (result.success) {
        setShowNewFolder(false);
        setNewFolderName('');
        browsePath(currentPath); // 刷新列表
      }
    } catch (err) { setError(err.message); }
    finally { setCreating(false); }
  };

  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-[#1a1f2e] rounded-2xl w-full max-w-2xl border border-white/10 overflow-hidden">
        <div className="p-6 border-b border-white/10 bg-gradient-to-r from-blue-500/10 to-purple-500/10"><div className="flex items-center justify-between"><h2 className="text-xl font-bold text-white flex items-center gap-3"><FolderOpen className="w-6 h-6 text-blue-400" />{title}</h2><button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10"><X size={20} className="text-gray-400" /></button></div></div>
        <div className="px-6 py-3 bg-white/5 border-b border-white/10 flex items-center gap-2">
          <button onClick={handleBack} disabled={history.length === 0} className="p-2 rounded-lg hover:bg-white/10 disabled:opacity-30"><ChevronLeft size={20} className="text-gray-400" /></button>
          {showInput ? (<form onSubmit={handleInputSubmit} className="flex-1 flex gap-2"><input type="text" value={inputPath} onChange={(e) => setInputPath(e.target.value)} placeholder="输入路径，按回车跳转" className="flex-1 px-4 py-2.5 bg-black/30 border border-white/20 rounded-lg text-white font-mono text-sm focus:outline-none focus:border-blue-500" autoFocus /><button type="submit" className="px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 rounded-lg text-blue-400 text-sm">跳转</button><button type="button" onClick={() => setShowInput(false)} className="px-3 py-2 hover:bg-white/10 rounded-lg text-gray-400"><X size={16} /></button></form>) : (<><div className="flex-1 px-4 py-2.5 bg-black/30 rounded-lg cursor-pointer hover:bg-black/40" onClick={() => setShowInput(true)} title="点击手动输入路径"><span className="text-white font-mono text-sm truncate block">{currentPath || '选择目录...'}</span></div><button onClick={() => setShowInput(true)} className="p-2 hover:bg-white/10 rounded-lg text-gray-400" title="手动输入路径"><Settings size={18} /></button></>)}
          {/* 问题-4：新建文件夹按钮 */}
          <button onClick={() => setShowNewFolder(true)} disabled={!currentPath} className="p-2 hover:bg-white/10 rounded-lg text-gray-400 disabled:opacity-30" title="新建文件夹"><FolderPlus size={18} /></button>
        </div>
        {/* 问题-8：磁盘空间显示 */}
        {diskSpace && (
          <div className="px-6 py-2 bg-white/5 border-b border-white/10 flex items-center justify-between text-sm">
            <span className="text-gray-400">磁盘空间</span>
            <span className={`${diskSpace.percent_used > 90 ? 'text-red-400' : diskSpace.percent_used > 70 ? 'text-yellow-400' : 'text-green-400'}`}>
              可用 {diskSpace.free_formatted} / 共 {diskSpace.total_formatted} ({diskSpace.percent_used}% 已用)
            </span>
          </div>
        )}
        {/* 新建文件夹输入框 */}
        {showNewFolder && (
          <div className="px-6 py-3 bg-blue-500/10 border-b border-white/10 flex items-center gap-2">
            <FolderPlus size={18} className="text-blue-400" />
            <input type="text" value={newFolderName} onChange={(e) => setNewFolderName(e.target.value)} placeholder="新文件夹名称" className="flex-1 px-3 py-2 bg-black/30 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500" autoFocus onKeyDown={(e) => e.key === 'Enter' && handleCreateFolder()} />
            <button onClick={handleCreateFolder} disabled={creating || !newFolderName.trim()} className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm disabled:opacity-50">{creating ? '创建中...' : '创建'}</button>
            <button onClick={() => { setShowNewFolder(false); setNewFolderName(''); }} className="p-2 hover:bg-white/10 rounded-lg text-gray-400"><X size={16} /></button>
          </div>
        )}
        {error && <div className="mx-6 mt-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-300 text-sm">{error}</div>}
        <div className="h-72 overflow-y-auto p-4">
          {loading ? <div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 text-blue-400 animate-spin" /></div> : (<div className="space-y-1">{items.map((item) => (<div key={item.path} onClick={() => handleNavigate(item.path)} className="flex items-center gap-3 p-3 rounded-lg hover:bg-white/10 cursor-pointer group">{item.type === 'drive' ? <HardDrive size={20} className="text-blue-400" /> : <Folder size={20} className="text-yellow-400" />}<span className="text-white flex-1 truncate">{item.name}</span><ChevronRight size={16} className="text-gray-500 opacity-0 group-hover:opacity-100" /></div>))}{items.length === 0 && !loading && <div className="text-center py-8 text-gray-500">此目录为空</div>}</div>)}
        </div>
        <div className="p-6 border-t border-white/10 flex justify-end gap-3 bg-white/5"><button onClick={onClose} className="px-5 py-2.5 text-gray-400 hover:text-white">取消</button><button onClick={handleSelect} disabled={!currentPath} className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl text-white font-semibold disabled:opacity-50 hover:shadow-lg"><Check size={20} />{selectButtonText}</button></div>
      </div>
    </div>
  );
};

// ==================== Conda环境选择器 ====================
const CondaEnvSelector = ({ value, onChange }) => {
  const [condaEnvs, setCondaEnvs] = useState([]);
  const [loadingEnvs, setLoadingEnvs] = useState(false);
  const [envError, setEnvError] = useState('');
  const [showDirBrowser, setShowDirBrowser] = useState(false);

  useEffect(() => { fetchCondaEnvs(); }, []);

  const fetchCondaEnvs = async () => {
    setLoadingEnvs(true); 
    setEnvError('');
    try {
      const result = await api.getCondaEnvs();
      if (result.error && (!result.envs || result.envs.length === 0)) { 
        setEnvError(result.error); 
        setCondaEnvs([]); 
      } else { 
        setCondaEnvs(result.envs || []); 
        const active = result.envs?.find(e => e.is_active); 
        if (active && !value) onChange(active.name); 
      }
    } catch (err) { setEnvError(err.message); setCondaEnvs([]); }
    finally { setLoadingEnvs(false); }
  };

  const handleScanDirectory = async (path) => {
    try {
      const result = await api.scanCondaDirectory(path);
      if (result.error) {
        setEnvError(result.error);
      } else if (result.envs && result.envs.length > 0) {
        const existingNames = new Set(condaEnvs.map(e => e.name));
        const newEnvs = result.envs.filter(e => !existingNames.has(e.name));
        setCondaEnvs([...condaEnvs, ...newEnvs]);
        setEnvError('');
        if (!value && result.envs.length > 0) onChange(result.envs[0].name);
      }
    } catch (err) { setEnvError(err.message); }
    finally { setShowDirBrowser(false); }
  };

  return (
    <>
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Conda 环境 <span className="text-red-400">*</span></label>
        {loadingEnvs ? (
          <div className="flex items-center gap-3 px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-gray-400"><Loader2 size={18} className="animate-spin" />加载环境列表...</div>
        ) : envError && condaEnvs.length === 0 ? (
          <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl">
            <div className="flex items-center gap-2 text-yellow-400 mb-2"><AlertTriangle size={18} /><span className="font-medium">自动获取Conda环境失败</span></div>
            <p className="text-sm text-yellow-200/70 mb-3">{envError}</p>
            <div className="flex gap-2">
              <button onClick={fetchCondaEnvs} className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-1"><RefreshCw size={14} />重新检测</button>
              <span className="text-gray-600">|</span>
              <button onClick={() => setShowDirBrowser(true)} className="text-sm text-green-400 hover:text-green-300 flex items-center gap-1"><FolderOpen size={14} />手动选择Anaconda目录</button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex gap-3">
              <select value={value} onChange={(e) => onChange(e.target.value)} className="flex-1 px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-blue-500 appearance-none cursor-pointer" style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%239CA3AF'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center', backgroundSize: '20px' }}>
                <option value="" className="bg-[#1a1f2e]">请选择环境</option>
                {condaEnvs.map((env) => (<option key={env.name} value={env.name} className="bg-[#1a1f2e]">{env.name} {env.python_version ? `(Python ${env.python_version})` : ''} {env.is_active ? '✓' : ''}</option>))}
              </select>
              <button onClick={() => setShowDirBrowser(true)} className="px-4 py-3.5 bg-white/10 hover:bg-white/20 border border-white/10 rounded-xl text-white flex items-center gap-2" title="手动添加环境"><Plus size={18} /></button>
            </div>
            <p className="text-xs text-gray-500">已找到 {condaEnvs.length} 个环境 · <button onClick={() => setShowDirBrowser(true)} className="text-blue-400 hover:text-blue-300 ml-1">手动添加更多环境</button></p>
          </div>
        )}
      </div>
      <DirectoryBrowser isOpen={showDirBrowser} onClose={() => setShowDirBrowser(false)} onSelect={handleScanDirectory} title="选择Anaconda/Miniconda目录" selectButtonText="扫描此目录" />
    </>
  );
};

// ==================== 新建任务弹窗 ====================
const CreateTaskModal = ({ isOpen, onClose, software, onTaskCreated }) => {
  const [taskName, setTaskName] = useState(''); const [description, setDescription] = useState(''); const [workDir, setWorkDir] = useState(''); const [condaEnv, setCondaEnv] = useState('');
  const [isCreating, setIsCreating] = useState(false); const [error, setError] = useState(''); const [showDirBrowser, setShowDirBrowser] = useState(false);
  const [diskSpace, setDiskSpace] = useState(null);
  const { actions } = useApp(); const card = SOFTWARE_CARDS.find(c => c.id === software);
  useEffect(() => { if (isOpen) { setTaskName(''); setDescription(''); setWorkDir(''); setCondaEnv(''); setError(''); setDiskSpace(null); } }, [isOpen]);

  // 问题-8：选择目录后获取磁盘空间
  const handleSelectWorkDir = async (path) => {
    setWorkDir(path);
    try {
      const space = await api.getDiskSpace(path);
      setDiskSpace(space);
    } catch { setDiskSpace(null); }
  };

  if (!isOpen || !card) return null;
  const handleCreate = async () => {
    if (!taskName.trim()) { setError('请输入任务名称'); return; }
    if (!workDir.trim()) { setError('请选择工作目录'); return; }
    if (!condaEnv) { setError('请选择Conda环境'); return; }
    setIsCreating(true); setError('');
    try {
      const result = await api.createTask(software, taskName.trim(), workDir.trim(), condaEnv, description.trim());
      if (result.success) { actions.showSuccess(`任务"${taskName}"创建成功！`); onTaskCreated(result.task); onClose(); }
    } catch (err) { setError(err.message); }
    finally { setIsCreating(false); }
  };
  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
        <div className="relative bg-[#1a1f2e] rounded-2xl w-full max-w-xl border border-white/10 overflow-hidden max-h-[90vh] overflow-y-auto">
          <div className={`p-6 bg-gradient-to-r ${card.gradient} bg-opacity-20`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4"><span className="text-5xl">{card.icon}</span><div><h3 className="text-2xl font-bold text-white">新建{card.name}任务</h3><p className="text-base text-white/70">{card.subtitle}</p></div></div>
              <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10"><X size={24} className="text-white/70" /></button>
            </div>
          </div>
          <div className="p-6 space-y-5">
            {error && <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-center gap-3"><AlertCircle size={20} />{error}</div>}
            <div><label className="block text-sm font-medium text-gray-300 mb-2">任务名称 <span className="text-red-400">*</span></label><input type="text" value={taskName} onChange={(e) => setTaskName(e.target.value)} placeholder="例如：产品缺陷检测" className="w-full px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500" autoFocus /></div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">工作目录 <span className="text-red-400">*</span></label>
              <div className="flex gap-3"><div className="flex-1 px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl overflow-hidden">{workDir ? <span className="text-white font-mono truncate block">{workDir}</span> : <span className="text-gray-500">未选择目录</span>}</div><button onClick={() => setShowDirBrowser(true)} className="px-5 py-3.5 bg-white/10 hover:bg-white/20 border border-white/10 rounded-xl text-white flex items-center gap-2"><FolderOpen size={20} />浏览</button></div>
              {/* 问题-8：显示磁盘空间 */}
              {diskSpace && (
                <p className={`text-xs mt-2 ${diskSpace.percent_used > 90 ? 'text-red-400' : diskSpace.percent_used > 70 ? 'text-yellow-400' : 'text-gray-500'}`}>
                  磁盘剩余空间: {diskSpace.free_formatted} ({diskSpace.percent_used}% 已用)
                </p>
              )}
            </div>
            <CondaEnvSelector value={condaEnv} onChange={setCondaEnv} />
            <div><label className="block text-sm font-medium text-gray-300 mb-2">任务描述</label><textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="可选，描述任务的目标和用途" rows={3} className="w-full px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none" /></div>
          </div>
          <div className="p-6 border-t border-white/10 flex justify-end gap-3">
            <button onClick={onClose} className="px-6 py-3 text-gray-400 hover:text-white">取消</button>
            <button onClick={handleCreate} disabled={isCreating || !taskName.trim() || !workDir || !condaEnv} className={`flex items-center gap-2 px-8 py-3 bg-gradient-to-r ${card.gradient} rounded-xl text-white font-semibold disabled:opacity-50 hover:shadow-lg`}>{isCreating ? <Loader2 size={20} className="animate-spin" /> : <Plus size={20} />}{isCreating ? '创建中...' : '创建任务'}</button>
          </div>
        </div>
      </div>
      <DirectoryBrowser isOpen={showDirBrowser} onClose={() => setShowDirBrowser(false)} onSelect={handleSelectWorkDir} title="选择工作目录" selectButtonText="选择此目录" />
    </>
  );
};

// ==================== 导入任务弹窗（支持替换已存在的任务） ====================
const ImportTaskModal = ({ isOpen, onClose, software, onTaskImported }) => {
  const [taskDir, setTaskDir] = useState(''); 
  const [isImporting, setIsImporting] = useState(false); 
  const [error, setError] = useState(''); 
  const [showDirBrowser, setShowDirBrowser] = useState(false);
  const [needCondaEnv, setNeedCondaEnv] = useState(false); 
  const [condaEnv, setCondaEnv] = useState('');
  const [showReplaceConfirm, setShowReplaceConfirm] = useState(false);  // 替换确认状态
  
  const { actions } = useApp(); 
  const card = SOFTWARE_CARDS.find(c => c.id === software);
  
  useEffect(() => { 
    if (isOpen) { 
      setTaskDir(''); 
      setError(''); 
      setNeedCondaEnv(false); 
      setCondaEnv(''); 
      setShowReplaceConfirm(false);
    } 
  }, [isOpen]);
  
  if (!isOpen || !card) return null;
  
  // 导入任务，支持 force 参数
  const handleImport = async (force = false) => { 
    if (!taskDir.trim()) { setError('请选择要导入的任务目录'); return; } 
    setIsImporting(true); 
    setError(''); 
    setShowReplaceConfirm(false);
    
    try { 
      const result = await api.importTask(taskDir.trim(), software, needCondaEnv ? condaEnv : null, force); 
      if (result.success) { 
        actions.showSuccess(`任务"${result.task.task_name}"导入成功！`); 
        onTaskImported(result.task); 
        onClose(); 
      } 
    } catch (err) { 
      // 检查是否需要选择Conda环境
      if (err.message && err.message.includes('NEED_CONDA_ENV')) {
        setNeedCondaEnv(true);
        setError('该任务缺少Conda环境配置，请选择一个环境');
      } 
      // 检查是否任务已存在
      else if (err.message && err.message.includes('TASK_EXISTS:')) {
        setShowReplaceConfirm(true);
      } 
      else {
        setError(err.message); 
      }
    } 
    finally { setIsImporting(false); } 
  };
  
  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
        <div className="relative bg-[#1a1f2e] rounded-2xl w-full max-w-xl border border-white/10 overflow-hidden">
          <div className={`p-6 bg-gradient-to-r ${card.gradient} bg-opacity-20`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Upload className="w-12 h-12 text-white/80" />
                <div>
                  <h3 className="text-2xl font-bold text-white">导入{card.name}任务</h3>
                  <p className="text-base text-white/70">导入已有的任务目录</p>
                </div>
              </div>
              <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10">
                <X size={24} className="text-white/70" />
              </button>
            </div>
          </div>
          <div className="p-6 space-y-5">
            {/* 错误提示 */}
            {error && (
              <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-start gap-3">
                <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
                <span style={{ whiteSpace: 'pre-wrap' }}>{error}</span>
              </div>
            )}
            
            {/* 替换确认提示 */}
            {showReplaceConfirm && (
              <div className="p-4 bg-yellow-500/20 border border-yellow-500/30 rounded-xl">
                <div className="flex items-start gap-3">
                  <AlertTriangle size={20} className="flex-shrink-0 mt-0.5 text-yellow-300" />
                  <div className="flex-1">
                    <p className="text-yellow-300 font-medium">该任务目录已经导入过了</p>
                    <p className="text-yellow-200/70 text-sm mt-1">是否要替换现有的任务记录？（任务文件不会被删除）</p>
                    <div className="flex gap-3 mt-3">
                      <button 
                        onClick={() => handleImport(true)} 
                        disabled={isImporting}
                        className="px-4 py-2 bg-yellow-500/30 hover:bg-yellow-500/50 rounded-lg text-yellow-200 text-sm font-medium disabled:opacity-50"
                      >
                        {isImporting ? '导入中...' : '替换导入'}
                      </button>
                      <button 
                        onClick={() => setShowReplaceConfirm(false)} 
                        className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-gray-300 text-sm"
                      >
                        取消
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">任务目录 <span className="text-red-400">*</span></label>
              <div className="flex gap-3">
                <div className="flex-1 px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                  {taskDir ? <span className="text-white font-mono truncate block">{taskDir}</span> : <span className="text-gray-500">请选择任务目录</span>}
                </div>
                <button onClick={() => setShowDirBrowser(true)} className="px-5 py-3.5 bg-white/10 hover:bg-white/20 border border-white/10 rounded-xl text-white flex items-center gap-2">
                  <FolderOpen size={20} />浏览
                </button>
              </div>
            </div>
            
            {/* 需要选择Conda环境时显示 */}
            {needCondaEnv && <CondaEnvSelector value={condaEnv} onChange={setCondaEnv} />}
            
            <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
              <h4 className="text-sm font-medium text-blue-300 mb-2">导入说明</h4>
              <ul className="text-xs text-blue-200/70 space-y-1">
                <li>• 任务目录必须包含 .dlhub/task.json 配置文件</li>
                <li>• 任务类型必须与当前页面匹配（{card.name}）</li>
              </ul>
            </div>
          </div>
          <div className="p-6 border-t border-white/10 flex justify-end gap-3">
            <button onClick={onClose} className="px-6 py-3 text-gray-400 hover:text-white">取消</button>
            <button 
              onClick={() => handleImport(false)} 
              disabled={isImporting || !taskDir.trim() || (needCondaEnv && !condaEnv) || showReplaceConfirm} 
              className={`flex items-center gap-2 px-8 py-3 bg-gradient-to-r ${card.gradient} rounded-xl text-white font-semibold disabled:opacity-50 hover:shadow-lg`}
            >
              {isImporting ? <Loader2 size={20} className="animate-spin" /> : <Upload size={20} />}
              {isImporting ? '导入中...' : '导入任务'}
            </button>
          </div>
        </div>
      </div>
      <DirectoryBrowser isOpen={showDirBrowser} onClose={() => setShowDirBrowser(false)} onSelect={(path) => setTaskDir(path)} title="选择任务目录" selectButtonText="选择此目录" />
    </>
  );
};

// ==================== 任务编辑弹窗（问题-3） ====================
const EditTaskModal = ({ isOpen, onClose, task, onTaskUpdated }) => {
  const [taskName, setTaskName] = useState(''); const [description, setDescription] = useState(''); const [condaEnv, setCondaEnv] = useState('');
  const [isSaving, setIsSaving] = useState(false); const [error, setError] = useState('');
  const { actions } = useApp();

  useEffect(() => { 
    if (isOpen && task) { 
      setTaskName(task.task_name || ''); 
      setDescription(task.description || ''); 
      setCondaEnv(task.conda_env || ''); 
      setError(''); 
    } 
  }, [isOpen, task]);

  if (!isOpen || !task) return null;

  const handleSave = async () => {
    if (!taskName.trim()) { setError('任务名称不能为空'); return; }
    setIsSaving(true); setError('');
    try {
      await api.updateTask(task.task_id, { task_name: taskName.trim(), description: description.trim(), conda_env: condaEnv });
      actions.showSuccess('任务信息已更新');
      onTaskUpdated({ ...task, task_name: taskName.trim(), description: description.trim(), conda_env: condaEnv });
      onClose();
    } catch (err) { setError(err.message); }
    finally { setIsSaving(false); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-[#1a1f2e] rounded-2xl w-full max-w-xl border border-white/10 overflow-hidden">
        <div className="p-6 border-b border-white/10 bg-gradient-to-r from-blue-500/10 to-purple-500/10">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold text-white flex items-center gap-3"><Edit className="w-6 h-6 text-blue-400" />编辑任务</h3>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10"><X size={20} className="text-gray-400" /></button>
          </div>
        </div>
        <div className="p-6 space-y-5">
          {error && <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-center gap-3"><AlertCircle size={20} />{error}</div>}
          <div><label className="block text-sm font-medium text-gray-300 mb-2">任务名称 <span className="text-red-400">*</span></label><input type="text" value={taskName} onChange={(e) => setTaskName(e.target.value)} className="w-full px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-blue-500" /></div>
          <CondaEnvSelector value={condaEnv} onChange={setCondaEnv} />
          <div><label className="block text-sm font-medium text-gray-300 mb-2">任务描述</label><textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} className="w-full px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-blue-500 resize-none" /></div>
          <div className="p-3 bg-white/5 rounded-xl"><p className="text-xs text-gray-500">任务ID: {task.task_id}</p><p className="text-xs text-gray-500">工作目录: {task.path}</p></div>
        </div>
        <div className="p-6 border-t border-white/10 flex justify-end gap-3">
          <button onClick={onClose} className="px-6 py-3 text-gray-400 hover:text-white">取消</button>
          <button onClick={handleSave} disabled={isSaving || !taskName.trim()} className="flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl text-white font-semibold disabled:opacity-50">{isSaving ? <Loader2 size={20} className="animate-spin" /> : <Check size={20} />}{isSaving ? '保存中...' : '保存'}</button>
        </div>
      </div>
    </div>
  );
};

// ==================== 任务卡片（问题-6：显示磁盘占用，优化-7：复制功能，新增：使用中状态和结束任务） ====================
const TaskCard = ({ task, cardConfig, onLaunch, onDelete, onEdit, onCopy, runningTaskId, onStop, isDragging }) => {
  const [deleting, setDeleting] = useState(false); const [showMenu, setShowMenu] = useState(false); const [showDeleteConfirm, setShowDeleteConfirm] = useState(false); 
  const [expanded, setExpanded] = useState(false); const [taskSize, setTaskSize] = useState(null); const [copying, setCopying] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [loadingSize, setLoadingSize] = useState(false);  // 新增：加载状态
  const menuRef = useRef(null); const { actions } = useApp();

  // 判断是否是正在运行的任务
  const isRunning = runningTaskId === task.task_id;

  useEffect(() => { const handleClickOutside = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) setShowMenu(false); }; if (showMenu) document.addEventListener('mousedown', handleClickOutside); return () => document.removeEventListener('mousedown', handleClickOutside); }, [showMenu]);

  // 问题-6：展开时获取磁盘占用（改进：添加加载状态和错误处理）
  useEffect(() => {
    if (expanded && !taskSize && !loadingSize) {
      setLoadingSize(true);
      api.getTaskSize(task.task_id)
        .then(setTaskSize)
        .catch(err => console.error('获取任务大小失败:', err))
        .finally(() => setLoadingSize(false));
    }
  }, [expanded, task.task_id, taskSize, loadingSize]);

  // 状态配置 - 新增"使用中"状态
  const statusConfig = { 
    idle: { color: 'bg-gray-500', text: '未开始', textColor: 'text-gray-400' }, 
    training: { color: 'bg-blue-500 animate-pulse', text: '训练中', textColor: 'text-blue-400' }, 
    completed: { color: 'bg-green-500', text: '已完成', textColor: 'text-green-400' }, 
    interrupted: { color: 'bg-yellow-500', text: '已中断', textColor: 'text-yellow-400' }, 
    error: { color: 'bg-red-500', text: '出错', textColor: 'text-red-400' } 
  };
  
  // 如果是正在运行的任务，覆盖状态显示为"使用中"
  const status = isRunning 
    ? { color: 'bg-emerald-500 animate-pulse', text: '使用中', textColor: 'text-emerald-400' }
    : (statusConfig[task.status] || statusConfig.idle);

  const handleDeleteConfirm = async () => { setDeleting(true); try { await onDelete(task.task_id); actions.showSuccess(`任务"${task.task_name}"已删除`); } catch (err) { actions.showError(err.message); } finally { setDeleting(false); setShowMenu(false); } };

  // 优化-7：复制任务
  const handleCopy = async () => {
    setCopying(true);
    try {
      await onCopy(task.task_id);
      actions.showSuccess(`任务已复制`);
    } catch (err) { actions.showError(err.message); }
    finally { setCopying(false); setShowMenu(false); }
  };

  // 新增：结束任务
  const handleStop = async () => {
    setStopping(true);
    try {
      await onStop(task.task_id);
      actions.showSuccess(`任务"${task.task_name}"已结束`);
    } catch (err) { actions.showError(err.message); }
    finally { setStopping(false); setShowMenu(false); }
  };

  const formatTime = (isoString) => { if (!isoString) return '未知'; const date = new Date(isoString); const now = new Date(); const diff = now - date; if (diff < 60000) return '刚刚'; if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`; if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`; if (diff < 604800000) return `${Math.floor(diff / 86400000)}天前`; return date.toLocaleDateString(); };
  const formatDate = (isoString) => { if (!isoString) return '未知'; return new Date(isoString).toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }); };
  const truncatePath = (path, maxLen = 40) => { if (!path || path.length <= maxLen) return path; return '...' + path.slice(-maxLen); };

  return (
    <>
      <div className={`relative bg-white/[0.03] hover:bg-white/[0.06] border border-white/10 rounded-2xl transition-all group ${isDragging ? 'opacity-50 scale-95' : ''} ${isRunning ? 'ring-2 ring-emerald-500/50' : ''}`}>
        {/* 使用中角标 */}
        {isRunning && (
          <div className="absolute -top-2 -right-2 bg-emerald-500 text-white text-xs px-2 py-0.5 rounded-full font-semibold shadow-lg z-10">
            使用中
          </div>
        )}
        <div className="p-5 cursor-pointer" onClick={() => onLaunch(task.task_id)}>
          <div className="absolute top-4 right-4 flex items-center gap-2">
            {/* [UI优化] 使用新的状态徽章组件 */}
            <TaskStatusBadge status={task.status} isRunning={isRunning} size="sm" />
            <div className="relative" ref={menuRef} onClick={e => e.stopPropagation()}>
              <button onClick={() => setShowMenu(!showMenu)} className="p-1.5 rounded-lg hover:bg-white/10 opacity-0 group-hover:opacity-100"><MoreHorizontal size={18} className="text-gray-500" /></button>
              {showMenu && (
                <div className="absolute right-0 top-8 bg-[#1a1f2e] border border-white/10 rounded-xl shadow-xl z-10 py-2 min-w-[160px]">
                  <button onClick={() => { setShowMenu(false); onEdit(task); }} className="w-full px-4 py-2.5 text-left text-gray-300 hover:bg-white/5 flex items-center gap-3 text-sm"><Edit size={16} />编辑任务</button>
                  <button onClick={handleCopy} disabled={copying} className="w-full px-4 py-2.5 text-left text-gray-300 hover:bg-white/5 flex items-center gap-3 text-sm">{copying ? <Loader2 size={16} className="animate-spin" /> : <Copy size={16} />}复制任务</button>
                  {/* 仅当任务正在运行时显示结束按钮 */}
                  {isRunning && (
                    <>
                      <div className="border-t border-white/10 my-1" />
                      <button onClick={handleStop} disabled={stopping} className="w-full px-4 py-2.5 text-left text-orange-400 hover:bg-white/5 flex items-center gap-3 text-sm">{stopping ? <Loader2 size={16} className="animate-spin" /> : <StopCircle size={16} />}结束任务</button>
                    </>
                  )}
                  <div className="border-t border-white/10 my-1" />
                  <button onClick={() => { setShowMenu(false); setShowDeleteConfirm(true); }} disabled={deleting} className="w-full px-4 py-2.5 text-left text-red-400 hover:bg-white/5 flex items-center gap-3 text-sm">{deleting ? <Loader2 size={16} className="animate-spin" /> : <Trash2 size={16} />}删除任务</button>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-start gap-4"><span className="text-3xl">{cardConfig?.icon || '📋'}</span><div className="flex-1 min-w-0"><h4 className="font-semibold text-white text-lg truncate pr-20">{task.task_name}</h4><p className="text-sm text-gray-500 truncate">{task.description || '无描述'}</p></div></div>
          {task.conda_env && <div className="mt-3 flex items-center gap-2"><Terminal size={14} className="text-gray-500" /><span className="text-xs text-gray-500 font-mono">{task.conda_env}</span></div>}
          <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between"><div className="flex items-center gap-2 text-sm text-gray-500"><Clock size={14} /><span>{formatTime(task.updated_at)}</span></div>{task.best_metric && <div className={`text-base font-semibold bg-gradient-to-r ${cardConfig?.gradient || 'from-blue-500 to-purple-500'} bg-clip-text text-transparent`}>{task.best_metric.name}: {task.best_metric.value}%</div>}</div>
        </div>
        <button onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }} className="w-full py-2 border-t border-white/5 text-gray-500 hover:text-gray-300 text-xs flex items-center justify-center gap-1 hover:bg-white/5 transition-colors">{expanded ? '收起详情' : '展开详情'}<ChevronRight size={14} className={`transform transition-transform ${expanded ? 'rotate-90' : ''}`} /></button>
        {expanded && (
          <div className="px-5 pb-5 space-y-3 border-t border-white/5 bg-white/[0.02]">
            <div className="pt-4 grid grid-cols-2 gap-3 text-sm">
              <div><p className="text-gray-500 text-xs mb-1">任务ID</p><p className="text-white font-mono text-xs truncate">{task.task_id}</p></div>
              <div><p className="text-gray-500 text-xs mb-1">任务类型</p><p className="text-white">{cardConfig?.name || task.task_type}</p></div>
              <div><p className="text-gray-500 text-xs mb-1">创建时间</p><p className="text-white text-xs">{formatDate(task.created_at)}</p></div>
              <div><p className="text-gray-500 text-xs mb-1">更新时间</p><p className="text-white text-xs">{formatDate(task.updated_at)}</p></div>
            </div>
            {task.path && <div><p className="text-gray-500 text-xs mb-1">工作目录</p><p className="text-white font-mono text-xs bg-black/20 px-2 py-1.5 rounded truncate" title={task.path}>{truncatePath(task.path, 50)}</p></div>}
            {/* 问题-6：显示磁盘占用（改进：加载状态） */}
            <div className="flex items-center justify-between p-2 bg-black/20 rounded">
              <span className="text-gray-500 text-xs">磁盘占用</span>
              {loadingSize ? (
                <span className="text-gray-400 text-xs flex items-center gap-1"><Loader2 size={12} className="animate-spin" />计算中...</span>
              ) : taskSize ? (
                <span className="text-white text-xs font-mono">{taskSize.size_formatted} ({taskSize.file_count} 文件)</span>
              ) : (
                <span className="text-gray-500 text-xs">无法获取</span>
              )}
            </div>
          </div>
        )}
      </div>
      <ConfirmModal isOpen={showDeleteConfirm} onClose={() => setShowDeleteConfirm(false)} onConfirm={handleDeleteConfirm} title="确认删除任务" message={`确定要删除任务"${task.task_name}"吗？这将删除任务目录下的所有文件。此操作无法撤销。`} confirmText="删除" cancelText="取消" danger={true} />
    </>
  );
};

// ==================== 任务管理页面（BUG-3修复、问题-7排序、新增：运行状态和结束任务） ====================
const TaskManagerPage = ({ software, onBack }) => {
  const [showCreateModal, setShowCreateModal] = useState(false); const [showImportModal, setShowImportModal] = useState(false); 
  const [showEditModal, setShowEditModal] = useState(false); const [editingTask, setEditingTask] = useState(null);
  const [tasks, setTasks] = useState([]); const [loading, setLoading] = useState(true); const [launching, setLaunching] = useState(null); const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('updated_at'); const [sortOrder, setSortOrder] = useState('desc');
  const [runningTaskId, setRunningTaskId] = useState(null);  // 新增：当前运行的任务ID
  
  // BUG-3修复：使用ref追踪组件是否已卸载
  const isMountedRef = useRef(true);
  const intervalRef = useRef(null); 
  const { actions } = useApp(); 
  const card = SOFTWARE_CARDS.find(c => c.id === software);

  const fetchTasks = useCallback(async () => { 
    try { 
      const data = await api.listTasks(software); 
      if (isMountedRef.current) {
        setTasks(data.tasks || []); 
        setRunningTaskId(data.running_task_id || null);  // 获取运行中的任务
      }
    } catch (err) { console.error('获取任务列表失败:', err); } 
    finally { 
      if (isMountedRef.current) {
        setLoading(false); 
      }
    } 
  }, [software]);

  // BUG-3修复：清理轮询和组件卸载标记
  useEffect(() => { 
    isMountedRef.current = true;
    fetchTasks(); 
    
    const startPolling = () => { 
      if (!intervalRef.current) {
        intervalRef.current = setInterval(() => {
          if (isMountedRef.current) fetchTasks();
        }, 5000); 
      }
    }; 
    
    const stopPolling = () => { 
      if (intervalRef.current) { 
        clearInterval(intervalRef.current); 
        intervalRef.current = null; 
      } 
    }; 
    
    const handleVisibilityChange = () => { 
      if (document.hidden) stopPolling(); 
      else { fetchTasks(); startPolling(); } 
    }; 
    
    startPolling(); 
    document.addEventListener('visibilitychange', handleVisibilityChange); 
    
    return () => { 
      isMountedRef.current = false;
      stopPolling(); 
      document.removeEventListener('visibilitychange', handleVisibilityChange); 
    }; 
  }, [fetchTasks]);

  // 运行中任务冲突确认
  const [conflictInfo, setConflictInfo] = useState(null);
  const [pendingTaskId, setPendingTaskId] = useState(null);

  const handleLaunch = async (taskId, force = false) => { 
    if (launching) return; 
    setLaunching(taskId); 
    try { 
      const result = await api.launchApp(taskId, force); 
      if (result.success) { 
        actions.showSuccess(`训练界面已在新标签页打开！`); 
        actions.setRunningApp({ taskId, url: result.url }); 
        setRunningTaskId(taskId);  // BUG修复：立即更新运行状态
        window.open(result.url, '_blank'); 
      } 
    } catch (err) { 
      // 检查是否是APP_RUNNING冲突错误
      if (err.message && err.message.includes('APP_RUNNING:')) {
        const parts = err.message.split(':');
        const runningTaskId = parts[1];
        const runningTaskName = parts[2] || '未知任务';
        setConflictInfo({ taskId: runningTaskId, taskName: runningTaskName });
        setPendingTaskId(taskId);
      } else {
        actions.showError(err.message); 
      }
    } 
    finally { setLaunching(null); } 
  };

  // 确认强制启动
  const handleForceConfirm = () => {
    if (pendingTaskId) {
      handleLaunch(pendingTaskId, true);
    }
    setConflictInfo(null);
    setPendingTaskId(null);
  };

  // 取消强制启动
  const handleForceCancel = () => {
    setConflictInfo(null);
    setPendingTaskId(null);
  };

  const handleDelete = async (taskId) => { try { await api.deleteTask(taskId); setTasks(tasks.filter(t => t.task_id !== taskId)); } catch (err) { throw err; } };
  const handleTaskCreated = (task) => { setTasks([task, ...tasks]); handleLaunch(task.task_id); };
  const handleTaskImported = (task) => { setTasks([task, ...tasks]); };
  const handleEdit = (task) => { setEditingTask(task); setShowEditModal(true); };
  const handleTaskUpdated = (updatedTask) => { setTasks(tasks.map(t => t.task_id === updatedTask.task_id ? { ...t, ...updatedTask } : t)); };
  
  // 优化-7：复制任务
  const handleCopy = async (taskId) => {
    const result = await api.copyTask(taskId);
    if (result.success) {
      setTasks([result.task, ...tasks]);
    }
  };

  // 新增：结束任务
  const handleStop = async (taskId) => {
    try {
      // BUG修复：验证确保停止的是正确的任务
      if (runningTaskId && runningTaskId !== taskId) {
        actions.showError('任务状态已变更，请刷新页面');
        fetchTasks();
        return;
      }
      await api.stopApp();
      setRunningTaskId(null);  // 清除运行中的任务
      fetchTasks();  // 刷新列表
    } catch (err) {
      throw err;
    }
  };

  // 问题-7：排序功能
  const sortedTasks = useMemo(() => {
    let filtered = tasks.filter(task => {
      if (!searchTerm) return true;
      const term = searchTerm.toLowerCase();
      return task.task_name?.toLowerCase().includes(term) || task.description?.toLowerCase().includes(term) || task.conda_env?.toLowerCase().includes(term);
    });
    
    filtered.sort((a, b) => {
      let aVal, bVal;
      switch (sortBy) {
        case 'name': aVal = a.task_name?.toLowerCase() || ''; bVal = b.task_name?.toLowerCase() || ''; break;
        case 'status': aVal = a.status || ''; bVal = b.status || ''; break;
        case 'created_at': aVal = a.created_at || ''; bVal = b.created_at || ''; break;
        default: aVal = a.updated_at || ''; bVal = b.updated_at || '';
      }
      if (sortOrder === 'asc') return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
    });
    
    return filtered;
  }, [tasks, searchTerm, sortBy, sortOrder]);

  const toggleSort = (field) => {
    if (sortBy === field) setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    else { setSortBy(field); setSortOrder('desc'); }
  };

  // [UI优化] 快捷键提示显示
  const [showShortcuts, setShowShortcuts] = useState(false);
  const searchInputRef = useRef(null);

  // [UI优化] 快捷键支持
  useKeyboardShortcuts([
    { key: 'n', ctrl: true, action: () => setShowCreateModal(true) },
    { key: 'i', ctrl: true, action: () => setShowImportModal(true) },
    { key: 'f', ctrl: true, action: () => searchInputRef.current?.focus() },
    { key: 'F5', action: () => fetchTasks() },
    { key: 'Escape', action: () => {
      if (showCreateModal) setShowCreateModal(false);
      else if (showImportModal) setShowImportModal(false);
      else if (showEditModal) { setShowEditModal(false); setEditingTask(null); }
      else if (showShortcuts) setShowShortcuts(false);
      else onBack();
    }},
    { key: '?', shift: true, action: () => setShowShortcuts(true) },
  ]);

  if (!card) return null;
  return (
    <div className="min-h-screen bg-[#0a0d12] text-white">
      <ParticleBackground />
      <header className="relative z-10 border-b border-white/10 bg-[#0a0d12]/80 backdrop-blur-xl sticky top-0">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button onClick={onBack} className="p-2.5 rounded-xl hover:bg-white/10"><ChevronLeft size={26} /></button>
              <div className="flex items-center gap-4">
                <span className="text-4xl">{card.icon}</span>
                <div><h1 className="text-2xl font-bold">{card.name}</h1><p className="text-base text-gray-500">{card.subtitle}</p></div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {/* [UI优化] 紧凑的系统资源监控 */}
              <SystemResourceMonitor collapsed={true} />
              <div className="relative">
                <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input 
                  ref={searchInputRef}
                  type="text" 
                  value={searchTerm} 
                  onChange={(e) => setSearchTerm(e.target.value)} 
                  placeholder="搜索任务... (Ctrl+F)" 
                  className="pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 w-52" 
                />
              </div>
              <button onClick={fetchTasks} className="p-2.5 rounded-xl hover:bg-white/10" title="刷新 (F5)"><RefreshCw size={22} className={loading ? 'animate-spin' : ''} /></button>
              <button onClick={() => setShowShortcuts(true)} className="p-2.5 rounded-xl hover:bg-white/10" title="快捷键 (Shift+?)">⌨️</button>
              <button onClick={() => setShowImportModal(true)} className="flex items-center gap-2 px-5 py-3 bg-white/10 hover:bg-white/20 border border-white/10 rounded-xl text-white font-medium" title="Ctrl+I"><Upload size={20} />导入</button>
              <button onClick={() => setShowCreateModal(true)} className={`flex items-center gap-2 px-6 py-3 bg-gradient-to-r ${card.gradient} rounded-xl text-white font-semibold hover:shadow-lg`} title="Ctrl+N"><Plus size={22} />新建</button>
            </div>
          </div>
        </div>
      </header>
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* 主内容区 */}
          <div className="lg:col-span-3">
            {launching && <div className="mb-6 p-4 bg-blue-500/20 border border-blue-500/30 rounded-xl text-blue-300 flex items-center gap-3"><Loader2 className="animate-spin" size={20} />正在启动训练界面...</div>}
            <div className="mb-6">
              {/* 问题-7：排序控制 */}
              <div className="flex items-center justify-between mb-5">
                <h2 className="text-xl font-semibold text-white flex items-center gap-3"><Clock size={22} className="text-gray-500" />任务列表<span className="text-base font-normal text-gray-500">({sortedTasks.length}{searchTerm && tasks.length !== sortedTasks.length ? ` / ${tasks.length}` : ''})</span></h2>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-500">排序:</span>
                  <button onClick={() => toggleSort('updated_at')} className={`px-3 py-1.5 text-sm rounded-lg ${sortBy === 'updated_at' ? 'bg-blue-500/20 text-blue-400' : 'text-gray-400 hover:bg-white/10'}`}>更新时间 {sortBy === 'updated_at' && (sortOrder === 'desc' ? '↓' : '↑')}</button>
                  <button onClick={() => toggleSort('name')} className={`px-3 py-1.5 text-sm rounded-lg ${sortBy === 'name' ? 'bg-blue-500/20 text-blue-400' : 'text-gray-400 hover:bg-white/10'}`}>名称 {sortBy === 'name' && (sortOrder === 'desc' ? '↓' : '↑')}</button>
                  <button onClick={() => toggleSort('created_at')} className={`px-3 py-1.5 text-sm rounded-lg ${sortBy === 'created_at' ? 'bg-blue-500/20 text-blue-400' : 'text-gray-400 hover:bg-white/10'}`}>创建时间 {sortBy === 'created_at' && (sortOrder === 'desc' ? '↓' : '↑')}</button>
                </div>
              </div>
              {/* 优化-1：骨架屏 */}
              {loading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {[1,2,3,4].map(i => <TaskCardSkeleton key={i} />)}
                </div>
              ) : sortedTasks.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {sortedTasks.map((task) => <TaskCard key={task.task_id} task={task} cardConfig={card} onLaunch={handleLaunch} onDelete={handleDelete} onEdit={handleEdit} onCopy={handleCopy} runningTaskId={runningTaskId} onStop={handleStop} />)}
                </div>
              ) : searchTerm ? (
                <div className="text-center py-20 bg-white/[0.02] rounded-2xl border border-white/5"><div className="w-20 h-20 rounded-2xl bg-white/5 flex items-center justify-center mx-auto mb-5"><Search size={40} className="text-gray-600" /></div><p className="text-lg text-gray-500 mb-2">未找到匹配的任务</p></div>
              ) : (
                <div className="text-center py-20 bg-white/[0.02] rounded-2xl border border-white/5"><div className="w-20 h-20 rounded-2xl bg-white/5 flex items-center justify-center mx-auto mb-5"><FolderOpen size={40} className="text-gray-600" /></div><p className="text-lg text-gray-500 mb-2">暂无任务</p><p className="text-base text-gray-600">点击"新建任务"开始</p></div>
              )}
            </div>
          </div>
          
          {/* [UI优化] 右侧边栏 - 系统资源监控 */}
          <div className="lg:col-span-1 space-y-4">
            <SystemResourceMonitor collapsed={false} />
            
            {/* 快捷操作提示 */}
            <div className="bg-white/[0.03] border border-white/10 rounded-xl p-4">
              <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                ⚡ 快捷操作
              </h4>
              <div className="space-y-2 text-xs text-gray-500">
                <div className="flex justify-between"><span>新建任务</span><kbd className="px-1.5 py-0.5 bg-white/10 rounded">Ctrl+N</kbd></div>
                <div className="flex justify-between"><span>导入任务</span><kbd className="px-1.5 py-0.5 bg-white/10 rounded">Ctrl+I</kbd></div>
                <div className="flex justify-between"><span>搜索</span><kbd className="px-1.5 py-0.5 bg-white/10 rounded">Ctrl+F</kbd></div>
                <div className="flex justify-between"><span>刷新</span><kbd className="px-1.5 py-0.5 bg-white/10 rounded">F5</kbd></div>
              </div>
            </div>
            
            {/* 使用指南按钮 */}
            <button
              onClick={() => {
                const manualFiles = {
                  classification: 'classification-manual.html',
                  detection: 'detection-manual.html',
                  segmentation: 'segmentation-manual.html',
                  anomaly: 'anomaly-detection-manual.html',
                  ocr: 'ocr-manual.html',
                  sevseg: 'sevseg-manual.html'
                };
                window.open(`/manuals/${manualFiles[software]}`, '_blank');
              }}
              className={`w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r ${card.gradient} bg-opacity-20 hover:bg-opacity-30 border border-white/10 hover:border-white/20 rounded-xl text-white font-medium transition-all duration-300 group`}
            >
              <BookOpen size={18} className="group-hover:scale-110 transition-transform" />
              <span>使用指南</span>
              <ExternalLink size={14} className="text-white/60 group-hover:text-white/90 transition-colors" />
            </button>
          </div>
        </div>
      </main>
      <CreateTaskModal isOpen={showCreateModal} onClose={() => setShowCreateModal(false)} software={software} onTaskCreated={handleTaskCreated} />
      <ImportTaskModal isOpen={showImportModal} onClose={() => setShowImportModal(false)} software={software} onTaskImported={handleTaskImported} />
      <EditTaskModal isOpen={showEditModal} onClose={() => { setShowEditModal(false); setEditingTask(null); }} task={editingTask} onTaskUpdated={handleTaskUpdated} />
      
      {/* [UI优化] 快捷键提示弹窗 */}
      <KeyboardShortcutHint show={showShortcuts} onClose={() => setShowShortcuts(false)} />
      
      {/* 任务冲突确认弹窗 */}
      <ConfirmModal 
        isOpen={conflictInfo !== null} 
        onClose={handleForceCancel} 
        onConfirm={handleForceConfirm}
        title="有任务正在运行"
        message={`当前有任务"${conflictInfo?.taskName || ''}"正在运行中。\n\n启动新任务将会停止当前任务，可能导致训练进度丢失。\n\n确定要继续吗？`}
        confirmText="停止并启动新任务"
        cancelText="取消"
        danger={true}
      />
    </div>
  );
};

// ==================== 软件卡片组件 ====================
const SoftwareCard = ({ card, onClick, isHovered, onHover }) => {
  const isDisabled = card.disabled;
  
  return (
    <div 
      className={`group relative transition-all duration-500 ${isDisabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'} ${isHovered && !isDisabled ? 'scale-[1.03] z-20' : 'scale-100'}`} 
      style={{ transform: isHovered && !isDisabled ? 'translateY(-8px)' : 'translateY(0)' }} 
      onMouseEnter={() => !isDisabled && onHover(card.id)} 
      onMouseLeave={() => onHover(null)} 
      onClick={() => !isDisabled && onClick()}
    >
      {!isDisabled && <div className={`absolute -inset-1 bg-gradient-to-r ${card.gradient} rounded-2xl blur-xl opacity-0 group-hover:opacity-40 transition-opacity duration-500`} />}
      <div className={`relative bg-[#0f1419]/90 backdrop-blur-xl border border-white/10 rounded-2xl p-6 overflow-hidden transition-all duration-300 h-[320px] flex flex-col ${isHovered && !isDisabled ? 'border-white/20' : ''}`}>
        <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${card.gradient} transform origin-left transition-transform duration-500 ${isHovered && !isDisabled ? 'scale-x-100' : 'scale-x-0'}`} />
        
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`text-5xl transform transition-transform duration-500 ${isHovered && !isDisabled ? 'scale-110 rotate-6' : ''}`}>{card.icon}</div>
            <div><h3 className="text-xl font-bold text-white">{card.name}</h3><p className="text-sm text-gray-500 font-mono">{card.subtitle}</p></div>
          </div>
          <div className={`w-11 h-11 rounded-xl bg-gradient-to-r ${card.gradient} flex items-center justify-center transform transition-all duration-300 ${isHovered && !isDisabled ? 'rotate-0 scale-100' : '-rotate-45 scale-75'}`}><ChevronRight className="w-6 h-6 text-white" /></div>
        </div>
        <p className="text-sm text-gray-400 mb-5 line-clamp-2 flex-shrink-0">{card.description}</p>
        <div className="flex flex-wrap gap-2 mb-5 flex-shrink-0">{card.features.slice(0, 3).map((feature, i) => (<span key={i} className={`px-3 py-1.5 text-sm rounded-full bg-white/5 text-gray-400 border border-white/10 transition-all duration-300 ${isHovered && !isDisabled ? 'bg-white/10 border-white/20' : ''}`}>{feature}</span>))}</div>
        <div className="flex items-center justify-between pt-5 border-t border-white/5 mt-auto">
          <div><p className="text-sm text-gray-500">{card.metric.name}</p><p className={`text-2xl font-bold font-mono bg-gradient-to-r ${card.gradient} bg-clip-text text-transparent`}>{card.metric.value}</p></div>
          <div className="flex gap-5 text-sm text-gray-500"><div className="text-center"><p className="text-white font-semibold text-base">{card.stats.models}</p><p>模型</p></div><div className="text-center"><p className="text-white font-semibold text-base">{card.stats.speed}</p><p>推理</p></div></div>
        </div>
        {isDisabled && (<div className="absolute inset-0 bg-black/40 rounded-2xl flex items-center justify-center"><span className="text-gray-400 text-lg font-medium">敬请期待</span></div>)}
      </div>
    </div>
  );
};

// ==================== 主页面（优化-3：主题切换） ====================
const HomePage = ({ onSelectSoftware }) => {
  const [hoveredCard, setHoveredCard] = useState(null);
  const [showHelp, setShowHelp] = useState(false);
  const { state, actions } = useApp();
  const { isDark, toggleTheme } = useTheme();

  useEffect(() => {
    if (state.isFirstVisit && !state.gpuLoading) {
      setShowHelp(true);
    }
  }, [state.isFirstVisit, state.gpuLoading]);

  return (
    <div className={`min-h-screen ${isDark ? 'bg-[#0a0d12]' : 'bg-gray-100'} text-white overflow-hidden flex flex-col`}>
      <ParticleBackground />
      
      {/* 顶部工具栏 */}
      <div className="fixed top-6 right-6 z-50 flex items-center gap-3">
        {/* 优化-3：主题切换按钮 */}
        <button 
          onClick={toggleTheme}
          className="w-12 h-12 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 flex items-center justify-center transition-all hover:scale-110"
          title={isDark ? '切换到浅色主题' : '切换到深色主题'}
        >
          {isDark ? <Sun size={22} className="text-yellow-400" /> : <Moon size={22} className="text-blue-400" />}
        </button>
        <button 
          onClick={() => setShowHelp(true)}
          className="w-12 h-12 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 flex items-center justify-center transition-all hover:scale-110 group"
          title="使用帮助"
        >
          <HelpCircle size={24} className="text-white/70 group-hover:text-white" />
        </button>
      </div>
      
      <header className="relative z-10 text-center pt-8 pb-6 px-8">
        <div className="inline-flex items-center gap-5">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur-xl opacity-50 animate-pulse" />
            <div className="relative w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center text-3xl shadow-2xl">🧠</div>
          </div>
          <div className="text-left">
            <h1 className={`text-4xl font-black bg-gradient-to-r ${isDark ? 'from-white via-blue-200 to-purple-200' : 'from-gray-800 via-blue-600 to-purple-600'} bg-clip-text text-transparent`}>DL-Hub</h1>
            <p className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'} font-mono`}>Deep Learning Workstation v2.0</p>
          </div>
        </div>
      </header>
      
      <main className="flex-1 flex items-center justify-center relative z-10 px-6 pb-6">
        <div className="w-full max-w-7xl">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {SOFTWARE_CARDS.map((card) => (
              <SoftwareCard key={card.id} card={card} onClick={() => onSelectSoftware(card.id)} isHovered={hoveredCard === card.id} onHover={setHoveredCard} />
            ))}
          </div>
        </div>
      </main>
      
      <footer className={`relative z-10 text-center py-4 ${isDark ? 'text-gray-600' : 'text-gray-500'} text-sm`}>
        <p>DL-Hub v2.0 · Powered by Deep Learning</p>
      </footer>
      
      <HelpGuideModal isOpen={showHelp} onClose={() => setShowHelp(false)} />
    </div>
  );
};

// ==================== 主应用内容 ====================
function AppContent() {
  const [currentPage, setCurrentPage] = useState('home');
  const [selectedSoftware, setSelectedSoftware] = useState(null);
  const { state } = useApp();

  if (state.gpuLoading) {
    return (
      <div className="min-h-screen bg-[#0a0d12] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-14 h-14 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-lg text-gray-400">加载中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="font-sans">
      <GlobalNotification />
      {currentPage === 'home' && (
        <HomePage onSelectSoftware={(id) => { if (id !== 'developing') { setSelectedSoftware(id); setCurrentPage('task-manager'); } }} />
      )}
      {currentPage === 'task-manager' && (
        <TaskManagerPage software={selectedSoftware} onBack={() => { setCurrentPage('home'); setSelectedSoftware(null); }} />
      )}
    </div>
  );
}

// ==================== 主应用 ====================
export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
