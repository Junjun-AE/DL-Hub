/**
 * DL-Hub 全局状态管理
 * 
 * 修复：
 * - 错误通知持久显示，直到用户手动关闭
 */

import React, { createContext, useContext, useReducer, useEffect } from 'react';
import api from '../api';

// ==================== 初始状态 ====================
const initialState = {
  // GPU信息
  gpuInfo: null,
  gpuLoading: true,
  
  // 全局通知
  notification: null, // { type: 'success' | 'error' | 'warning' | 'info', message: string }
  
  // 帮助弹窗
  showHelpModal: false,
  
  // 运行中的应用
  runningApp: null, // { taskId, url }
  
  // 首次使用标记
  isFirstVisit: true,
};

// ==================== Action Types ====================
const ActionTypes = {
  SET_GPU_INFO: 'SET_GPU_INFO',
  SET_GPU_LOADING: 'SET_GPU_LOADING',
  SHOW_NOTIFICATION: 'SHOW_NOTIFICATION',
  HIDE_NOTIFICATION: 'HIDE_NOTIFICATION',
  SET_SHOW_HELP_MODAL: 'SET_SHOW_HELP_MODAL',
  SET_RUNNING_APP: 'SET_RUNNING_APP',
  SET_FIRST_VISIT: 'SET_FIRST_VISIT',
};

// ==================== Reducer ====================
function appReducer(state, action) {
  switch (action.type) {
    case ActionTypes.SET_GPU_INFO:
      return { ...state, gpuInfo: action.payload, gpuLoading: false };
    case ActionTypes.SET_GPU_LOADING:
      return { ...state, gpuLoading: action.payload };
    case ActionTypes.SHOW_NOTIFICATION:
      return { ...state, notification: action.payload };
    case ActionTypes.HIDE_NOTIFICATION:
      return { ...state, notification: null };
    case ActionTypes.SET_SHOW_HELP_MODAL:
      return { ...state, showHelpModal: action.payload };
    case ActionTypes.SET_RUNNING_APP:
      return { ...state, runningApp: action.payload };
    case ActionTypes.SET_FIRST_VISIT:
      return { ...state, isFirstVisit: action.payload };
    default:
      return state;
  }
}

// ==================== Context ====================
const AppContext = createContext(null);

// ==================== Provider ====================
export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  
  // 初始化：获取GPU信息和检查首次访问
  useEffect(() => {
    const init = async () => {
      // 检查首次访问
      const visited = localStorage.getItem('dlhub_visited');
      if (visited) {
        dispatch({ type: ActionTypes.SET_FIRST_VISIT, payload: false });
      }
      
      // 获取GPU信息
      try {
        const gpu = await api.getGpuInfo();
        dispatch({ type: ActionTypes.SET_GPU_INFO, payload: gpu });
      } catch (e) {
        console.log('获取GPU信息失败:', e);
        dispatch({ type: ActionTypes.SET_GPU_LOADING, payload: false });
      }
    };
    
    init();
  }, []);
  
  // Actions
  const actions = {
    /**
     * 显示通知
     * @param {string} type - 通知类型: 'success' | 'error' | 'warning' | 'info'
     * @param {string} message - 通知内容
     * @param {number} duration - 显示时长(毫秒)，0表示不自动关闭
     */
    showNotification: (type, message, duration = 5000) => {
      dispatch({ type: ActionTypes.SHOW_NOTIFICATION, payload: { type, message } });
      if (duration > 0) {
        setTimeout(() => {
          dispatch({ type: ActionTypes.HIDE_NOTIFICATION });
        }, duration);
      }
    },
    
    hideNotification: () => {
      dispatch({ type: ActionTypes.HIDE_NOTIFICATION });
    },
    
    // 成功通知：5秒后自动关闭
    showSuccess: (message) => actions.showNotification('success', message, 5000),
    
    // 【修复】错误通知：不自动关闭，需要用户手动点击关闭
    showError: (message) => actions.showNotification('error', message, 0),
    
    // 警告通知：8秒后自动关闭
    showWarning: (message) => actions.showNotification('warning', message, 8000),
    
    // 信息通知：5秒后自动关闭
    showInfo: (message) => actions.showNotification('info', message, 5000),
    
    setShowHelpModal: (show) => {
      dispatch({ type: ActionTypes.SET_SHOW_HELP_MODAL, payload: show });
    },
    
    setRunningApp: (app) => {
      dispatch({ type: ActionTypes.SET_RUNNING_APP, payload: app });
    },
    
    markVisited: () => {
      localStorage.setItem('dlhub_visited', 'true');
      dispatch({ type: ActionTypes.SET_FIRST_VISIT, payload: false });
    },
    
    resetFirstVisit: () => {
      localStorage.removeItem('dlhub_visited');
      dispatch({ type: ActionTypes.SET_FIRST_VISIT, payload: true });
    },
  };
  
  return (
    <AppContext.Provider value={{ state, actions }}>
      {children}
    </AppContext.Provider>
  );
}

// ==================== Hook ====================
export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
}

export default AppContext;
