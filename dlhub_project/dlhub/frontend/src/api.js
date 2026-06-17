/**
 * DL-Hub API 客户端
 * 
 * 修复：
 * - 添加 listTasks 方法（与 getTasks 相同）
 * - importTask 支持 force 参数
 * - 正确处理非字符串错误
 * - 正确的 createTask 方法签名
 */

const API_BASE = '/api';

// 错误消息映射表
const ERROR_MESSAGES = {
  'Failed to fetch': '网络连接失败，请检查网络连接',
  'Network Error': '网络错误，请稍后重试',
  'NetworkError': '网络错误，请稍后重试',
  'Internal Server Error': '服务器内部错误',
};

// 错误码映射
const ERROR_CODES = {
  400: '请求参数错误',
  401: '未授权，请重新登录',
  403: '没有权限执行此操作',
  404: '请求的资源不存在',
  408: '请求超时，请稍后重试',
  422: '请求数据验证失败',
  429: '请求过于频繁，请稍后重试',
  500: '服务器内部错误',
  502: '服务器网关错误',
  503: '服务暂不可用',
  504: '服务器网关超时',
};

/**
 * 将后端错误消息转换为用户友好的提示
 */
function translateError(message) {
  if (!message) return '未知错误，请稍后重试';
  
  // 确保 message 是字符串
  if (typeof message !== 'string') {
    if (typeof message === 'object') {
      if (message.detail) return translateError(message.detail);
      if (message.message) return translateError(message.message);
      if (message.msg) return translateError(message.msg);
      if (Array.isArray(message)) {
        const errors = message.map(e => {
          if (typeof e === 'object' && e.msg) {
            const loc = e.loc ? e.loc.slice(1).join('.') : '';
            return loc ? `${loc}: ${e.msg}` : e.msg;
          }
          return String(e);
        });
        return errors.join('; ');
      }
      try {
        return JSON.stringify(message);
      } catch {
        return String(message);
      }
    }
    return String(message);
  }
  
  // 特殊处理：保留原始格式供前端解析
  if (message.startsWith('APP_RUNNING:') || message.startsWith('TASK_EXISTS:') || message.startsWith('NEED_CONDA_ENV:')) {
    return message;
  }
  
  // 直接匹配
  if (ERROR_MESSAGES[message]) {
    return ERROR_MESSAGES[message];
  }
  
  // 部分匹配
  for (const [key, value] of Object.entries(ERROR_MESSAGES)) {
    if (message.includes(key)) {
      if (message.length > key.length + 10) {
        return `${value}: ${message.replace(key, '').trim()}`;
      }
      return value;
    }
  }
  
  // 中文消息直接返回
  if (/[\u4e00-\u9fa5]/.test(message)) {
    return message;
  }
  
  return message;
}

class ApiClient {
  // 防止多个并发401同时触发多次跳转
  _redirecting = false;

  // 跟踪所有正在进行的请求，以便在logout时取消它们
  _activeControllers = new Set();

  _handleUnauthorized() {
    if (this._redirecting) return;
    this._redirecting = true;

    // 1. 立即取消所有正在飞行的请求（最重要！阻止更多401响应回来）
    for (const controller of this._activeControllers) {
      try { controller.abort(); } catch(e) {}
    }
    this._activeControllers.clear();

    // 2. 拦截fetch，阻止所有后续API请求
    try {
      window.fetch = () => new Promise(() => {});
    } catch(e) {}

    // 3. 暴力清除所有setInterval/setTimeout
    try {
      const maxId = setTimeout(() => {}, 0);
      for (let i = 1; i <= maxId + 200; i++) { clearTimeout(i); clearInterval(i); }
    } catch(e) {}

    // 4. 标记全局退出状态（供其他非ApiClient的代码检测）
    window.__dlhub_logging_out = true;

    // 5. 清除前端session
    try { localStorage.removeItem('dlhub_session'); } catch(e) {}

    // 6. 跳转到登录页（通过/logout清除cookie）
    window.location.href = '/logout';
  }

  async request(method, path, data = null) {
    // 如果已在跳转中或已标记退出，直接抛出，不再发请求
    if (this._redirecting || window.__dlhub_logging_out) {
      throw new Error('会话已过期，正在跳转登录页...');
    }

    const controller = new AbortController();
    this._activeControllers.add(controller);

    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
    };
    
    if (data) {
      options.body = JSON.stringify(data);
    }
    
    try {
      const response = await fetch(`${API_BASE}${path}`, options);
      
      // 401: 会话过期，自动跳转登录页
      if (response.status === 401) {
        this._handleUnauthorized();
        throw new Error('会话已过期，正在跳转登录页...');
      }
      
      if (!response.ok) {
        let errorMessage;
        try {
          const error = await response.json();
          errorMessage = error.detail || error.message || ERROR_CODES[response.status];
        } catch {
          errorMessage = ERROR_CODES[response.status] || response.statusText;
        }
        throw new Error(translateError(errorMessage));
      }
      
      return response.json();
    } catch (error) {
      // 请求被abort时不要报错（这是正常的logout流程）
      if (error.name === 'AbortError') {
        throw new Error('请求已取消');
      }
      if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
        throw new Error(ERROR_MESSAGES['Failed to fetch']);
      }
      throw error;
    } finally {
      this._activeControllers.delete(controller);
    }
  }
  
  // ========== 目录浏览 ==========
  
  async browseDirectory(path = '') {
    return this.request('GET', `/workspace/browse?path=${encodeURIComponent(path)}`);
  }
  
  async validateDirectory(path) {
    return this.request('POST', '/workspace/validate', { path });
  }
  
  async getDiskSpace(path) {
    return this.request('GET', `/workspace/disk-space?path=${encodeURIComponent(path)}`);
  }
  
  // 创建文件夹（问题-4修复）
  async createDirectory(parentPath, name) {
    return this.request('POST', `/workspace/mkdir?parent_path=${encodeURIComponent(parentPath)}&name=${encodeURIComponent(name)}`);
  }
  
  async setWorkspace(path) {
    return { success: true, workspace: path };
  }
  
  // ========== 任务管理 ==========
  
  // 获取任务列表
  async listTasks(taskType = null) {
    const query = taskType ? `?task_type=${taskType}` : '';
    return this.request('GET', `/tasks${query}`);
  }
  
  // 别名
  async getTasks(taskType = null) {
    return this.listTasks(taskType);
  }
  
  // 创建任务
  async createTask(taskType, taskName, workDir, condaEnv, description = '') {
    return this.request('POST', '/tasks', {
      task_type: taskType,
      task_name: taskName,
      work_dir: workDir,
      conda_env: condaEnv,
      description
    });
  }
  
  // 导入任务（支持 force 参数）
  async importTask(taskDir, taskType, condaEnv = null, force = false) {
    return this.request('POST', '/tasks/import', {
      task_dir: taskDir,
      task_type: taskType,
      conda_env: condaEnv,
      force: force
    });
  }
  
  async getTask(taskId) {
    return this.request('GET', `/tasks/${taskId}`);
  }
  
  async deleteTask(taskId) {
    return this.request('DELETE', `/tasks/${taskId}`);
  }
  
  async updateTask(taskId, updates) {
    return this.request('PUT', `/tasks/${taskId}`, updates);
  }
  
  async getTaskStatus(taskId) {
    return this.request('GET', `/tasks/${taskId}/status`);
  }
  
  async copyTask(taskId, newName = null) {
    const data = newName ? { new_name: newName } : {};
    return this.request('POST', `/tasks/${taskId}/copy`, data);
  }
  
  // 获取任务磁盘占用
  async getTaskSize(taskId) {
    return this.request('GET', `/tasks/${taskId}/size`);
  }
  
  // ========== 应用管理 ==========
  
  async launchApp(taskId, force = false) {
    const query = force ? '?force=true' : '';
    return this.request('POST', `/app/launch/${taskId}${query}`);
  }
  
  async stopApp() {
    return this.request('POST', '/app/stop');
  }
  
  async getAppStatus() {
    return this.request('GET', '/app/status');
  }
  
  async getAppLogs(lines = 100) {
    return this.request('GET', `/app/logs?lines=${lines}`);
  }
  
  async restartApp(taskId) {
    return this.request('POST', `/app/restart/${taskId}`);
  }
  
  // ========== 系统信息 ==========
  
  async getCondaEnvs() {
    return this.request('GET', '/system/conda/envs');
  }
  
  // 扫描指定目录下的Conda环境
  async scanCondaDirectory(path) {
    return this.request('POST', `/system/conda/scan?path=${encodeURIComponent(path)}`);
  }
  
  async getGpuInfo() {
    return this.request('GET', '/system/gpu');
  }
  
  async getSystemInfo() {
    return this.request('GET', '/system/info');
  }
  
  // [新增] 获取系统资源实时状态
  async getSystemResources() {
    return this.request('GET', '/system/resources');
  }
  
  // [新增] 获取训练进度
  async getTrainingProgress(taskId) {
    return this.request('GET', `/tasks/${taskId}/progress`);
  }
}

const api = new ApiClient();
export default api;
