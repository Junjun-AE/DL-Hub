# -*- coding: utf-8 -*-
"""
DL-Hub 用户组件 v4 (终极兼容版)
================================
双重注入策略：
  1. head 参数注入 CSS+JS（Gradio 4+ 生效）
  2. gr.HTML() 用 <img onerror> 触发 JS（所有 Gradio 版本生效）
     因为 innerHTML 中 <script> 不执行，但 <img onerror> 会执行

用户信息来源：环境变量 DLHUB_USER_NAME / DLHUB_USER_NICKNAME
"""

import os
import json
import html as html_mod


def _get_user_info():
    username = os.environ.get('DLHUB_USER_NAME', '')
    nickname = os.environ.get('DLHUB_USER_NICKNAME', '')
    return username, nickname or username


def _build_widget_js(username, nickname, initial):
    """构建widget的完整JS代码（纯字符串，不含<script>标签）"""
    return f"""
(function(){{
  if(document.getElementById('dlhub-uw-mounted')) return;
  var d=document.createElement('div'); d.id='dlhub-uw-mounted';
  
  // 内联CSS
  var st=document.createElement('style');
  st.textContent='.dlhub-uw{{position:fixed;top:12px;right:16px;z-index:99999;font-family:system-ui,sans-serif}}.dlhub-uw-trigger{{width:40px;height:40px;border-radius:50%;cursor:pointer;border:2px solid rgba(124,58,237,.4);overflow:hidden;background:linear-gradient(135deg,#7c3aed,#c026d3);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:16px;transition:all .2s;box-shadow:0 2px 12px rgba(124,58,237,.3)}}.dlhub-uw-trigger:hover{{transform:scale(1.08);box-shadow:0 4px 20px rgba(124,58,237,.5)}}.dlhub-uw-trigger img{{width:100%;height:100%;object-fit:cover}}.dlhub-uw-menu{{position:absolute;top:50px;right:0;width:260px;background:#fff;border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.15);opacity:0;visibility:hidden;transform:translateY(-8px) scale(.96);transition:all .2s}}.dlhub-uw-menu.open{{opacity:1;visibility:visible;transform:translateY(0) scale(1)}}.dlhub-uw-hd{{padding:16px;border-bottom:1px solid #eee;display:flex;align-items:center;gap:12px}}.dlhub-uw-av{{width:44px;height:44px;border-radius:50%;overflow:hidden;background:linear-gradient(135deg,#7c3aed,#c026d3);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:18px;flex-shrink:0}}.dlhub-uw-av img{{width:100%;height:100%;object-fit:cover}}.dlhub-uw-nm{{font-weight:600;font-size:14px;color:#1a1a1a}}.dlhub-uw-un{{font-size:12px;color:#888;margin-top:2px}}.dlhub-uw-it{{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;cursor:pointer;font-size:13px;color:#444;width:100%;border:none;background:none;font-family:inherit;text-align:left}}.dlhub-uw-it:hover{{background:#f5f3ff;color:#7c3aed}}.dlhub-uw-it.dg{{color:#dc2626}}.dlhub-uw-it.dg:hover{{background:#fef2f2}}.dlhub-uw-sp{{height:1px;background:#eee;margin:4px 12px}}';
  document.head.appendChild(st);

  var un={json.dumps(username)};
  var nn={json.dumps(nickname)};
  var ini={json.dumps(initial)};
  var avt='';try{{avt=localStorage.getItem('dlhub_avatar_'+un)||'';}}catch(e){{}}

  d.innerHTML='<div class="dlhub-uw"><div class="dlhub-uw-trigger" id="uwT">'+(avt?'<img src="'+avt+'"/>':ini)+'</div><div class="dlhub-uw-menu" id="uwM"><div class="dlhub-uw-hd"><div class="dlhub-uw-av" id="uwA">'+(avt?'<img src="'+avt+'"/>':ini)+'</div><div><div class="dlhub-uw-nm">'+nn+'</div><div class="dlhub-uw-un">@'+un+'</div></div></div><div style="padding:8px"><button class="dlhub-uw-it" id="uwS">\\u{1F5BC} 设置头像</button><div class="dlhub-uw-sp"></div><button class="dlhub-uw-it" id="uwW">\\u{1F504} 切换用户</button><button class="dlhub-uw-it dg" id="uwL">\\u{1F6AA} 退出登录</button></div></div></div>';
  document.body.appendChild(d);

  document.getElementById('uwT').onclick=function(e){{e.stopPropagation();document.getElementById('uwM').classList.toggle('open')}};
  document.addEventListener('click',function(){{var m=document.getElementById('uwM');if(m)m.classList.remove('open')}});
  document.getElementById('uwM').onclick=function(e){{e.stopPropagation()}};

  function logout(){{
    // 1. 标记全局退出状态
    window.__dlhub_logging_out = true;
    // 2. 拦截fetch，阻止后续请求
    try{{ window.fetch = function(){{ return new Promise(function(){{}}); }}; }}catch(e){{}}
    // 3. 拦截XMLHttpRequest
    try{{ XMLHttpRequest.prototype.send = function(){{}}; }}catch(e){{}}
    // 4. 暴力清除所有定时器
    var maxId = setTimeout(function(){{}},0);
    for(var i=1;i<=maxId+200;i++){{ clearTimeout(i); clearInterval(i); }}
    // 5. 清除session
    try{{ localStorage.removeItem('dlhub_session'); }}catch(e){{}}
    // 6. 跳转到DL-Hub主界面的logout（端口7860）
    // Gradio子应用运行在不同端口，需要指定完整URL
    var dlhubPort = '7860';
    try{{
      // 尝试从环境推断DL-Hub端口：检查opener或parent
      var topHost = window.top.location.host;
      if(topHost){{ dlhubPort = topHost.split(':')[1] || '7860'; }}
    }}catch(e){{}}
    var logoutUrl = window.location.protocol + '//' + window.location.hostname + ':' + dlhubPort + '/logout';
    try{{ window.top.location.replace(logoutUrl); }}catch(e){{ window.location.href = logoutUrl; }}
  }}
  document.getElementById('uwL').onclick=logout;
  document.getElementById('uwW').onclick=logout;

  document.getElementById('uwS').onclick=function(){{
    document.getElementById('uwM').classList.remove('open');
    var inp=document.createElement('input');inp.type='file';inp.accept='image/*';
    inp.onchange=function(ev){{
      var f=ev.target.files[0];if(!f)return;
      var r=new FileReader();
      r.onload=function(e){{
        // 简单裁剪为正方形
        var img=new Image();
        img.onload=function(){{
          var sz=Math.min(img.width,img.height);
          var c=document.createElement('canvas');c.width=200;c.height=200;
          var cx=c.getContext('2d');
          cx.beginPath();cx.arc(100,100,100,0,Math.PI*2);cx.clip();
          cx.drawImage(img,(img.width-sz)/2,(img.height-sz)/2,sz,sz,0,0,200,200);
          var b=c.toDataURL('image/png',0.9);
          try{{localStorage.setItem('dlhub_avatar_'+un,b)}}catch(e){{}}
          document.getElementById('uwT').innerHTML='<img src="'+b+'"/>';
          document.getElementById('uwA').innerHTML='<img src="'+b+'"/>';
        }};
        img.src=e.target.result;
      }};
      r.readAsDataURL(f);
    }};
    inp.click();
  }};
}})();
"""


def get_widget_head():
    """CSS+JS 注入到 <head>（Gradio 4+ 支持）"""
    username, nickname = _get_user_info()
    if not username:
        return ""
    initial = nickname[0].upper() if nickname else "U"
    js_code = _build_widget_js(username, nickname, initial)
    return f"<script>{js_code}</script>"


def get_widget_html():
    """
    gr.HTML() 内容 — 用 <img onerror> 触发JS执行
    这是因为 innerHTML 中 <script> 不执行，但 <img onerror> 会执行！
    兼容所有 Gradio 版本。
    """
    username, nickname = _get_user_info()
    if not username:
        return ""
    initial = nickname[0].upper() if nickname else "U"
    js_code = _build_widget_js(username, nickname, initial)
    # 转义用于HTML attribute
    escaped = html_mod.escape(js_code, quote=True)
    return f'<img src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" onload="{escaped}" style="display:none" />'
