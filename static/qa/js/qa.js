/* 问答页逻辑：SSE 流式渲染 + 引用弹窗 + 头像/思考动画 */
const chat = document.getElementById('chat');
const qInput = document.getElementById('q');
const btn = document.getElementById('btn');
let srcModal = null;
let currentSources = [];   // 本轮引用片段
let pendingImage = null;   // 用户上传、待随问题一起发送的图片 data URL

document.addEventListener('DOMContentLoaded', () => {
  srcModal = new bootstrap.Modal('#srcModal');
  btn.onclick = ask;
  qInput.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

  // 识图提问：上传图片后不自动填问题，而是把图片暂存，
  // 让用户输入自己的问题（如"这张图是什么药材"），提问时图片与问题一起发送。
  const imgBtn = document.getElementById('imgBtn');
  const imgFile = document.getElementById('imgFile');
  const ocrHint = document.getElementById('ocrHint');
  if (imgBtn && imgFile) {
    imgBtn.addEventListener('click', () => imgFile.click());
    imgFile.addEventListener('change', () => {
      const file = imgFile.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        pendingImage = reader.result;   // data:image/...;base64,...
        showImagePreview(pendingImage);
        ocrHint.innerHTML = '已附上图片，请输入你的问题（例如：这张图是什么药材？），再点提问。';
        qInput.focus();
      };
      reader.onerror = () => { ocrHint.textContent = '图片读取失败，请重试。'; };
      reader.readAsDataURL(file);
      imgFile.value = '';
    });
  }

  // 快捷问题：点击直接发问
  document.querySelectorAll('.suggestion').forEach(b => {
    b.addEventListener('click', () => {
      qInput.value = b.dataset.q;
      ask();
    });
  });

  // 主页/分享链接带 ?q= 时，预填并自动发问
  const prefill = new URLSearchParams(location.search).get('q');
  if (prefill) {
    qInput.value = prefill;
    ask();
  } else {
    greet();
  }
});

/** 在输入框上方显示待发送的图片缩略图（可点 × 移除） */
function showImagePreview(dataUrl) {
  clearImagePreview();
  const wrap = document.createElement('div');
  wrap.id = 'imgPreview';
  wrap.className = 'img-preview';
  wrap.innerHTML = '<img src="' + dataUrl + '" alt="待发送图片">' +
    '<button type="button" title="移除图片" aria-label="移除图片">×</button>';
  wrap.querySelector('button').onclick = () => {
    pendingImage = null;
    clearImagePreview();
    const hint = document.getElementById('ocrHint');
    if (hint) hint.textContent = '已移除图片。';
  };
  const inputGroup = document.querySelector('.card-footer .input-group');
  if (inputGroup) inputGroup.parentNode.insertBefore(wrap, inputGroup);
}

function clearImagePreview() {
  const old = document.getElementById('imgPreview');
  if (old) old.remove();
}

function greet() {
  addMsg('bot', '你好，我是「岐黄溯智」问答助手。可以问我方剂、药材或古籍出处，例如：治痞块用什么方');
}

function esc(s) {
  return (s ?? '').toString().replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const face = role === 'user'
    ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4.5 20c0-4 3.6-6.5 7.5-6.5s7.5 2.5 7.5 6.5"/></svg>'
    : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M12 21V8"/><path d="M12 12c-3-1-5-3-5-6 3 0 5 2 5 6z"/><path d="M12 14c3-1 5-3 5-6-3 0-5 2-5 6z"/></svg>';
  div.innerHTML = `<div class="avatar">${face}</div><div class="bubble"></div>`;
  div.querySelector('.bubble').textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function ask() {
  const question = qInput.value.trim();
  if (!question) return;

  // 首问后收起快捷问题
  const sg = document.getElementById('suggestions');
  if (sg) sg.style.display = 'none';

  qInput.value = '';
  btn.disabled = true;

  const sentImage = pendingImage;
  const userDiv = addMsg('user', question);
  if (sentImage) {
    const thumb = document.createElement('img');
    thumb.src = sentImage;
    thumb.className = 'user-img';
    thumb.alt = '附图';
    userDiv.querySelector('.bubble').appendChild(thumb);
  }
  pendingImage = null;
  clearImagePreview();
  // 机器人气泡：先显示「思考中」三点，收到首个 token 后切换为流式区
  const div = addMsg('bot', '');
  const bubble = div.querySelector('.bubble');
  bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  bubble.classList.add('cursor');
  const answerEl = document.createElement('span');
  let started = false;
  currentSources = [];

  try {
    const resp = await fetch('/api/qa/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, image: sentImage || null }),
    });
    if (!resp.ok || !resp.body) throw new Error('服务不可用（HTTP ' + resp.status + '）');

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (line.startsWith('data:')) handleEvent(JSON.parse(line.slice(5)), answerEl, bubble, () => { started = true; });
      }
    }
  } catch (e) {
    answerEl.textContent += '\n[请求失败] ' + e.message;
  } finally {
    bubble.classList.remove('cursor');
    renderCites(div, bubble);
    btn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}

function handleEvent(evt, answerEl, bubble, onStart) {
  if (evt.type === 'sources') {
    currentSources = evt.data || [];
  } else if (evt.type === 'token') {
    if (!answerEl.parentNode) { bubble.innerHTML = ''; bubble.appendChild(answerEl); if (onStart) onStart(); }
    answerEl.textContent += evt.data;
    chat.scrollTop = chat.scrollHeight;
  } else if (evt.type === 'out_of_scope') {
    // 意图识别拒绝：非中医药问题，用提示样式完整显示一次
    if (!answerEl.parentNode) { bubble.innerHTML = ''; bubble.appendChild(answerEl); if (onStart) onStart(); }
    bubble.classList.add('notice');
    answerEl.textContent = evt.data;
    chat.scrollTop = chat.scrollHeight;
  } else if (evt.type === 'error') {
    if (!answerEl.parentNode) { bubble.innerHTML = ''; bubble.appendChild(answerEl); }
    answerEl.textContent += '\n[出错] ' + evt.data;
    bubble.classList.remove('cursor');
  } else if (evt.type === 'done') {
    bubble.classList.remove('cursor');
  }
}

/** 回答末尾渲染引用列表，点击看原文 */
function renderCites(div, bubble) {
  if (!currentSources.length) return;
  const box = document.createElement('div');
  box.className = 'cites';
  box.innerHTML = `<div class="cite-title">引用来源（点击查看原文）</div>` +
    currentSources.map((c, i) =>
      `<span class="cite-chip" data-i="${i}">[${i + 1}] ${esc(c.title)} · ${esc(c.seg)}（${(c.score * 100).toFixed(0)}%）</span>`
    ).join('');
  box.querySelectorAll('.cite-chip').forEach(chip => {
    chip.onclick = () => showSource(currentSources[chip.dataset.i]);
  });
  bubble.appendChild(box);
}

function showSource(c) {
  if (!c) return;
  document.getElementById('srcTitle').textContent =
    `${c.title} · ${c.seg}（来源 ${c.source}，相似度 ${(c.score * 100).toFixed(0)}%）`;
  document.getElementById('srcBody').textContent = c.text || '（无正文）';
  srcModal.show();
}
