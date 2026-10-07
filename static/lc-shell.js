/* 법률 상담 데스크 — 공통 셸
   상단 바, 탭 스트립, 3열 골격, 글자 크기, 공통 스타일 헬퍼, 서버 호출.
   각 페이지는 LC.definePage({ id, side(), main(), rail(), acts, init }) 로 자기 화면만 정의합니다. */

window.LC = window.LC || {};

(function () {
  const a = LC.ACCENT;

  let csrfPromise;
  LC.secureFetch = async function (url, options = {}) {
    if (!csrfPromise) {
      csrfPromise = fetch('/get_initial_state', { credentials: 'same-origin', cache: 'no-store' })
        .then(async response => {
          if (!response.ok) throw new Error('세션을 확인하지 못했습니다.');
          const state = await response.json();
          if (!state.csrf_token) throw new Error('세션을 확인하지 못했습니다.');
          return state.csrf_token;
        }).catch(error => { csrfPromise = null; throw error; });
    }
    const headers = new Headers(options.headers || {});
    headers.set('X-CSRF-Token', await csrfPromise);
    return fetch(url, { ...options, headers, credentials: 'same-origin', cache: 'no-store' });
  };

  /* ---------- 공통 상태 ---------- */

  LC.S = {
    scale: localStorage.getItem('lc.scale') || '보통',
    chunkLabel: '지식베이스 확인 중',
    traceOpen: false,
    traceLines: [],
    activeMatter: 0
  };
  LC.pinThread = true;   // 첫 렌더와 새 메시지 후에는 스레드를 맨 아래로

  /* ---------- 헬퍼 ---------- */

  LC.esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  LC.pct = n => Math.round(n * 100) + '%';

  LC.pill = on =>
    'flex:none;white-space:nowrap;padding:3px 9px;border-radius:999px;border:1px solid ' + (on ? a : '#e6e3dd') +
    ';background:' + (on ? '#e6f0f0' : '#fff') + ';color:' + (on ? a : '#767d83') + ';font-size:10.5px;cursor:pointer';

  LC.chip = on =>
    'display:flex;align-items:center;gap:8px;width:100%;text-align:left;padding:8px 10px;border-radius:9px;border:0;background:' +
    (on ? '#e6f0f0' : 'transparent') + ';color:' + (on ? a : '#3d4348') + ';font-size:12.5px;font-weight:' + (on ? '500' : '400') + ';cursor:pointer';

  LC.sortPill = on =>
    'padding:6px 12px;border-radius:999px;border:1px solid ' + (on ? a : '#e6e3dd') + ';background:' +
    (on ? '#e6f0f0' : '#fff') + ';color:' + (on ? a : '#767d83') + ';font-size:11.5px;cursor:pointer';

  LC.bar = (score, w, h) =>
    '<span style="width:' + w + 'px;height:' + h + 'px;border-radius:3px;background:#eeeae2;overflow:hidden;display:block">' +
    '<span style="display:block;height:100%;width:' + LC.pct(score) + ';background:' + a + '"></span></span>';

  /* 선형 SVG 아이콘 — 이모지 대신 사용 */
  const ICONS = {
    plus:   '<path d="M12 5v14M5 12h14"/>',
    send:   '<path d="M12 19V5M5 12l7-7 7 7"/>',
    mic:    '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    check:  '<path d="M5 12l4 4L19 6"/>',
    alert:  '<path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
    up:     '<path d="M7 11l5-5 5 5M12 6v13"/>',
    down:   '<path d="M17 13l-5 5-5-5M12 18V5"/>',
    doc:    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
    book:   '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    upload: '<path d="M12 16V4M6 10l6-6 6 6"/><path d="M4 20h16"/>',
    print:  '<path d="M6 9V3h12v6M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="7"/>',
    chart:  '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'
  };
  LC.icon = (name, size, color) =>
    '<svg width="' + (size || 14) + '" height="' + (size || 14) + '" viewBox="0 0 24 24" fill="none" stroke="' + (color || 'currentColor') +
    '" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex:none;display:block">' + (ICONS[name] || '') + '</svg>';

  LC.sectionLabel = t => '<div style="font-size:10px;letter-spacing:.16em;color:#9aa0a5">' + LC.esc(t) + '</div>';

  LC.pageHeader = (title, note) => `
<div style="display:flex;flex-direction:column;gap:5px">
  <div style="font-family:'Noto Serif KR',serif;font-size:20px;font-weight:700">${LC.esc(title)}</div>
  <div style="font-size:12.5px;color:#767d83">${LC.esc(note)}</div>
</div>`;

  /* ---------- 사이드바 공통 블록 ---------- */

  LC.matterList = () => `
<div style="display:flex;flex-direction:column;gap:3px">
  ${LC.sectionLabel('진행 중').replace('font-size:10px', 'padding:0 8px 7px;font-size:10px')}
  ${LC.MATTERS.map((c, i) => `
  <button class="lc-ghost" data-act="pickMatter" data-key="${i}" style="display:flex;flex-direction:column;gap:3px;width:100%;text-align:left;padding:10px;border:0;border-radius:10px;background:${LC.S.activeMatter === i ? '#f2efe9' : 'transparent'};color:#14171a;cursor:pointer">
    <div style="display:flex;align-items:center;gap:7px;min-width:0">
      <span style="width:7px;height:7px;flex:none;border-radius:50%;background:${c.live ? a : '#d8d4cc'};display:inline-block"></span>
      <span style="font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${LC.esc(c.title)}</span>
    </div>
    <div style="display:flex;gap:6px;padding-left:14px;font-size:11px;color:#9aa0a5"><span>${LC.esc(c.kind)}</span><span>·</span><span>${LC.esc(c.when)}</span></div>
  </button>`).join('')}
</div>`;

  LC.helpCard = () => `
<div style="margin-top:auto;display:flex;flex-direction:column;gap:9px;padding:14px;border-radius:12px;background:#f4faf9;border:1px solid #dbeae8">
  <div style="font-size:12px;font-weight:600">혼자 하기 어려우신가요?</div>
  <div style="font-size:11.5px;line-height:1.65;color:#3d4348">서류 작성과 접수까지 단계별로 함께 진행합니다. 비용은 들지 않습니다.</div>
  <button class="lc-primary" data-act="guide" style="padding:9px;border:0;border-radius:9px;background:${a};color:#fff;font-size:12px;font-weight:500;cursor:pointer">단계별로 함께 하기</button>
  <button class="lc-outline" data-act="lawyer" style="padding:8px;border:1px solid #d8d4cc;border-radius:9px;background:#fff;font-size:11.5px;color:#3d4348;cursor:pointer">복잡한 사건은 변호사 연결</button>
</div>`;

  /* ---------- 셸 ---------- */

  function topBar(){
    return `
<div style="flex:none;height:52px;display:flex;align-items:center;gap:18px;padding:0 20px;background:#fff;border-bottom:1px solid #e6e3dd">
  <div style="display:flex;align-items:center;gap:9px;flex:none">
    <span style="width:24px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:7px;background:${a};color:#fff;font-family:'Noto Serif KR',serif;font-size:12px;font-weight:700">法</span>
    <span style="font-size:14.5px;font-weight:700;letter-spacing:-.01em">법률 상담 데스크</span>
    <span style="padding:2px 7px;border-radius:5px;background:#f2efe9;font-size:10px;letter-spacing:.06em;color:#767d83">BETA</span>
  </div>
  <a class="lc-outline" href="cases.html" style="flex:1;max-width:420px;min-width:0;display:flex;align-items:center;gap:8px;padding:7px 12px;border:1px solid #e6e3dd;border-radius:9px;background:#faf9f6;color:#767d83;font-size:12.5px;white-space:nowrap;overflow:hidden;text-decoration:none">
    ${LC.icon('search', 13, '#9aa0a5')}
    <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">사건, 법령, 판례 검색</span>
    <span style="margin-left:auto;font-family:ui-monospace,Menlo,monospace;font-size:10.5px;padding:1px 5px;border:1px solid #e6e3dd;border-radius:4px;background:#fff">⌘K</span>
  </a>
  <div style="margin-left:auto;flex:none;display:flex;align-items:center;gap:12px">
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:11px;color:#9aa0a5">글자 크기</span>
      <div style="display:flex;gap:3px">
        ${['작게','보통','크게'].map(k => `<button data-act="scale" data-key="${k}" style="flex:none;white-space:nowrap;padding:3px 8px;border-radius:7px;border:1px solid ${LC.S.scale === k ? a : '#e6e3dd'};background:${LC.S.scale === k ? '#e6f0f0' : '#fff'};color:${LC.S.scale === k ? a : '#767d83'};font-size:10.5px;cursor:pointer">${k}</button>`).join('')}
      </div>
    </div>
    <span style="padding:4px 10px;border-radius:999px;background:#e6f0f0;color:${a};font-size:11px;font-weight:500">무료 이용</span>
    <span style="width:28px;height:28px;border-radius:50%;background:#e8dcc8;display:flex;align-items:center;justify-content:center;font-size:11.5px;font-weight:600;color:#5a4c36">김</span>
  </div>
</div>`;
  }

  function tabStrip(){
    return `
<div style="flex:none;height:46px;display:flex;align-items:stretch;gap:0;padding:0 30px;background:#fff;border-bottom:1px solid #e6e3dd">
  ${LC.TABS.map(t => {
    const on = t.label === LC.page.id;
    const style = 'flex:none;width:112px;display:flex;align-items:center;justify-content:center;padding:0;border:0;border-bottom:2px solid ' +
      (on ? a : 'transparent') + ';background:transparent;color:' + (on ? '#14171a' : '#9aa0a5') +
      ';font-size:13px;font-weight:' + (on ? '600' : '400') + ';white-space:nowrap;cursor:pointer;text-decoration:none';
    return on
      ? `<span aria-current="page" style="${style}">${LC.esc(t.label)}</span>`
      : `<a class="lc-tab" href="${t.href}" style="${style}">${LC.esc(t.label)}</a>`;
  }).join('')}
</div>`;
  }

  /* ---------- 우측 근거 패널 (여러 페이지가 공유) ---------- */

  LC.evidenceRail = (opts) => {
    const o = opts || {};
    const kinds = ['전체'].concat([...new Set(LC.SOURCES.map(x => x.kind))]);
    const tab = kinds.includes(o.tab) ? o.tab : '전체';
    const list = LC.SOURCES.filter(x => tab === '전체' || x.kind === tab);
    return `
<aside style="min-height:0;width:356px;flex:none;overflow:hidden;background:#fff;border-left:1px solid #e6e3dd;display:flex;flex-direction:column">
  <div style="display:flex;flex-direction:column;height:100%;min-height:0">
    ${list.length ? `
    <div style="flex:none;padding:16px 20px 0;display:flex;flex-direction:column;gap:12px">
      <div style="display:flex;align-items:baseline;justify-content:space-between;gap:10px">
        <span style="font-size:13px;font-weight:600">왜 이렇게 답했나요</span>
        <span style="font-size:11px;color:#9aa0a5">근거 ${LC.SOURCES.length}건</span>
      </div>
      ${kinds.length > 2 ? `
      <div style="display:flex;gap:4px;border-bottom:1px solid #f0ede7">
        ${kinds.map(k => `<button class="lc-tab" data-act="railTab" data-key="${LC.esc(k)}" style="padding:8px 10px;border:0;border-bottom:2px solid ${tab === k ? a : 'transparent'};background:transparent;font-size:12px;font-weight:500;color:${tab === k ? '#14171a' : '#9aa0a5'};cursor:pointer">${LC.esc(k)}</button>`).join('')}
      </div>` : ''}
    </div>

    <div id="knowledgeSources" style="flex:1;min-height:0;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:9px">
      ${list.map(s => `
      <div style="display:flex;flex-direction:column;gap:7px;padding:14px 16px;border-radius:12px;border:1px solid #e6e3dd;background:#fff">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
          <span style="font-size:10px;letter-spacing:.14em;color:#9aa0a5">${LC.esc(s.kind)}</span>
          ${s.score == null ? '' : `<span style="display:flex;align-items:center;gap:7px">
            ${LC.bar(s.score, 40, 4)}
            <span style="font-family:ui-monospace,Menlo,monospace;font-size:10px;color:${a}">${LC.pct(s.score)}</span>
          </span>`}
        </div>
        <div style="font-size:13px;font-weight:500;line-height:1.55">${LC.esc(s.title)}</div>
        <div style="font-size:12px;line-height:1.7;color:#767d83;text-wrap:pretty;white-space:pre-wrap;${LC.S.openSource === s.id ? '' : 'display:-webkit-box;-webkit-line-clamp:6;-webkit-box-orient:vertical;overflow:hidden'}">${LC.esc(s.quote)}</div>
        <div style="display:flex;gap:10px;padding-top:2px">
          <button class="lc-link" data-act="viewSource" data-key="${LC.esc(s.id)}" style="background:transparent;border:0;padding:0;font-size:11px;color:${a};cursor:pointer">${s.kind === '내 문서' ? '원문 보기' : LC.S.openSource === s.id ? '접기' : '전문 보기'}</button>
        </div>
      </div>`).join('')}
    </div>` : '<div style="flex:1"></div>'}

    ${o.footer || ''}
  </div>
</aside>`;
  };

  LC.traceBlock = () => `
<button class="lc-link" data-act="toggleTrace" style="display:flex;align-items:center;justify-content:space-between;padding:0;border:0;background:transparent;font-size:10px;letter-spacing:.16em;color:#9aa0a5;cursor:pointer">
  <span>처리 과정</span><span>${LC.S.traceOpen ? '−' : '+'}</span>
</button>
${LC.S.traceOpen ? `
<div id="thoughtsLog" style="display:flex;flex-direction:column;gap:5px;max-height:124px;overflow-y:auto;padding:11px 13px;border-radius:10px;background:#faf9f6;border:1px solid #f0ede7">
  ${(LC.S.traceLines.length ? LC.S.traceLines : LC.TRACE_FALLBACK).map(l => `<div style="font-family:ui-monospace,Menlo,monospace;font-size:10.5px;line-height:1.65;color:#767d83">${LC.esc(l)}</div>`).join('')}
</div>` : ''}`;

  /* ---------- 공통 상호작용 스타일 (인라인으로 못 쓰는 hover/focus만) ---------- */
  (function injectCss(){
    if (document.getElementById('lc-css')) return;
    const st = document.createElement('style');
    st.id = 'lc-css';
    st.textContent = `
      [data-act]{transition:background-color .12s,border-color .12s,color .12s,opacity .12s,box-shadow .12s}
      button[data-act]:hover,a[data-act]:hover{filter:brightness(.97)}
      button[data-act]:active{transform:translateY(.5px)}
      button[data-act]:focus-visible,a:focus-visible,textarea:focus-visible,input:focus-visible{outline:2px solid ${a};outline-offset:2px}
      .lc-card-btn:hover{border-color:${a} !important;box-shadow:0 8px 24px rgba(20,23,26,.06)}
      .lc-ghost:hover{background:#faf9f6 !important}
      .lc-link:hover{color:${a} !important;text-decoration:underline}
      .lc-primary:hover{opacity:.9}
      .lc-outline:hover{border-color:${a} !important;color:${a} !important}
      .lc-tab:hover{color:#3d4348 !important}
      textarea::placeholder,input::placeholder{color:#9aa0a5}
    `;
    document.head.appendChild(st);
  })();

  /* ---------- 렌더 ---------- */

  LC.render = function () {
    const z = LC.S.scale === '작게' ? .92 : LC.S.scale === '크게' ? 1.14 : 1;
    const app = document.getElementById('app');
    const keep = (document.getElementById('thread') || {}).scrollTop;

    app.innerHTML = `
<div style="--accent:${a};zoom:${z};height:calc(100vh / ${z});overflow:auto">
<div style="height:100%;min-width:1212px;display:flex;flex-direction:column;background:#f7f5f1;color:#14171a;font-family:'Noto Sans KR',system-ui,sans-serif;overflow:hidden">
  ${topBar()}
  <div style="flex:1;min-height:0;display:grid;grid-template-columns:256px minmax(600px,1fr) auto;grid-template-rows:minmax(0,1fr);overflow:auto">
    <aside style="min-height:0;overflow-y:auto;background:#fff;border-right:1px solid #e6e3dd;padding:16px 12px 22px;display:flex;flex-direction:column;gap:18px">
      ${LC.page.side ? LC.page.side() : ''}
    </aside>
    <main style="min-height:0;min-width:0;overflow:hidden;display:flex;flex-direction:column">
      ${tabStrip()}
      ${LC.page.main()}
    </main>
    ${LC.page.rail ? (LC.page.rail() || '') : ''}
  </div>
</div>
</div>`;

    const thread = document.getElementById('thread');
    if (thread) {
      thread.scrollTop = (LC.pinThread || keep == null) ? thread.scrollHeight : keep;
      LC.pinThread = false;
    }
    if (LC.page.afterRender) LC.page.afterRender();
  };

  /* ---------- 공통 동작 ---------- */

  const SHELL_ACTS = {
    scale: k => { LC.S.scale = k; localStorage.setItem('lc.scale', k); },
    toggleTrace: () => { LC.S.traceOpen = !LC.S.traceOpen; },
    pickMatter: k => { LC.S.activeMatter = Number(k); },
    refreshKb: () => { LC.fetchKnowledgeBaseStatus(); return false; },
    clearKb: () => { LC.clearKnowledgeBase(); return false; },
    print: () => { window.print(); return false; },
    guide: () => { location.href = 'chatbot.html?ask=' + encodeURIComponent('단계별로 함께 진행해 주세요'); return false; },
    lawyer: () => { alert('변호사 연결은 준비 중입니다. 상담 기록을 바탕으로 적합한 변호사를 안내할 예정입니다.'); return false; },
    viewSource: k => {
      const s = LC.SOURCES.find(x => x.id === k);
      if (s && s.kind === '내 문서') { location.href = 'documents.html'; return false; }
      LC.S.openSource = LC.S.openSource === k ? null : k;   // 법령·판례는 제자리에서 전문 펼치기
    }
  };
  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); location.href = 'cases.html'; }
  });

  LC.definePage = function (page) {
    LC.page = page;
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-act]');
      if (!btn) return;
      const act = btn.dataset.act, key = btn.dataset.key;
      const fn = (page.acts && page.acts[act]) || SHELL_ACTS[act];
      if (!fn) return;
      if (fn(key) === false) return;
      LC.render();
    });
    document.addEventListener('DOMContentLoaded', () => {
      LC.render();
      if (page.init) page.init();
      LC.fetchKnowledgeBaseStatus();
    });
  };

  /* ---------- 서버 호출 ---------- */

  LC.docKind = name =>
    /\.(png|jpe?g)$/i.test(name) ? '이미지 · OCR'
    : /\.docx?$/i.test(name) ? '문서'
    : /\.txt$/i.test(name) ? '문서'
    : /판결|판례|선고/.test(name) ? '판례' : '계약서';

  /* 서버의 지식베이스 목록을 LC.DOCS 로 반영. 아직 처리 중인 업로드는 그대로 둔다 */
  LC.syncDocs = function (sources) {
    const pending = (LC.DOCS || []).filter(d => d.state === '처리 중' && !sources.some(s => s.source === d.name));
    LC.DOCS = sources.map(s => ({ name: s.source, kind: LC.docKind(s.source), chunks: s.chunk_count || 0, state: '색인 완료' })).concat(pending);
  };

  LC.fetchKnowledgeBaseStatus = async function () {
    try {
      const r = await fetch('/knowledge_base/status');
      if (!r.ok) throw new Error(r.statusText);
      const st = await r.json();
      const srcs = Array.isArray(st.sources) ? st.sources : [];
      LC.syncDocs(srcs);
      LC.S.chunkLabel = st.chunk_count
        ? st.chunk_count.toLocaleString() + ' 청크 · 자료 ' + srcs.length + '건'
        : '저장된 자료 없음';
      if (st.error) LC.S.chunkLabel = '상태 오류';
    } catch (err) {
      console.error('Knowledge base status error:', err);
      LC.S.chunkLabel = '지식베이스 확인 실패';
    }
    LC.render();
  };

  LC.clearKnowledgeBase = async function () {
    if (!confirm('지식베이스에 저장된 모든 자료를 삭제하시겠습니까?')) return;
    try {
      const r = await LC.secureFetch('/knowledge_base/clear', { method: 'POST' });
      const res = await r.json().catch(() => ({}));
      if (!r.ok || !res.ok) throw new Error(res.message || r.statusText);
      LC.DOCS = [];
    } catch (err) {
      console.error('Knowledge base clear error:', err);
    }
    LC.fetchKnowledgeBaseStatus();
  };

  LC.removeDoc = async function (idx) {
    const d = LC.DOCS[idx];
    if (!d) return;
    if (!confirm('"' + d.name + '" 자료와 색인을 지식베이스에서 삭제할까요?')) return;
    try {
      const r = await LC.secureFetch('/knowledge_base/source', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: d.name })
      });
      const res = await r.json().catch(() => ({}));
      if (!r.ok || !res.ok) throw new Error(res.message || r.statusText);
      LC.DOCS.splice(idx, 1);
    } catch (err) {
      console.error('Knowledge base source delete error:', err);
    }
    LC.fetchKnowledgeBaseStatus();
  };

  LC.searchKnowledgeBase = async function (query, k) {
    const r = await LC.secureFetch('/knowledge_base/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, k: k || 6 })
    });
    const res = await r.json().catch(() => ({}));
    if (!r.ok || !res.ok) throw new Error(res.error || r.statusText);
    return res.results || [];
  };

  LC.clearChatHistory = async function () {
    const response = await LC.secureFetch('/clear_chat_history_endpoint', { method: 'POST' });
    if (!response.ok) throw new Error('상담 기록을 삭제하지 못했습니다. 다시 시도해 주세요.');
  };

  /* 자료 업로드 — 사건 자료·상담 페이지가 공유 */
  LC.uploadFile = function (file, onDone) {
    const kind = /\.(png|jpe?g)$/i.test(file.name) ? '이미지 · OCR'
      : /\.docx?$/i.test(file.name) ? '문서' : '계약서';
    LC.DOCS = LC.DOCS.concat([{ name: file.name, kind: kind, chunks: 0, state: '처리 중' }]);
    LC.render();

    const fd = new FormData();
    fd.append('file', file);
    fd.append('analysis_mode', 'summary');
    fd.append('user_instructions', '');
    fd.append('question', '');
    fd.append('add_to_knowledge_base', 'true');

    LC.secureFetch('/process_document_ocr', { method: 'POST', body: fd })
      .then(r => r.json().catch(() => ({})))
      .then(res => {
        const kb = (res && res.knowledge_base) || {};
        const ok = res && !res.error && kb.ok !== false;
        const entry = LC.DOCS.find(d => d.name === file.name && d.state === '처리 중') || LC.DOCS[LC.DOCS.length - 1];
        entry.state = ok ? '색인 완료' : '처리 실패';
        entry.chunks = kb.chunk_count || 0;
        if (!ok) console.error('Upload failed:', (res && res.error) || kb.message);
        LC.render();
        LC.fetchKnowledgeBaseStatus();
        if (onDone) onDone(res);
      })
      .catch(err => {
        console.error('Upload error:', err);
        const entry = LC.DOCS.find(d => d.name === file.name && d.state === '처리 중');
        if (entry) entry.state = '처리 실패';
        LC.render();
      });
  };

  /* 조문 계층 필터 — 판례 검색 페이지가 사용, 다른 페이지에서도 재사용 가능 */
  LC.drill = function (st) {
    const law = LC.LAWS[st.lawIdx] || LC.LAWS[0];
    const joOrder = [];
    law.jangs.forEach(j => j.jos.forEach(jo => joOrder.push(jo.no)));
    const match = c => (c.refs || []).some(r => r.law === law.name &&
      (!st.joSel || r.jo === st.joSel) && (!st.hangSel || r.hang === st.hangSel) && (!st.hoSel || r.ho === st.hoSel));
    const count = joNo => LC.CASES.filter(c => (c.refs || []).some(r => r.law === law.name && r.jo === joNo)).length;

    let list = LC.CASES.filter(match);
    if (st.sort === '조문순') list = list.slice().sort((x, y) => {
      const k = c => Math.min.apply(null, c.refs.filter(r => r.law === law.name).map(r => joOrder.indexOf(r.jo)).concat([99]));
      return k(x) - k(y);
    });
    else if (st.sort === '최신순') list = list.slice().sort((x, y) => (y.name.match(/\d{4}/) || [0])[0] - (x.name.match(/\d{4}/) || [0])[0]);
    else list = list.slice().sort((x, y) => y.score - x.score);

    const selJo = law.jangs.reduce((acc, j) => acc || j.jos.find(jo => jo.no === st.joSel), null);
    const selJang = selJo ? (law.jangs.find(j => j.jos.indexOf(selJo) > -1) || {}).name : null;
    const selHang = selJo && st.hangSel ? selJo.hangs.find(h => h.no === st.hangSel) : null;
    const hos = selHang ? (st.hoSel ? selHang.hos.filter(h => h.no === st.hoSel) : selHang.hos) : [];
    return { law, list, count, selJo, selJang, selHang, hos };
  };

  /* 판례 상세 패널 — 판례 검색 페이지 */
  LC.caseDetailRail = function (det) {
    return `
<aside style="min-height:0;width:356px;flex:none;overflow:hidden;background:#fff;border-left:1px solid #e6e3dd;display:flex;flex-direction:column">
  <div style="display:flex;flex-direction:column;height:100%;min-height:0">
    <div style="flex:none;padding:16px 20px 14px;border-bottom:1px solid #f0ede7;display:flex;flex-direction:column;gap:8px">
      <button class="lc-link" data-act="closeDetail" style="align-self:flex-start;padding:0;border:0;background:transparent;font-size:11.5px;color:#9aa0a5;cursor:pointer">← 근거 목록</button>
      <div style="font-family:'Noto Serif KR',serif;font-size:16px;font-weight:700;line-height:1.45">${LC.esc(det.name)}</div>
      <div style="display:flex;align-items:center;gap:8px">
        ${LC.bar(det.score, 64, 5)}
        <span style="font-family:ui-monospace,Menlo,monospace;font-size:11px;color:${a}">유사도 ${LC.pct(det.score)}</span>
      </div>
    </div>
    <div style="flex:1;min-height:0;overflow-y:auto;padding:18px 20px;display:flex;flex-direction:column;gap:18px">
      <div style="display:flex;flex-direction:column;gap:6px">
        ${LC.sectionLabel('사건 요지')}
        <div style="font-size:13px;line-height:1.85;text-wrap:pretty">${LC.esc(det.summary)}</div>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px">
        ${LC.sectionLabel('판시 사항')}
        <div style="font-family:'Noto Serif KR',serif;font-size:13px;line-height:1.9;text-wrap:pretty">${LC.esc(det.holding)}</div>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px">
        ${LC.sectionLabel('내 사건과 다른 점')}
        <div style="padding:13px 15px;border-radius:12px;background:#fdf4e8;border:1px solid #f2e2c9;font-size:12.5px;line-height:1.8;color:#8a5215;text-wrap:pretty">${LC.esc(det.diff)}</div>
      </div>
    </div>
    <div style="flex:none;border-top:1px solid #f0ede7;padding:14px 18px">
      <a class="lc-primary" href="chatbot.html?ask=${encodeURIComponent(det.name)}" style="display:block;text-align:center;width:100%;padding:11px;border:0;border-radius:10px;background:${a};color:#fff;font-size:13px;font-weight:500;cursor:pointer;text-decoration:none">이 판례로 상담 이어가기</a>
    </div>
  </div>
</aside>`;
  };
})();
