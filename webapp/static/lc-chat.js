/* 상담 페이지 */

(function () {
  const a = LC.ACCENT, esc = LC.esc;

  const P = {
    tone: '쉬운 말',
    railTab: '전체',
    draft: '',
    thinking: false,
    extra: [],
    questions: [],   /* 최신 답변이 사용자에게 묻는 질문 — /chat final_response.questions */
    answering: ''    /* 사용자가 답하려고 고른 질문 */
  };

  let sse = null;

  /* ---------- 화면 ---------- */

  function side(){
    LC.MATTERS = [{ title: title(), kind: '상담', when: '오늘', live: true }];
    return `
<div style="display:flex;flex-direction:column;gap:18px;height:100%;min-height:0">
  <button class="lc-primary" data-act="newCase" style="display:flex;align-items:center;justify-content:center;gap:7px;padding:11px;border:0;border-radius:10px;background:${a};color:#fff;font-size:13.5px;font-weight:500;cursor:pointer">${LC.icon('plus', 14, '#fff')} 새 상담</button>
  ${LC.matterList()}
  <div style="display:flex;flex-direction:column;gap:3px">
    <div style="padding:0 8px 7px;font-size:10px;letter-spacing:.16em;color:#9aa0a5">도구</div>
    <a class="lc-ghost" href="documents.html" style="display:flex;align-items:center;gap:9px;width:100%;padding:9px 10px;border-radius:10px;font-size:12.5px;color:#3d4348;text-decoration:none">${LC.icon('search', 15, '#9aa0a5')}계약서 검토</a>
  </div>
  ${LC.helpCard()}
</div>`;
  }

  const SUGGESTIONS = ['전세 보증금을 못 돌려받고 있어요','부당해고를 당한 것 같아요','온라인 구매 환불을 거절당했어요'];

  function title(){
    const first = P.extra.find(m => m.role === 'user');
    return first ? first.text.split('\n')[0] : '새 상담';
  }

  function emptyState(){
    return `
      <div style="display:flex;flex-direction:column;align-items:center;gap:14px;padding:64px 0 24px;text-align:center">
        <span style="width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:11px;background:${a};color:#fff;font-family:'Noto Serif KR',serif;font-size:17px">法</span>
        <div style="font-family:'Noto Serif KR',serif;font-size:19px;font-weight:700">어떤 문제로 오셨나요?</div>
        <div style="max-width:440px;font-size:13px;line-height:1.75;color:#767d83;text-wrap:pretty">상황을 편하게 적어주시면 관련 법령과 판례를 찾아 근거와 함께 설명해 드립니다.</div>
        <div style="display:flex;gap:7px;flex-wrap:wrap;justify-content:center;padding-top:6px">
          ${SUGGESTIONS.map(p =>
            `<button class="lc-outline" data-act="prompt" data-key="${esc(p)}" style="padding:7px 13px;border:1px solid #e6e3dd;border-radius:999px;background:#faf9f6;font-size:12px;color:#3d4348;cursor:pointer">${esc(p)}</button>`).join('')}
        </div>
      </div>`;
  }

  function main(){
    return `
<div style="flex:none;padding:13px 30px 12px;background:#fff;border-bottom:1px solid #e6e3dd">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:20px">
    <div style="display:flex;align-items:baseline;gap:10px;min-width:0">
      <div style="font-family:'Noto Serif KR',serif;font-size:17px;font-weight:700;letter-spacing:-.01em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(title())}</div>
      <div style="flex:none;font-size:11.5px;color:#767d83;white-space:nowrap">자료 ${LC.DOCS.length}건</div>
    </div>
    <div style="display:flex;gap:3px;flex:none">
      ${['쉬운 말','법률 용어'].map(k => `<button class="lc-outline" data-act="tone" data-key="${esc(k)}" style="${LC.pill(P.tone === k)}">${esc(k)}</button>`).join('')}
    </div>
  </div>
</div>

<div style="flex:1;min-height:0;overflow:hidden;display:flex;flex-direction:column">
  <div id="thread" style="flex:1;min-height:0;overflow-y:auto;padding:26px 30px 12px">
    <div style="max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:20px">
      ${P.extra.length || P.thinking ? '' : emptyState()}

      ${P.extra.map(m => m.role === 'user'
        ? `<div style="display:flex;justify-content:flex-end"><div style="max-width:78%;padding:14px 18px;border-radius:16px 16px 5px 16px;background:#eeeae2;font-size:14px;line-height:1.75;white-space:pre-wrap;text-wrap:pretty">${esc(m.text)}</div></div>`
        : `<div style="display:flex;justify-content:flex-start"><div style="max-width:100%;padding:18px 22px;border-radius:5px 16px 16px 16px;border:1px solid #e6e3dd;background:#fff;box-shadow:0 1px 2px rgba(20,23,26,.04),0 10px 26px rgba(20,23,26,.05);font-size:14px;line-height:1.85;white-space:pre-wrap;text-wrap:pretty">${esc(m.text)}</div></div>`
      ).join('')}

      ${!P.thinking && P.questions.length ? `
      <div style="display:flex;flex-direction:column;gap:7px">
        <div style="font-size:11.5px;color:#767d83">아래 질문에 답하시면 사건에 맞춰 더 좁혀서 안내합니다</div>
        ${P.questions.map(q => `<button class="lc-outline" data-act="answer" data-key="${esc(q)}" style="text-align:left;padding:9px 13px;border:1px solid ${P.answering === q ? a : '#e6e3dd'};border-radius:11px;background:#fff;font-size:13px;line-height:1.6;color:#3d4348;cursor:pointer">${esc(q)}</button>`).join('')}
      </div>` : ''}

      ${P.thinking ? `
      <div style="display:flex;align-items:center;gap:10px;padding:14px 18px;border:1px solid #e6e3dd;border-radius:14px;background:#fff;font-size:12.5px;color:#767d83">
        <span style="font-family:ui-monospace,Menlo,monospace;font-size:11.5px">${esc(LC.S.traceLines[LC.S.traceLines.length - 1] || LC.TRACE_FALLBACK[0])}</span>
        <span style="animation:lpb 1s infinite">●</span>
      </div>` : ''}

    </div>
  </div>

  <div style="flex:none;padding:10px 30px 12px;background:#fff;border-top:1px solid #e6e3dd">
    <div style="max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:8px">
      ${P.answering ? `
      <div style="display:flex;align-items:center;gap:8px;padding:7px 12px;border-radius:10px;background:#f4f1ea;font-size:12px;color:#3d4348">
        <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">답하는 질문: ${esc(P.answering)}</span>
        <button class="lc-ghost" data-act="cancelAnswer" title="취소" style="border:0;background:transparent;color:#767d83;cursor:pointer">✕</button>
      </div>` : ''}
      <div style="display:flex;gap:10px;align-items:flex-end;padding:10px 10px 10px 18px;border:1px solid #ddd9d1;border-radius:18px;background:#fff;box-shadow:0 1px 2px rgba(20,23,26,.04),0 10px 26px rgba(20,23,26,.05)">
        <textarea id="userInput" rows="3" placeholder="${P.answering ? '답을 적어 주세요.' : '상황을 적어주시면 법령과 판례를 찾아 설명해 드립니다.'}" style="flex:1;min-width:0;padding:6px 0;border:0;background:transparent;font-size:14.5px;line-height:1.7;color:#14171a;resize:none;outline:none;overflow-y:hidden;height:auto">${esc(P.draft)}</textarea>
        <div style="display:flex;align-items:center;gap:8px;flex:none">
          <button class="lc-outline" data-act="upload" title="자료 첨부" style="width:34px;height:34px;display:flex;align-items:center;justify-content:center;border:1px solid #e6e3dd;border-radius:9px;background:#fff;color:#767d83;cursor:pointer">${LC.icon('plus', 15)}</button>
          <button class="lc-outline" data-act="voice" style="height:34px;display:flex;align-items:center;gap:6px;padding:0 12px;border:1px solid #e6e3dd;border-radius:9px;background:#fff;font-size:12px;color:#3d4348;white-space:nowrap;cursor:pointer">${LC.icon('mic', 14, '#767d83')} 말로 하기</button>
          <button class="lc-primary" id="sendButton" data-act="send" title="전송 (Enter)" style="width:40px;height:40px;display:flex;align-items:center;justify-content:center;border:0;border-radius:12px;background:${a};color:#fff;cursor:pointer">${LC.icon('send', 17, '#fff')}</button>
        </div>
      </div>
      <div style="font-size:11px;color:#9aa0a5;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">법령·판례를 근거로 답변합니다 · Enter 전송 · 중요한 결정 전에는 근거를 함께 확인하세요</div>
    </div>
  </div>
</div>
<input type="file" id="fileUpload" accept=".pdf,.png,.jpg,.jpeg,.docx,.txt" style="display:none">`;
  }

  function rail(){
    return LC.evidenceRail({
      tab: P.railTab,
      footer: `
<div style="flex:none;border-top:1px solid #f0ede7;padding:14px 18px;display:flex;flex-direction:column;gap:8px">
  <div style="display:flex;align-items:center;gap:6px">
    <button class="lc-outline" data-act="refreshKb" id="refreshKnowledgeButton" style="flex:1;padding:6px;border:1px solid #e6e3dd;border-radius:8px;background:#fff;font-size:11px;color:#767d83;cursor:pointer">지식베이스 새로고침</button>
    <button class="lc-outline" data-act="clearKb" id="clearKnowledgeButton" style="flex:1;padding:6px;border:1px solid #e6e3dd;border-radius:8px;background:#fff;font-size:11px;color:#767d83;cursor:pointer">비우기</button>
  </div>
  ${LC.traceBlock()}
</div>`
    });
  }

  /* ---------- 서버 ---------- */

  function sendMessage(){
    const typed = (P.draft || '').trim();
    if (!typed || P.thinking) return;
    /* 고른 질문과 답을 한 메시지로 묶는다. 서버는 "두 달이요"가 어느 질문의 답인지 이 문장으로 안다. */
    const text = P.answering ? `상담의 질문: ${P.answering}\n답변: ${typed}` : typed;

    P.extra.push({ role:'user', text });
    P.draft = '';
    P.questions = [];
    P.answering = '';
    P.thinking = true;
    LC.S.traceLines = [];
    LC.pinThread = true;
    LC.render();

    fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ message: text })
    }).then(() => {
      if (sse) sse.close();
      const qs = new URLSearchParams({ message: text });
      sse = new EventSource('/chat?' + qs.toString());

      sse.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'thought') { LC.S.traceLines.push(data.content); LC.render(); }
          else if (data.type === 'final_response') {
            P.extra.push({ role:'agent', text: data.chat_response });
            P.questions = data.questions || [];
            LC.SOURCES = (data.sources || []).map((h, i) => ({ id: String(i + 1), kind: h.kind || '내 문서', title: h.source, quote: h.quote, score: h.score, chunkId: h.chunk_id }));
            P.thinking = false; sse.close(); LC.pinThread = true; LC.render();
          }
          else if (data.type === 'error') {
            P.extra.push({ role:'agent', text:'오류: ' + data.content }); P.thinking = false; sse.close(); LC.pinThread = true; LC.render();
          }
          else if (data.type === 'stream_end') { P.thinking = false; sse.close(); LC.render(); }
        } catch (err) {
          console.error('SSE parse error:', err);
          P.extra.push({ role:'agent', text:'응답을 해석하는 중 오류가 발생했습니다.' });
          P.thinking = false; LC.pinThread = true; LC.render();
        }
      };

      sse.onerror = () => {
        P.extra.push({ role:'agent', text:'서버 연결이 끊어졌습니다.' });
        P.thinking = false; sse.close(); LC.pinThread = true; LC.render();
      };
    }).catch((err) => {
      console.error('Message send error:', err);
      P.extra.push({ role:'agent', text:'메시지를 보내는 중 오류가 발생했습니다.' });
      P.thinking = false; LC.pinThread = true; LC.render();
    });
  }

  async function fetchInitialState(){
    try {
      const r = await fetch('/get_initial_state');
      if (!r.ok) throw new Error(r.statusText);
      const st = await r.json();
      if (Array.isArray(st.chat_history)) {
        P.extra = st.chat_history.map(m => ({ role: m.role === 'user' ? 'user' : 'agent', text: m.content }));
      }
    } catch (err) {
      console.error('Initial state error:', err);
    }
    LC.pinThread = true;
    LC.render();
  }

  /* ---------- 등록 ---------- */

  LC.definePage({
    id: '상담',
    side, main, rail,
    afterRender: () => {
      const el = document.getElementById('userInput');
      if (el) { el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px'; }
    },
    acts: {
      tone: k => { P.tone = k; },
      railTab: k => { P.railTab = k; },
      prompt: k => { P.draft = k; setTimeout(() => { const el = document.getElementById('userInput'); if (el) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); } }, 0); },
      answer: k => { P.answering = k; setTimeout(() => { const el = document.getElementById('userInput'); if (el) el.focus(); }, 0); },
      cancelAnswer: () => { P.answering = ''; },
      easier: () => { P.tone = '쉬운 말'; P.draft = '위 내용을 더 쉽게 설명해 주세요.'; },
      newCase: () => {
        if (sse) { sse.close(); sse = null; }
        Object.assign(P, { extra: [], draft: '', thinking: false, railTab: '전체', questions: [], answering: '' });
        LC.S.traceLines = [];
        LC.SOURCES = [];
        LC.clearChatHistory();
        if (location.search) history.replaceState(null, '', location.pathname);
      },
      send: () => { sendMessage(); return false; },
      voice: () => { alert('음성 입력은 준비 중입니다.'); return false; },
      upload: () => { const f = document.getElementById('fileUpload'); if (f) f.click(); return false; }
    },
    init: () => {
      const ask = new URLSearchParams(location.search).get('ask');
      if (ask) P.draft = ask + ' 판례가 제 사건에도 적용될 수 있나요?';

      document.addEventListener('input', (e) => {
        if (e.target.id === 'userInput') {
          P.draft = e.target.value;
          e.target.style.height = 'auto';
          e.target.style.height = e.target.scrollHeight + 'px';
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.target.id === 'userInput' && e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });

      document.addEventListener('change', (e) => {
        if (e.target.id === 'fileUpload' && e.target.files.length) LC.uploadFile(e.target.files[0]);
      });

      fetchInitialState();
    }
  });
})();
