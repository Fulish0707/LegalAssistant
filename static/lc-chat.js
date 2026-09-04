/* 상담 페이지 */

(function () {
  const a = LC.ACCENT, esc = LC.esc;

  const P = {
    tone: '쉬운 말',
    term: null,
    activeCite: null,
    railTab: '전체',
    draft: '',
    thinking: false,
    vote: null,
    extra: []
  };

  let sse = null, draftTimer = null;

  /* ---------- 답변 카드 ---------- */

  function answerCard(){
    const easy = P.tone === '쉬운 말';
    const conf = 82;
    const verdict = easy
      ? "보증금은 돌려받을 수 있습니다. 다만 이사하기 전에 법원에 '임차권등기명령'을 먼저 신청하는 것이 가장 중요합니다."
      : '보증금 반환 청구가 가능한 사안입니다. 이사 전에 임차권등기명령을 먼저 신청해 대항력을 유지하는 것이 가장 중요합니다.';
    const facts = [
      {k:'드는 비용', v: easy ? '3만원 안팎' : '인지·송달료 약 3만원', note: easy ? '우편·법원 접수 비용' : '지급명령 기준'},
      {k:'걸리는 기간', v: easy ? '2~3개월' : '지급명령 2~3개월', note: easy ? '이의가 없을 때' : '이의 시 소송 전환'},
      {k:'돌려받을 가능성', v:'높음', note: easy ? '계약서 특약이 유리합니다' : '특약·확정일자 확보'}
    ];
    const riskTitle = easy ? '이사 예정일까지 9일 남았습니다' : '기한 주의 · 이사 예정일 D-9';
    const riskBody = easy
      ? '등기 신청 없이 이사하고 주민등록을 옮기면 보증금을 지킬 권리가 약해질 수 있습니다. 이사 전에 2단계를 먼저 끝내세요.'
      : '등기 없이 주민등록을 옮기면 대항력이 상실될 수 있습니다. 이사 전 임차권등기명령 신청과 등기 완료 여부를 반드시 확인하세요.';

    return `
<div style="display:flex;flex-direction:column;gap:0;border:1px solid #e6e3dd;border-radius:16px;background:#fff;box-shadow:0 1px 2px rgba(20,23,26,.04),0 12px 32px rgba(20,23,26,.05);overflow:hidden">
  <div style="display:flex;align-items:center;gap:10px;padding:14px 22px;border-bottom:1px solid #f0ede7">
    <span style="width:24px;height:24px;flex:none;display:flex;align-items:center;justify-content:center;border-radius:7px;background:${a};color:#fff;font-family:'Noto Serif KR',serif;font-size:11px">法</span>
    <span style="font-size:12.5px;font-weight:600">상담 답변</span>
    <span style="padding:2px 8px;border-radius:999px;background:#e6f0f0;color:${a};font-size:10.5px;font-weight:500">근거 ${LC.SOURCES.length}건 인용</span>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px">
      <span style="font-size:10.5px;color:#9aa0a5">근거 충분도</span>
      ${LC.bar(conf / 100, 56, 5)}
      <span style="font-family:ui-monospace,Menlo,monospace;font-size:10.5px;color:#767d83">${conf}%</span>
    </div>
  </div>

  <div style="padding:22px 22px 6px;display:flex;flex-direction:column;gap:8px">
    ${LC.sectionLabel('결론')}
    <div style="font-family:'Noto Serif KR',serif;font-size:17.5px;font-weight:600;line-height:1.65;letter-spacing:-.01em;text-wrap:pretty">${esc(verdict)}</div>
  </div>

  <div style="margin:16px 22px 0;display:grid;grid-template-columns:repeat(3,1fr);gap:1px;border:1px solid #e6e3dd;border-radius:12px;background:#e6e3dd;overflow:hidden">
    ${facts.map(f => `
    <div style="display:flex;flex-direction:column;gap:4px;padding:14px 16px;background:#faf9f6">
      <span style="font-size:10.5px;color:#9aa0a5">${esc(f.k)}</span>
      <span style="font-size:15px;font-weight:600;letter-spacing:-.01em">${esc(f.v)}</span>
      <span style="font-size:11px;color:#767d83">${esc(f.note)}</span>
    </div>`).join('')}
  </div>

  <div style="margin:14px 22px 0;display:flex;flex-direction:column;gap:7px">
    <div style="display:flex;gap:6px;flex-wrap:wrap">
      ${LC.TERMS.map(tm => {
        const on = P.term === tm.k;
        return `<button class="lc-outline" data-act="term" data-key="${esc(tm.k)}" style="padding:5px 11px;border-radius:999px;border:1px dashed ${on ? a : '#d8d4cc'};background:${on ? '#e6f0f0' : '#fff'};color:${on ? a : '#767d83'};font-size:11px;cursor:pointer">${esc(tm.k)} 뜻 보기</button>`;
      }).join('')}
    </div>
    ${P.term ? `
    <div style="display:flex;flex-direction:column;gap:4px;padding:13px 15px;border-radius:12px;background:#f4faf9;border:1px solid #dbeae8">
      <span style="font-size:12.5px;font-weight:600">${esc(P.term)}</span>
      <span style="font-size:12.5px;line-height:1.8;color:#3d4348;text-wrap:pretty">${esc((LC.TERMS.find(x => x.k === P.term) || {}).body || '')}</span>
    </div>` : ''}
  </div>

  <div style="margin:16px 22px 0;display:flex;gap:12px;padding:14px 16px;border-radius:12px;background:#fdf4e8;border:1px solid #f2e2c9">
    <span style="flex:none;color:#a8641a;padding-top:1px">${LC.icon('alert', 15)}</span>
    <div style="display:flex;flex-direction:column;gap:3px">
      <div style="font-size:12.5px;font-weight:600;color:#8a5215">${esc(riskTitle)}</div>
      <div style="font-size:12.5px;line-height:1.7;color:#8a5215;text-wrap:pretty">${esc(riskBody)}</div>
    </div>
  </div>

  <div style="padding:20px 22px 4px;display:flex;flex-direction:column;gap:14px">
    ${LC.sectionLabel('권장 절차')}
    ${LC.STEPS.map(st => `
    <div style="display:grid;grid-template-columns:26px 1fr;gap:12px;align-items:start">
      <span style="width:26px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:50%;border:1px solid ${a};color:${a};font-family:ui-monospace,Menlo,monospace;font-size:11.5px">${st.n}</span>
      <div style="display:flex;flex-direction:column;gap:7px;min-width:0;padding-bottom:14px;border-bottom:1px solid #f2efe9">
        <div style="display:flex;align-items:baseline;gap:9px;flex-wrap:wrap">
          <span style="font-size:14px;font-weight:600">${esc(easy ? st.easyTitle : st.title)}</span>
          <span style="padding:2px 8px;border-radius:999px;background:#f2efe9;font-size:10.5px;color:#767d83;white-space:nowrap">${esc(st.due)}</span>
        </div>
        <div style="font-size:13.5px;line-height:1.85;color:#3d4348;text-wrap:pretty">${esc(easy ? st.easy : st.body)}</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          ${st.cites.map(id => {
            const src = LC.SOURCES.find(x => x.id === id) || {};
            const on = P.activeCite === id;
            return `<button class="lc-outline" data-act="cite" data-key="${id}" style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 10px;border-radius:999px;border:1px solid ${on ? a : '#e6e3dd'};background:${on ? '#e6f0f0' : '#fff'};font-size:11px;color:${on ? a : '#767d83'};cursor:pointer">${esc((src.title || '').split(' (')[0])}</button>`;
          }).join('')}
        </div>
      </div>
    </div>`).join('')}
  </div>

  <div style="padding:12px 22px 20px;display:flex;flex-wrap:wrap;gap:8px">
    <button class="lc-primary" data-act="draftDoc" style="padding:9px 15px;border:0;border-radius:9px;background:${a};color:#fff;font-size:12.5px;font-weight:500;display:flex;align-items:center;gap:6px;cursor:pointer">${LC.icon('doc', 14, '#fff')} 내용증명 초안 만들기</button>
    <button class="lc-outline" data-act="easier" style="padding:9px 15px;border:1px solid #d8d4cc;border-radius:9px;background:#fff;font-size:12.5px;cursor:pointer">더 쉽게 설명해 주세요</button>
    <button class="lc-outline" data-act="print" style="padding:9px 15px;border:1px solid #d8d4cc;border-radius:9px;background:#fff;font-size:12.5px;display:flex;align-items:center;gap:6px;cursor:pointer">${LC.icon('print', 14, '#767d83')} PDF로 저장</button>
    <div style="margin-left:auto;display:flex;align-items:center;gap:6px">
      <button class="lc-outline" data-act="vote" data-key="up" title="도움이 됐어요" style="padding:5px 8px;border:1px solid ${P.vote === 'up' ? a : '#e6e3dd'};border-radius:7px;background:${P.vote === 'up' ? '#e6f0f0' : '#fff'};color:${P.vote === 'up' ? a : '#767d83'};cursor:pointer">${LC.icon('up', 13)}</button>
      <button class="lc-outline" data-act="vote" data-key="down" title="아쉬워요" style="padding:5px 8px;border:1px solid ${P.vote === 'down' ? a : '#e6e3dd'};border-radius:7px;background:${P.vote === 'down' ? '#e6f0f0' : '#fff'};color:${P.vote === 'down' ? a : '#767d83'};cursor:pointer">${LC.icon('down', 13)}</button>
    </div>
  </div>
</div>`;
  }

  /* ---------- 화면 ---------- */

  function side(){
    return `
<div style="display:flex;flex-direction:column;gap:18px;height:100%;min-height:0">
  <button class="lc-primary" data-act="newCase" style="display:flex;align-items:center;justify-content:center;gap:7px;padding:11px;border:0;border-radius:10px;background:${a};color:#fff;font-size:13.5px;font-weight:500;cursor:pointer">${LC.icon('plus', 14, '#fff')} 새 상담</button>
  ${LC.matterList()}
  <div style="display:flex;flex-direction:column;gap:3px">
    <div style="padding:0 8px 7px;font-size:10px;letter-spacing:.16em;color:#9aa0a5">도구</div>
    ${[{l:'내용증명 만들기', ic:'doc', act:'draftDoc'},{l:'계약서 검토', ic:'search', href:'documents.html'},{l:'용어 사전', ic:'book', act:'glossary'}].map(t =>
      t.href
        ? `<a class="lc-ghost" href="${t.href}" style="display:flex;align-items:center;gap:9px;width:100%;padding:9px 10px;border-radius:10px;font-size:12.5px;color:#3d4348;text-decoration:none">${LC.icon(t.ic, 15, '#9aa0a5')}${esc(t.l)}</a>`
        : `<button class="lc-ghost" data-act="${t.act}" style="display:flex;align-items:center;gap:9px;width:100%;text-align:left;padding:9px 10px;border:0;border-radius:10px;background:transparent;font-size:12.5px;color:#3d4348;cursor:pointer">${LC.icon(t.ic, 15, '#9aa0a5')}${esc(t.l)}</button>`
    ).join('')}
  </div>
  ${LC.helpCard()}
</div>`;
  }

  function main(){
    const timeline = [
      {name:'상담 접수', done:true},
      {name:'내용증명', now:true},
      {name:'임차권등기명령'},
      {name:'지급명령·소제기'}
    ];
    return `
<div style="flex:none;padding:13px 30px 12px;background:#fff;border-bottom:1px solid #e6e3dd;display:flex;flex-direction:column;gap:11px">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:20px">
    <div style="display:flex;align-items:baseline;gap:10px;min-width:0">
      <div style="font-family:'Noto Serif KR',serif;font-size:17px;font-weight:700;letter-spacing:-.01em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(LC.MATTERS[LC.S.activeMatter].title)}</div>
      <div style="display:flex;align-items:baseline;gap:8px;flex:none;font-size:11.5px;color:#767d83">
        <span style="white-space:nowrap">집·전세 문제</span><span>·</span><span style="white-space:nowrap">자료 ${LC.DOCS.length}건</span>
      </div>
    </div>
    <div style="display:flex;gap:3px;flex:none">
      ${['쉬운 말','법률 용어'].map(k => `<button class="lc-outline" data-act="tone" data-key="${esc(k)}" style="${LC.pill(P.tone === k)}">${esc(k)}</button>`).join('')}
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:0;overflow-x:auto">
    ${timeline.map((st, i) => `
    <div style="display:flex;align-items:center;gap:0;flex:none">
      <div style="display:flex;align-items:center;gap:7px;padding-right:10px">
        <span style="width:20px;height:20px;flex:none;display:flex;align-items:center;justify-content:center;border-radius:50%;font-size:10px;${st.done ? 'background:' + a + ';color:#fff;border:0' : st.now ? 'background:#fff;color:' + a + ';border:1.5px solid ' + a : 'background:#fff;color:#9aa0a5;border:1px solid #e0dcd4'}">${st.done ? LC.icon('check', 11, '#fff') : (i + 1)}</span>
        <span style="font-size:11.5px;white-space:nowrap;font-weight:${st.now ? '600' : '400'};color:${st.done || st.now ? '#14171a' : '#9aa0a5'}">${esc(st.name)}</span>
      </div>
      ${i === timeline.length - 1 ? '' : '<span style="display:block;width:34px;height:1px;background:#e0dcd4;margin-right:10px"></span>'}
    </div>`).join('')}
  </div>
</div>

<div style="flex:1;min-height:0;overflow:hidden;display:flex;flex-direction:column">
  <div id="thread" style="flex:1;min-height:0;overflow-y:auto;padding:26px 30px 12px">
    <div style="max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:20px">
      <div style="display:flex;justify-content:flex-end">
        <div style="max-width:78%;padding:14px 18px;border-radius:16px 16px 5px 16px;background:#eeeae2;font-size:14px;line-height:1.75;text-wrap:pretty">전세 계약이 지난달 만료됐는데 집주인이 보증금을 아직 안 돌려줍니다. 이사도 가야 하는데 어떻게 해야 하나요?</div>
      </div>

      ${answerCard()}

      ${P.extra.map(m => m.role === 'user'
        ? `<div style="display:flex;justify-content:flex-end"><div style="max-width:78%;padding:14px 18px;border-radius:16px 16px 5px 16px;background:#eeeae2;font-size:14px;line-height:1.75;white-space:pre-wrap;text-wrap:pretty">${esc(m.text)}</div></div>`
        : `<div style="display:flex;justify-content:flex-start"><div style="max-width:100%;padding:18px 22px;border-radius:5px 16px 16px 16px;border:1px solid #e6e3dd;background:#fff;box-shadow:0 1px 2px rgba(20,23,26,.04),0 10px 26px rgba(20,23,26,.05);font-size:14px;line-height:1.85;white-space:pre-wrap;text-wrap:pretty">${esc(m.text)}</div></div>`
      ).join('')}

      ${P.thinking ? `
      <div style="display:flex;align-items:center;gap:10px;padding:14px 18px;border:1px solid #e6e3dd;border-radius:14px;background:#fff;font-size:12.5px;color:#767d83">
        <span style="font-family:ui-monospace,Menlo,monospace;font-size:11.5px">${esc(LC.S.traceLines[LC.S.traceLines.length - 1] || LC.TRACE_FALLBACK[0])}</span>
        <span style="animation:lpb 1s infinite">●</span>
      </div>` : ''}

      <div style="display:flex;gap:7px;flex-wrap:wrap;padding-bottom:4px">
        ${['내용증명 문구 만들어 주세요','등기명령 절차가 궁금해요','소송 비용은 얼마나 드나요'].map(p =>
          `<button class="lc-outline" data-act="prompt" data-key="${esc(p)}" style="padding:7px 13px;border:1px solid #e6e3dd;border-radius:999px;background:#faf9f6;font-size:12px;color:#3d4348;cursor:pointer">${esc(p)}</button>`).join('')}
      </div>
    </div>
  </div>

  <div style="flex:none;padding:10px 30px 12px;background:#fff;border-top:1px solid #e6e3dd">
    <div style="max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:8px">
      <div style="display:flex;gap:10px;align-items:flex-end;padding:10px 10px 10px 18px;border:1px solid #ddd9d1;border-radius:18px;background:#fff;box-shadow:0 1px 2px rgba(20,23,26,.04),0 10px 26px rgba(20,23,26,.05)">
        <textarea id="userInput" rows="3" placeholder="상황을 적어주시면 법령과 판례를 찾아 설명해 드립니다." style="flex:1;min-width:0;padding:6px 0;border:0;background:transparent;font-size:14.5px;line-height:1.7;color:#14171a;resize:none;outline:none;overflow-y:hidden;height:auto">${esc(P.draft)}</textarea>
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
      activeCite: P.activeCite,
      footer: `
<div style="flex:none;border-top:1px solid #f0ede7;padding:14px 18px;display:flex;flex-direction:column;gap:8px">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
    <span style="font-size:12px;font-weight:600">작성 중인 문서</span>
    <span id="savingIndicator" style="font-size:10.5px;color:#9aa0a5">${esc(LC.S.saveState)}</span>
  </div>
  <textarea id="personalStatementDraft" placeholder="내용증명 등 서류 초안이 여기에 표시됩니다." style="min-height:96px;padding:11px 13px;border:1px solid #e6e3dd;border-radius:10px;background:#faf9f6;font-size:12.5px;line-height:1.75;color:#14171a;resize:vertical;outline:none">${esc(LC.S.statementDraft)}</textarea>
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
    const text = (P.draft || '').trim();
    if (!text || P.thinking) return;

    P.extra.push({ role:'user', text });
    P.draft = '';
    P.thinking = true;
    LC.S.traceLines = [];
    LC.pinThread = true;
    LC.render();

    let streamed = '', pending = null;
    const push = chunk => {
      if (!pending) { pending = { role:'agent', text:'' }; P.extra.push(pending); }
      pending.text += chunk;
    };

    fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ message: text, personal_statement_draft: LC.S.statementDraft || '' })
    }).then(() => {
      if (sse) sse.close();
      const qs = new URLSearchParams({ message: text, personal_statement_draft: LC.S.statementDraft || '' });
      sse = new EventSource('/chat?' + qs.toString());

      sse.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'token_chunk') { streamed += data.content; push(data.content); LC.pinThread = true; LC.render(); }
          else if (data.type === 'thought') { LC.S.traceLines.push(data.content); LC.render(); }
          else if (data.type === 'final_response') {
            if (streamed.trim() && pending) pending.text = data.chat_response || streamed;
            else P.extra.push({ role:'agent', text: data.chat_response });
            if (data.statement_suggestion) P.extra.push({ role:'agent', text:'서류 초안 제안:\n' + data.statement_suggestion });
            if (data.job_statement_suggestion) P.extra.push({ role:'agent', text:'신청서 제안:\n' + data.job_statement_suggestion });
            if (data.report_suggestion) P.extra.push({ role:'agent', text:'검토 의견 제안:\n' + data.report_suggestion });
            pending = null; streamed = ''; P.thinking = false; sse.close(); LC.pinThread = true; LC.render();
          }
          else if (data.type === 'update_draft') {
            LC.S.statementDraft = data.content; P.thinking = false; sse.close(); LC.render(); LC.saveDraft();
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
      LC.S.statementDraft = st.personal_statement_draft || '';
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
      term: k => { P.term = P.term === k ? null : k; },
      cite: k => { P.activeCite = P.activeCite === k ? null : k; P.railTab = '전체'; },
      railTab: k => { P.railTab = k; },
      prompt: k => { P.draft = k; setTimeout(() => { const el = document.getElementById('userInput'); if (el) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); } }, 0); },
      draftDoc: () => { P.draft = '내용증명 초안을 만들어 주세요.'; },
      easier: () => { P.tone = '쉬운 말'; P.draft = '위 내용을 더 쉽게 설명해 주세요.'; },
      glossary: () => { P.term = P.term ? null : LC.TERMS[0].k; },
      vote: k => { P.vote = P.vote === k ? null : k; },
      newCase: () => { if (sse) sse.close(); P.extra = []; P.draft = ''; P.thinking = false; LC.S.traceLines = []; LC.clearChatHistory(); },
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
        if (e.target.id === 'personalStatementDraft') {
          LC.S.statementDraft = e.target.value;
          LC.S.saveState = '저장 중...';
          document.getElementById('savingIndicator').textContent = LC.S.saveState;
          clearTimeout(draftTimer);
          draftTimer = setTimeout(LC.saveDraft, 800);
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
