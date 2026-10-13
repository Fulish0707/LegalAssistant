/* 사건 자료 페이지 */

(function () {
  const a = LC.ACCENT, esc = LC.esc;

  const P = { filter: '전체', drag: false };
  const KINDS = ['전체','계약서','문서','이미지 · OCR','판례'];

  function side(){
    return `
<div style="display:flex;flex-direction:column;gap:16px">
  <button class="lc-primary" data-act="upload" style="display:flex;align-items:center;justify-content:center;gap:7px;padding:11px;border:0;border-radius:10px;background:${a};color:#fff;font-size:13.5px;font-weight:500;cursor:pointer">${LC.icon('plus', 14, '#fff')} 자료 올리기</button>
  <input type="file" id="fileUpload" accept=".pdf,.png,.jpg,.jpeg,.docx,.txt" style="display:none">
  <div style="display:flex;flex-direction:column;gap:5px">
    <div style="padding:0 8px 4px;font-size:10px;letter-spacing:.16em;color:#9aa0a5">분류</div>
    ${KINDS.map(k => {
      const on = P.filter === k;
      const cnt = k === '전체' ? LC.DOCS.length : LC.DOCS.filter(d => d.kind === k).length;
      return `<button class="lc-ghost" data-act="filter" data-key="${esc(k)}" style="${LC.chip(on)}"><span>${esc(k)}</span><span style="margin-left:auto;flex:none;font-family:ui-monospace,Menlo,monospace;font-size:10px;color:${on ? a : '#9aa0a5'}">${cnt}</span></button>`;
    }).join('')}
  </div>
  ${LC.matterList()}
</div>`;
  }

  function main(){
    const list = LC.DOCS.map((d, i) => ({ ...d, _i: i })).filter(d => P.filter === '전체' || d.kind === P.filter);
    return `
<div style="flex:1;min-height:0;overflow-y:auto;padding:24px 30px 30px">
  <div style="max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:16px">
    ${LC.pageHeader('사건 자료', '업로드한 계약서·판결문은 텍스트로 추출되어 답변 근거로 인용됩니다.')}
    <div style="border:1px solid #e6e3dd;border-radius:14px;background:#fff;overflow:hidden">
      <div style="display:grid;grid-template-columns:1fr 120px 90px 96px;gap:12px;padding:12px 20px;border-bottom:1px solid #f0ede7;background:#faf9f6;font-size:10px;letter-spacing:.14em;color:#9aa0a5">
        <span>파일</span><span>분류</span><span>색인</span><span style="text-align:right">상태</span>
      </div>
      ${list.map((d, i) => `
      <div style="display:grid;grid-template-columns:1fr 120px 90px 96px;gap:12px;align-items:center;padding:15px 20px;border-bottom:1px solid #f5f3ee;font-size:13px">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(d.name)}</span>
        <span style="font-size:12px;color:#767d83">${esc(d.kind)}</span>
        <span style="font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#767d83">${d.chunks} 청크</span>
        <span style="display:flex;align-items:center;justify-content:flex-end;gap:8px;font-size:11px;color:${d.state === '처리 중' ? a : d.state === '처리 실패' ? '#a4443d' : '#9aa0a5'}">
          ${d.state === '처리 중' ? '<span style="animation:lpb 1s infinite">●</span>' : ''}${esc(d.state)}
          <button class="lc-link" data-act="remove" data-key="${d._i}" title="지식베이스에서 삭제" style="border:0;background:transparent;padding:0 0 0 2px;font-size:13px;line-height:1;color:#c9c4bb;cursor:pointer">×</button>
        </span>
      </div>`).join('')}
      ${list.length ? '' : `<div style="padding:22px 20px;font-size:12.5px;color:#9aa0a5;text-align:center">이 분류에 해당하는 자료가 없습니다</div>`}
      <div id="dropZone" class="lc-ghost" data-act="upload" style="padding:26px;display:flex;flex-direction:column;align-items:center;gap:7px;color:#9aa0a5;cursor:pointer;background:${P.drag ? '#e6f0f0' : 'transparent'};border-top:1px dashed ${P.drag ? a : 'transparent'}">
        <span style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:10px;background:#f2efe9;color:#767d83">${LC.icon('upload', 17)}</span>
        <div style="font-size:13px">PDF · DOCX · 이미지를 끌어다 놓으세요</div>
        <div style="font-size:11px">스캔 문서는 OCR로 텍스트를 추출합니다</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 18px;border:1px solid #dbeae8;border-radius:12px;background:#f4faf9">
      <div style="font-size:12px;color:#3d4348">색인된 자료는 상담 답변의 근거로 인용됩니다.</div>
      <span style="display:flex;align-items:center;gap:6px;font-size:11px;color:#767d83">
        <span style="width:6px;height:6px;border-radius:50%;background:#3f9142;display:inline-block"></span>
        <span id="knowledgeSummary">${esc(LC.S.chunkLabel)}</span>
      </span>
    </div>
  </div>
</div>`;
  }

  LC.definePage({
    id: '사건 자료',
    side, main,
    acts: {
      filter: k => { P.filter = k; },
      remove: k => { LC.removeDoc(Number(k)); return false; },
      upload: () => { const f = document.getElementById('fileUpload'); if (f) f.click(); return false; },
      /* 상담을 고르면 그 상담을 열어 둔 채 상담 화면으로 간다 */
      pickMatter: k => {
        const c = LC.MATTERS[Number(k)];
        const open = !c || c.live ? Promise.resolve() : fetch('/conversations/select', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: c.id }) });
        open.catch(err => console.error('Conversation open error:', err)).then(() => { location.href = 'chatbot.html'; });
        return false;
      }
    },
    init: () => {
      fetch('/get_initial_state').then(r => r.json()).then(st => {
        LC.MATTERS = LC.consultations(st);
        LC.S.activeMatter = LC.MATTERS.findIndex(c => c.live);
        LC.render();
      }).catch(err => console.error('Initial state error:', err));
      document.addEventListener('change', (e) => {
        if (e.target.id === 'fileUpload' && e.target.files.length) LC.uploadFile(e.target.files[0]);
      });
      const inZone = e => e.target.closest && e.target.closest('#dropZone');
      document.addEventListener('dragover', e => { if (inZone(e)) { e.preventDefault(); if (!P.drag) { P.drag = true; LC.render(); } } });
      document.addEventListener('dragleave', e => { if (inZone(e) && P.drag) { P.drag = false; LC.render(); } });
      document.addEventListener('drop', e => {
        if (!inZone(e)) return;
        e.preventDefault(); P.drag = false;
        const f = e.dataTransfer && e.dataTransfer.files[0];
        if (f) LC.uploadFile(f); else LC.render();
      });
    }
  });
})();
