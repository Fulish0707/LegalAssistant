/* 판례 검색 페이지 */

(function () {
  const a = LC.ACCENT, esc = LC.esc;

  const P = {
    query: '임대차 보증금 반환 지연',
    lawIdx: 0, joSel: null, hangSel: null, hoSel: null,
    sort: '유사도순',
    detail: null,
    applied: '',
    hits: [], hitsState: ''
  };

  /* 지식베이스(업로드 자료) 유사도 검색 */
  async function runSearch(){
    const q = (P.applied || '').trim();
    if (!q) { P.hits = []; P.hitsState = ''; return; }
    P.hitsState = '검색 중'; LC.render();
    try {
      P.hits = await LC.searchKnowledgeBase(q, 6);
      P.hitsState = '';
    } catch (err) {
      console.error('Knowledge base search error:', err);
      P.hits = []; P.hitsState = '검색 실패';
    }
    LC.render();
  }

  function hitsBlock(q){
    if (!q) return '';
    const rows = P.hits.map((h, i) => `
      <div style="display:flex;flex-direction:column;gap:8px;padding:14px 18px;border-bottom:1px solid #f5f3ee">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:14px">
          <span style="font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(h.source)}</span>
          <span style="display:flex;align-items:center;gap:8px;flex:none">${LC.bar(h.score, 52, 5)}<span style="font-family:ui-monospace,Menlo,monospace;font-size:11px;color:${a}">${LC.pct(h.score)}</span></span>
        </div>
        <div style="font-size:12.5px;line-height:1.75;color:#3d4348;text-wrap:pretty;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden">${esc(h.quote)}</div>
        <a class="lc-link" href="chatbot.html?ask=${encodeURIComponent(h.source)}" style="align-self:flex-start;font-size:11.5px;color:${a}">이 자료로 상담 이어가기</a>
      </div>`).join('');
    const empty = P.hitsState === '검색 중' ? '<span style="animation:lpb 1s infinite">●</span> 내 자료를 검색하고 있습니다'
      : P.hitsState === '검색 실패' ? '내 자료 검색에 실패했습니다'
      : '검색어와 관련된 업로드 자료가 없습니다';
    return `
    <div style="border:1px solid #dbeae8;border-radius:14px;background:#fff;overflow:hidden">
      <div style="display:flex;align-items:center;justify-content:space-between;padding:11px 18px;background:#f4faf9;font-size:10px;letter-spacing:.14em;color:#767d83">
        <span>내 자료에서 찾은 근거</span><span style="font-family:ui-monospace,Menlo,monospace">${P.hits.length}건</span>
      </div>
      ${rows || `<div style="padding:18px;font-size:12.5px;color:#9aa0a5;text-align:center">${empty}</div>`}
    </div>`;
  }

  function side(){
    const { law, count, selJo, selHang } = LC.drill(P);
    const rows = [
      { label:'법령', items: LC.LAWS.map((l, i) => ({ label:l.name, on:P.lawIdx === i, act:'law', key:String(i) })) },
      { label:'조', items: law.jangs.reduce((acc, j) => acc.concat(j.jos), []).map(jo => ({
          label: jo.no + ' ' + jo.title, count: count(jo.no) || 0, on: P.joSel === jo.no, act:'jo', key: jo.no })) }
    ];
    if (selJo) rows.push({ label:'항', items: selJo.hangs.map(h => ({ label:h.no, on:P.hangSel === h.no, act:'hang', key:h.no })) });
    if (selHang && selHang.hos.length) rows.push({ label:'호', items: selHang.hos.map(ho => ({ label:ho.no, on:P.hoSel === ho.no, act:'ho', key:ho.no })) });

    return `
<div style="display:flex;flex-direction:column;gap:16px">
  <div style="padding:0 8px;font-size:10px;letter-spacing:.16em;color:#9aa0a5">조문으로 좁히기</div>
  ${rows.map(row => `
  <div style="display:flex;flex-direction:column;gap:7px;padding:0 8px 14px;border-bottom:1px solid #f5f3ee">
    <span style="font-size:11px;color:#9aa0a5">${esc(row.label)}</span>
    <div style="display:flex;flex-direction:column;gap:4px">
      ${row.items.map(it => `
      <button class="lc-ghost" data-act="${it.act}" data-key="${esc(it.key)}" style="${LC.chip(it.on)}">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(it.label)}</span>
        ${it.count == null ? '' : `<span style="margin-left:auto;flex:none;font-family:ui-monospace,Menlo,monospace;font-size:10px;color:${it.on ? a : '#9aa0a5'}">${it.count}</span>`}
      </button>`).join('')}
    </div>
  </div>`).join('')}
  <button class="lc-link" data-act="resetDrill" style="align-self:flex-start;margin:0 8px;padding:0;border:0;background:transparent;font-size:11.5px;color:#9aa0a5;text-decoration:underline;cursor:pointer">선택 모두 해제</button>
</div>`;
  }

  function main(){
    const d = LC.drill(P);
    const { law, selJo, selJang, selHang, hos } = d;
    const q = (P.applied || '').trim();
    const pool = q && !P.joSel ? LC.CASES.slice().sort((x, y) => y.score - x.score) : d.list;
    const list = q ? pool.filter(c => (c.name + c.summary + c.holding + c.refs.map(r => r.law + r.jo).join(' ')).indexOf(q) > -1 || q.split(/\s+/).some(w => w && (c.summary + c.holding).indexOf(w) > -1)) : d.list;
    const crumbs = [{ label: law.name, sep:'›', act:'crumb', key:'law' }];
    if (selJang) crumbs.push({ label: selJang, sep:'›', act:'crumb', key:'law' });
    if (selJo) crumbs.push({ label: selJo.no + ' ' + selJo.title, sep: selHang ? '›' : '', act:'crumb', key:'jo' });
    if (selHang) crumbs.push({ label: selHang.no, sep: P.hoSel ? '›' : '', act:'crumb', key:'hang' });
    if (P.hoSel) crumbs.push({ label: P.hoSel, sep:'', act:'crumb', key:'ho' });

    return `
<div style="flex:1;min-height:0;overflow-y:auto;padding:24px 30px 30px">
  <div style="max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:16px">
    ${LC.pageHeader('판례 검색', '법령 → 조 → 항 → 호 순서로 범위를 좁히거나, 사건 내용을 문장으로 입력해 유사도로 검색합니다.')}

    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
      ${crumbs.map(c => `
      <div style="display:flex;align-items:center;gap:8px">
        <button class="lc-outline" data-act="${c.act}" data-key="${c.key}" style="padding:4px 10px;border:1px solid #e6e3dd;border-radius:999px;background:#fff;font-size:11.5px;color:#3d4348;cursor:pointer">${esc(c.label)}</button>
        <span style="font-size:11px;color:#9aa0a5">${esc(c.sep)}</span>
      </div>`).join('')}
    </div>

    <div style="display:flex;gap:10px;padding:11px 14px;border:1px solid #e6e3dd;border-radius:12px;background:#fff">
      ${LC.icon('search', 15, '#9aa0a5')}
      <input id="caseQuery" value="${esc(P.query)}" placeholder="예: 임대차 보증금 반환 지연" style="flex:1;min-width:0;border:0;background:transparent;font-size:13.5px;color:#14171a;outline:none">
      ${P.joSel ? `<span style="flex:none;align-self:center;font-size:10.5px;color:#9aa0a5;white-space:nowrap">선택한 조문 안에서</span>` : ''}
      ${P.applied ? `<button class="lc-link" data-act="clearQuery" style="flex:none;border:0;background:transparent;font-size:11.5px;color:#9aa0a5;cursor:pointer">지우기</button>` : ''}
      <button class="lc-primary" data-act="search" style="flex:none;padding:8px 16px;border:0;border-radius:9px;background:${a};color:#fff;font-size:12.5px;cursor:pointer">검색</button>
    </div>

    ${selJo ? `
    <div style="display:flex;flex-direction:column;gap:9px;padding:18px 20px;border:1px solid #e6e3dd;border-radius:14px;background:#fff">
      <div style="display:flex;align-items:baseline;gap:8px">
        <span style="font-family:'Noto Serif KR',serif;font-size:14.5px;font-weight:700">${esc(law.name + ' ' + selJo.no + (selHang && selHang.no !== '본문' ? ' ' + selHang.no : '') + (P.hoSel ? ' ' + P.hoSel : ''))}</span>
        <span style="font-size:12px;color:#767d83">(${esc(selJo.title)})</span>
      </div>
      <div style="font-family:'Noto Serif KR',serif;font-size:13.5px;line-height:1.95;text-wrap:pretty">${esc((selHang || selJo.hangs[0]).text)}</div>
      ${hos.length ? `
      <div style="display:flex;flex-direction:column;gap:5px;padding-top:9px;border-top:1px solid #f0ede7">
        ${hos.map(h => `<div style="display:flex;gap:8px;font-size:12.5px;line-height:1.8;color:#767d83"><span style="flex:none;font-weight:600;color:#14171a">${esc(h.no)}</span><span style="text-wrap:pretty">${esc(h.text)}</span></div>`).join('')}
      </div>` : ''}
    </div>` : ''}

    ${hitsBlock(q)}

    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap">
      <div style="font-size:12px;color:#767d83">판례 ${list.length}건${P.joSel ? ' · 조문 필터 적용' : ''}${q ? ' · "' + esc(q) + '"' : ''}${!P.joSel && !q ? ' · 전체' : ''}</div>
      <div style="display:flex;gap:5px">
        ${['유사도순','조문순','최신순'].map(k => `<button class="lc-outline" data-act="sort" data-key="${esc(k)}" style="${LC.sortPill(P.sort === k)}">${esc(k)}</button>`).join('')}
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:10px">
      ${list.map(c => `
      <button class="lc-card-btn" data-act="detail" data-key="${c.id}" style="display:flex;flex-direction:column;gap:10px;text-align:left;padding:18px 20px;border:1px solid ${P.detail === c.id ? a : '#e6e3dd'};border-radius:14px;background:#fff;cursor:pointer">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:14px">
          <span style="font-family:'Noto Serif KR',serif;font-size:15px;font-weight:600">${esc(c.name)}</span>
          <span style="display:flex;align-items:center;gap:8px;flex:none">
            ${LC.bar(c.score, 52, 5)}
            <span style="font-family:ui-monospace,Menlo,monospace;font-size:11px;color:${a}">${LC.pct(c.score)}</span>
          </span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          ${c.refs.map(r => `<span style="padding:3px 9px;border-radius:7px;background:#faf9f6;border:1px solid #eeeae2;font-family:ui-monospace,Menlo,monospace;font-size:10.5px;color:#767d83">${esc(r.law + ' ' + r.jo + (r.hang && r.hang !== '본문' ? ' ' + r.hang : '') + (r.ho ? ' ' + r.ho : ''))}</span>`).join('')}
        </div>
        <div style="font-size:13px;line-height:1.75;color:#3d4348;text-wrap:pretty">${esc(c.summary)}</div>
      </button>`).join('')}
      ${list.length ? '' : `
      <div style="display:flex;flex-direction:column;align-items:center;gap:9px;padding:40px;border:1px dashed #ddd9d1;border-radius:14px;color:#767d83">
        <div style="font-size:13px">${q ? '검색어와 조문 조건에 맞는 판례가 없습니다' : '선택한 조문에 연결된 판례가 없습니다'}</div>
        <button class="lc-outline" data-act="${q ? 'clearQuery' : 'widen'}" style="padding:8px 15px;border:1px solid #ddd9d1;border-radius:9px;background:#fff;font-size:12px;cursor:pointer">${q ? '검색어 지우기' : '한 단계 넓혀서 검색'}</button>
      </div>`}
    </div>
  </div>
</div>`;
  }

  function rail(){
    const det = LC.CASES.find(c => c.id === P.detail);
    return det ? LC.caseDetailRail(det) : '';
  }

  LC.definePage({
    id: '판례 검색',
    side, main, rail,
    acts: {
      law: k => { P.lawIdx = Number(k); P.joSel = P.hangSel = P.hoSel = null; },
      jo: k => { P.joSel = P.joSel === k ? null : k; P.hangSel = P.hoSel = null; },
      hang: k => { P.hangSel = P.hangSel === k ? null : k; P.hoSel = null; },
      ho: k => { P.hoSel = P.hoSel === k ? null : k; },
      sort: k => { P.sort = k; },
      resetDrill: () => { P.joSel = P.hangSel = P.hoSel = null; },
      widen: () => { if (P.hoSel) P.hoSel = null; else if (P.hangSel) P.hangSel = null; else P.joSel = null; },
      detail: k => { P.detail = P.detail === k ? null : k; },
      closeDetail: () => { P.detail = null; },
      search: () => { P.applied = P.query; P.detail = null; runSearch(); },
      clearQuery: () => { P.applied = ''; P.query = ''; P.hits = []; P.hitsState = ''; },
      crumb: k => {
        if (k === 'law') { P.joSel = P.hangSel = P.hoSel = null; }
        else if (k === 'jo') { P.hangSel = P.hoSel = null; }
        else if (k === 'hang') { P.hoSel = null; }
      }
    },
    init: () => {
      document.addEventListener('input', (e) => {
        if (e.target.id === 'caseQuery') P.query = e.target.value;
      });
      document.addEventListener('keydown', (e) => {
        if (e.target.id === 'caseQuery' && e.key === 'Enter') { e.preventDefault(); P.applied = P.query; P.detail = null; runSearch(); LC.render(); }
      });
    }
  });
})();
