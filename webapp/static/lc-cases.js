/* 판례 검색 페이지
   사건 내용으로 판례를 검색한다 (/case_search). 왼쪽 목록은 검색 결과의 판례들이 본문에서
   인용한 조문을 법령 → 조 → 항으로 묶은 것이고, 고르면 그 조문을 인용한 판례만 남는다. */

(function () {
  const a = LC.ACCENT, esc = LC.esc;

  const P = {
    query: '',               /* 검색창 입력 */
    applied: '',             /* 검색에 쓴 입력 */
    cases: [],               /* 검색 결과 — {title, subject, date, kind, quote, score, chunk_id, statutes: [{law, article, paragraph}]} */
    caseState: '', searched: false,
    law: null, art: null, para: null,   /* 왼쪽에서 고른 법령, 조, 항 */
    laws: {},                /* 조문 본문을 보여주려고 읽은 법령 — 이름 → {articles} | '불러오는 중' | '없음' */
    hits: [], hitsState: ''  /* 내 자료(업로드) 검색 결과 */
  };
  let caseSeq = 0;           /* 늦게 도착한 이전 검색 결과를 버린다 */

  /* 상담 탭에 다녀와도 검색과 선택이 남도록 이 브라우저 탭에 적어 둔다. 탭을 닫으면 사라진다.
     진행 중 표시와 조문 본문(법령 하나가 수십만 자일 수 있다)은 적지 않고 다시 구한다 */
  const KEPT = 'lc.caseSearch';
  const KEPT_FIELDS = ['query', 'applied', 'cases', 'searched', 'law', 'art', 'para', 'hits'];
  function keep(){
    try { sessionStorage.setItem(KEPT, JSON.stringify(Object.fromEntries(KEPT_FIELDS.map(k => [k, P[k]])))); } catch (err) { /* 저장소를 못 쓰면 유지하지 않을 뿐이다 */ }
  }
  function restore(){
    try { Object.assign(P, JSON.parse(sessionStorage.getItem(KEPT) || '{}')); } catch (err) { /* 읽지 못하면 빈 화면으로 시작한다 */ }
  }
  restore();   /* 첫 화면을 그리기 전에 읽는다. 그리는 순간 keep()이 지금 상태를 적기 때문이다 */

  /* ---------- 결과에서 조문 목록 만들기 ---------- */

  const cites = (c, law, art, para) => c.statutes.some(s =>
    s.law === law && (!art || s.article === art) && (!para || s.paragraph === para));

  /* 값별로 그 값을 인용한 판례 수. 한 판례가 같은 값을 여러 번 인용해도 한 번만 센다 */
  function counted(cases, valuesOf){
    const counts = new Map();
    cases.forEach(c => new Set(valuesOf(c)).forEach(v => counts.set(v, (counts.get(v) || 0) + 1)));
    return [...counts].map(([key, count]) => ({ key, count }));
  }
  const number = label => (label.match(/\d+/g) || []).map(Number);          /* '제3조의2' → [3, 2] */
  const byNumber = (x, y) => { const p = number(x.key), q = number(y.key); return (p[0] - q[0]) || ((p[1] || 0) - (q[1] || 0)); };

  function drill(){
    const laws = counted(P.cases, c => c.statutes.map(s => s.law)).sort((x, y) => y.count - x.count || x.key.localeCompare(y.key));
    const inLaw = P.law ? P.cases.filter(c => cites(c, P.law)) : [];
    const arts = counted(inLaw, c => c.statutes.filter(s => s.law === P.law).map(s => s.article)).sort(byNumber);
    const inArt = P.art ? inLaw.filter(c => cites(c, P.law, P.art)) : [];
    const paras = counted(inArt, c => c.statutes.filter(s => s.law === P.law && s.article === P.art && s.paragraph).map(s => s.paragraph)).sort(byNumber);
    const list = P.law ? P.cases.filter(c => cites(c, P.law, P.art, P.para)) : P.cases;
    return { laws, arts, paras, list };
  }

  /* ---------- 서버 호출 ---------- */

  async function ask(url, options){
    const r = await fetch(url, options);
    const res = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(res.error || r.statusText);
    return res;
  }

  async function searchCases(){
    const q = P.applied;
    const seq = ++caseSeq;
    Object.assign(P, { law: null, art: null, para: null, searched: false });
    if (!q) { Object.assign(P, { cases: [], caseState: '', searched: false }); LC.render(); return; }
    P.caseState = '검색 중'; LC.render();
    try {
      const res = await ask('/case_search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q }) });
      if (seq !== caseSeq) return;
      Object.assign(P, { cases: res.results || [], caseState: '', searched: true });
    } catch (err) {
      if (seq !== caseSeq) return;
      console.error('Case search error:', err);
      Object.assign(P, { cases: [], caseState: '검색 실패', searched: true });
    }
    LC.render();
  }

  /* 고른 조의 본문을 보여주려고 그 법령의 조문을 한 번 읽어 둔다 */
  async function loadLaw(name){
    if (P.laws[name]) return;
    P.laws[name] = '불러오는 중';
    try {
      P.laws[name] = await ask('/law?name=' + encodeURIComponent(name));
    } catch (err) {
      console.error('Law load error:', err);
      P.laws[name] = '없음';
    }
    LC.render();
  }

  /* 지식베이스(업로드 자료) 유사도 검색 */
  async function searchOwnDocuments(){
    const q = P.applied;
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

  function applyQuery(){
    P.applied = P.query.trim();
    searchOwnDocuments();
    searchCases();
  }

  /* ---------- 화면 ---------- */

  function hitsBlock(q){
    if (!q) return '';
    const rows = P.hits.map(h => `
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

  const note = text => `<div style="padding:0 8px;font-size:11.5px;line-height:1.7;color:#9aa0a5">${esc(text)}</div>`;

  function side(){
    const d = drill();
    const rows = [{ label: '법령', act: 'law', on: P.law, items: d.laws }];
    if (P.law) rows.push({ label: '조', act: 'article', on: P.art, items: d.arts });
    if (P.art && d.paras.length) rows.push({ label: '항', act: 'paragraph', on: P.para, items: d.paras });

    return `
<div style="display:flex;flex-direction:column;gap:16px">
  <div style="padding:0 8px;font-size:10px;letter-spacing:.16em;color:#9aa0a5">조문으로 좁히기</div>
  ${!P.searched ? note('판례를 검색하면 결과의 판례들이 인용한 조문이 여기에 나옵니다.')
    : !d.laws.length ? note('검색 결과의 판례에서 읽어 낸 조문이 없습니다.')
    : rows.map(row => `
  <div style="display:flex;flex-direction:column;gap:7px;padding:0 8px 14px;border-bottom:1px solid #f5f3ee">
    <span style="font-size:11px;color:#9aa0a5">${esc(row.label)}</span>
    <div style="display:flex;flex-direction:column;gap:4px;max-height:38vh;overflow-y:auto">
      ${row.items.map(it => `
      <button class="lc-ghost" data-act="${row.act}" data-key="${esc(it.key)}" title="${esc(it.key)}" style="${LC.chip(row.on === it.key)}">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(it.key)}</span>
        <span style="margin-left:auto;flex:none;font-family:ui-monospace,Menlo,monospace;font-size:10px;color:${row.on === it.key ? a : '#9aa0a5'}">${it.count}</span>
      </button>`).join('')}
    </div>
  </div>`).join('')}
  ${P.law ? `<button class="lc-link" data-act="resetDrill" style="align-self:flex-start;margin:0 8px;padding:0;border:0;background:transparent;font-size:11.5px;color:#9aa0a5;text-decoration:underline;cursor:pointer">선택 모두 해제</button>` : ''}
</div>`;
  }

  /* 고른 조의 본문. 색인의 현행 법령에서 읽으므로, 판결 당시의 조문과 다를 수 있다 */
  function articleCard(){
    if (!P.art) return '';
    const law = P.laws[P.law];
    const art = law && law.articles ? law.articles.find(x => x.article === P.art) : null;
    const body = art ? esc(art.text)
      : `<span style="color:#9aa0a5">${law === '불러오는 중' || !law ? '조문을 불러오는 중입니다' : '현행 법령에서 이 조문의 본문을 찾지 못했습니다'}</span>`;
    return `
    <div style="display:flex;flex-direction:column;gap:9px;padding:18px 20px;border:1px solid #e6e3dd;border-radius:14px;background:#fff">
      <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px">
        <div style="display:flex;align-items:baseline;gap:8px;min-width:0">
          <span style="font-family:'Noto Serif KR',serif;font-size:14.5px;font-weight:700">${esc(P.law + ' ' + P.art)}</span>
          ${art && art.title ? `<span style="font-size:12px;color:#767d83">(${esc(art.title)})</span>` : ''}
        </div>
        ${art ? `<button class="lc-link" data-act="viewLaw" data-key="${esc(art.chunk_id)}" style="flex:none;background:transparent;border:0;padding:0;font-size:11.5px;color:${a};cursor:pointer">법령 전문 보기</button>` : ''}
      </div>
      <div style="font-family:'Noto Serif KR',serif;font-size:13.5px;line-height:1.95;text-wrap:pretty;white-space:pre-wrap;max-height:36vh;overflow-y:auto">${body}</div>
      ${art ? `<div style="font-size:11px;color:#9aa0a5">현행 법령의 조문입니다. 판결 당시의 조문과 다를 수 있습니다.</div>` : ''}
    </div>`;
  }

  function caseCards(list){
    if (P.caseState === '검색 중') return `<div style="padding:40px;border:1px dashed #ddd9d1;border-radius:14px;font-size:13px;color:#767d83;text-align:center"><span style="animation:lpb 1s infinite">●</span> 판례를 검색하고 있습니다</div>`;
    if (!list.length) return `<div style="padding:40px;border:1px dashed #ddd9d1;border-radius:14px;font-size:13px;line-height:1.8;color:#767d83;text-align:center">${
      P.caseState === '검색 실패' ? '판례를 검색하지 못했습니다. 잠시 후 다시 시도해 주세요.'
      : P.searched ? '검색어에 맞는 판례를 찾지 못했습니다.'
      : '위 검색창에 사건 내용을 입력하세요. 결과가 나오면 왼쪽에서 조문으로 좁힐 수 있습니다.'}</div>`;
    return list.map(c => `
      <div style="display:flex;flex-direction:column;gap:9px;padding:18px 20px;border:1px solid #e6e3dd;border-radius:14px;background:#fff">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:14px">
          <span style="font-family:'Noto Serif KR',serif;font-size:15px;font-weight:600">${esc(c.title)}</span>
          ${c.score == null ? '' : `<span style="display:flex;align-items:center;gap:8px;flex:none">
            ${LC.bar(c.score, 52, 5)}
            <span style="font-family:ui-monospace,Menlo,monospace;font-size:11px;color:${a}">${LC.pct(c.score)}</span>
          </span>`}
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;font-size:11.5px;color:#9aa0a5">${[c.subject, c.date, c.kind].filter(Boolean).map(esc).join('<span>·</span>')}</div>
        ${c.statutes.length ? `
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          ${c.statutes.slice(0, 8).map(s => {
            const on = s.law === P.law && (!P.art || s.article === P.art) && (!P.para || s.paragraph === P.para);
            return `<span style="padding:3px 9px;border-radius:7px;background:${on ? '#e6f0f0' : '#faf9f6'};border:1px solid ${on ? '#c9dede' : '#eeeae2'};font-size:10.5px;color:${on ? a : '#767d83'}">${esc([s.law, s.article, s.paragraph].filter(Boolean).join(' '))}</span>`;
          }).join('')}
          ${c.statutes.length > 8 ? `<span style="padding:3px 2px;font-size:10.5px;color:#9aa0a5">외 ${c.statutes.length - 8}개</span>` : ''}
        </div>` : ''}
        <div style="font-size:13px;line-height:1.75;color:#3d4348;text-wrap:pretty;white-space:pre-wrap;display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden">${esc(c.quote)}</div>
        <div style="display:flex;gap:14px">
          <button class="lc-link" data-act="viewCase" data-key="${esc(c.chunk_id)}" style="background:transparent;border:0;padding:0;font-size:11.5px;color:${a};cursor:pointer">전문 보기</button>
          <a class="lc-link" href="chatbot.html?ask=${encodeURIComponent(c.title)}&case=${encodeURIComponent(c.chunk_id)}" style="font-size:11.5px;color:${a}">이 판례로 상담 이어가기</a>
        </div>
      </div>`).join('');
  }

  function main(){
    const { list } = drill();
    const q = P.applied;
    const crumbs = [[P.law, 'law'], [P.art, 'article'], [P.para, 'paragraph']].filter(c => c[0]);

    return `
<div style="flex:1;min-height:0;overflow-y:auto;padding:24px 30px 30px">
  <div style="max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:16px">
    ${LC.pageHeader('판례 검색', '사건 내용을 문장으로 입력해 판례를 찾습니다. 결과가 나오면 왼쪽에서 판례가 인용한 법령 → 조 → 항으로 좁힐 수 있습니다.')}

    ${crumbs.length ? `
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
      ${crumbs.map((c, i) => `
      <div style="display:flex;align-items:center;gap:8px">
        <button class="lc-outline" data-act="crumb" data-key="${c[1]}" style="padding:4px 10px;border:1px solid #e6e3dd;border-radius:999px;background:#fff;font-size:11.5px;color:#3d4348;cursor:pointer">${esc(c[0])}</button>
        <span style="font-size:11px;color:#9aa0a5">${i < crumbs.length - 1 ? '›' : ''}</span>
      </div>`).join('')}
    </div>` : ''}

    <div style="display:flex;gap:10px;padding:11px 14px;border:1px solid #e6e3dd;border-radius:12px;background:#fff">
      ${LC.icon('search', 15, '#9aa0a5')}
      <input id="caseQuery" value="${esc(P.query)}" placeholder="예: 임대차 보증금 반환 지연" style="flex:1;min-width:0;border:0;background:transparent;font-size:13.5px;color:#14171a;outline:none">
      ${q ? `<button class="lc-link" data-act="clearQuery" style="flex:none;border:0;background:transparent;font-size:11.5px;color:#9aa0a5;cursor:pointer">지우기</button>` : ''}
      <button class="lc-primary" data-act="search" style="flex:none;padding:8px 16px;border:0;border-radius:9px;background:${a};color:#fff;font-size:12.5px;cursor:pointer">검색</button>
    </div>

    ${articleCard()}

    ${hitsBlock(q)}

    ${P.searched && P.caseState !== '검색 실패' ? `
    <div style="display:flex;flex-direction:column;gap:4px">
      <div style="font-size:12px;color:#767d83">판례 ${list.length}건${P.law ? ` · 검색 결과 ${P.cases.length}건 중 ${esc(crumbs.map(c => c[0]).join(' '))}을(를) 인용한 판례` : ''}${q ? ' · "' + esc(q) + '"' : ''}</div>
      ${P.cases.length ? `<div style="font-size:11.5px;line-height:1.6;color:#9aa0a5">조문은 판결문 본문에서 읽어 낸 것입니다. 법령 이름 없이 적힌 조문은 바로 앞에 나온 법령의 조문으로 보았으므로, 틀린 경우가 있을 수 있습니다.</div>` : ''}
    </div>` : ''}

    <div style="display:flex;flex-direction:column;gap:10px">${caseCards(list)}</div>
  </div>
</div>`;
  }

  LC.definePage({
    id: '판례 검색',
    side, main,
    afterRender: keep,
    acts: {
      law: k => { Object.assign(P, { law: P.law === k ? null : k, art: null, para: null }); },
      article: k => { Object.assign(P, { art: P.art === k ? null : k, para: null }); if (P.art) loadLaw(P.law); },
      paragraph: k => { P.para = P.para === k ? null : k; },
      /* 경로의 한 칸을 누르면 그 아래 단계의 선택을 푼다 */
      crumb: k => { if (k === 'law') P.art = P.para = null; else if (k === 'article') P.para = null; },
      resetDrill: () => { Object.assign(P, { law: null, art: null, para: null }); },
      search: () => { applyQuery(); return false; },
      clearQuery: () => { Object.assign(P, { applied: '', query: '', hits: [], hitsState: '' }); searchCases(); return false; },
      viewCase: k => { const c = P.cases.find(x => x.chunk_id === k); if (c) LC.openReader({ title: c.title, kind: c.kind, chunkId: k }); return false; },
      viewLaw: k => { LC.openReader({ title: P.law + ' ' + P.art, kind: '법령', chunkId: k }); return false; }
    },
    init: () => {
      if (P.applied && !P.searched) { searchOwnDocuments(); searchCases(); }   /* 검색 도중에 탭을 떠났다 */
      else if (P.art) loadLaw(P.law);
      document.addEventListener('input', (e) => {
        if (e.target.id === 'caseQuery') { P.query = e.target.value; keep(); }
      });
      document.addEventListener('keydown', (e) => {
        if (e.target.id === 'caseQuery' && e.key === 'Enter') { e.preventDefault(); applyQuery(); }
      });
    }
  });
})();
