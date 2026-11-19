const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function harness() {
  const listeners = {}, posts = [], streams = [];
  let page, ids = 0;
  const LC = {ACCENT: '#123', esc: String, S: {traceLines: []}, DOCS: [], SOURCES: [],
    render() {}, pill() {return '';}, icon() {return '';}, definePage(value) {page = value;}};
  const context = {LC, console, URLSearchParams, setTimeout,
    crypto: {randomUUID() {return `request-${++ids}`;}},
    location: {search: '', pathname: '/'},
    document: {getElementById() {return null;}, addEventListener(name, fn) {listeners[name] = fn;}},
    fetch(url, options) {
      if (options) posts.push(JSON.parse(options.body));
      return Promise.resolve({ok: true, json: async () => ({chat_history: [], conversations: [], sources: []})});
    },
    EventSource: class {constructor(url) {this.url = url; streams.push(this);} close() {}},
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/lc-chat.js'), 'utf8'), context);
  page.init();
  const type = text => listeners.input({target: {id: 'userInput', value: text, style: {}, scrollHeight: 40}});
  const change = (id, value) => listeners.change({target: {id, value, checked: value}});
  return {page, posts, streams, type, change, ids: () => ids};
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('default submit never enables pilot', async () => {
  const h = harness();
  assert.match(h.page.main(), /가상 계약 검토 파일럿/);
  assert.doesNotMatch(h.page.main(), /id="pilotOptIn"[^>]*checked/);
  h.type('기본 질문'); h.page.acts.send(); await settle();
  assert.deepEqual(h.posts[0], {message: '기본 질문'});
  assert.equal(h.ids(), 0);
  assert.equal(new URL(h.streams[0].url, 'http://local').searchParams.has('request_id'), false);
});

test('opt-in issue and identity survive stream retry, new message gets new id', async () => {
  const h = harness();
  h.change('pilotOptIn', true); h.change('pilotIssue', 'clause-explanation');
  h.type('특약 설명'); h.page.acts.send(); await settle();
  assert.equal(h.posts[0].pilot, true);
  assert.equal(h.posts[0].issue_id, 'clause-explanation');
  assert.equal(h.posts[0].request_id, 'request-1');
  const first = h.streams[0];
  assert.equal(new URL(first.url, 'http://local').searchParams.get('request_id'), 'request-1');
  first.onerror(); await settle();
  assert.equal(h.streams.length, 2);
  assert.equal(h.streams[1].url, first.url);
  assert.equal(h.ids(), 1);
  assert.equal(h.posts.length, 1);
  h.streams[1].onmessage({data: JSON.stringify({type: 'final_response', chat_response: '판단 보류', sources: [], questions: []})});
  h.type('다음 메시지'); h.page.acts.send(); await settle();
  assert.equal(h.posts[1].request_id, 'request-2');
});

test('pilot transport retry is bounded to one reconnect', async () => {
  const h = harness();
  h.change('pilotOptIn', true); h.type('파일럿 질문'); h.page.acts.send(); await settle();
  h.streams[0].onerror();
  h.streams[1].onerror();
  assert.equal(h.streams.length, 2);
  assert.equal(h.ids(), 1);
  assert.equal(h.posts.length, 1);
});
