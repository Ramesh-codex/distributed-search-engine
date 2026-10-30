"use strict";
// Lightweight DOM interaction tests for the inline UI script.
// No npm packages, browser downloads, or external services are required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(
  path.resolve(__dirname, "../src/search/distribution/static/index.html"), "utf8"
);
const match = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(match, "Missing inline UI JavaScript");

class Element {
  constructor(tagName = "DIV") {
    this.tagName = tagName;
    this.children = [];
    this.listeners = {};
    this.dataset = {};
    this.className = "";
    this.attributes = {};
    this._text = "";
    this.value = "";
    this.disabled = false;
    this.isContentEditable = false;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() {
    return this._text + this.children.map((x) => x.textContent).join("");
  }
  append(...nodes) {
    this.children.push(...nodes);
  }
  appendChild(node) { this.append(node); }
  replaceChildren(...nodes) { this._text = ""; this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  focus() { document.activeElement = this; }
  blur() { document.activeElement = null; }
  select() { this.selected = true; }
}

const nodes = {};
const document = {
  activeElement: null,
  listeners: {},
  getElementById(id) { return nodes[id] ||= new Element(); },
  createElement(tag) { return new Element(tag.toUpperCase()); },
  createTextNode(value) { const node = new Element("#TEXT"); node.textContent = value; return node; },
  querySelectorAll(selector) {
    assert.equal(selector, "[data-query]");
    return chips;
  },
  addEventListener(name, handler) { this.listeners[name] = handler; }
};
const chips = ["circuit breaker", "BM25 ranking", "vector search", "distributed systems"]
  .map((query) => {
    const item = new Element("BUTTON");
    item.dataset.query = query;
    return item;
  });
const shardUrls = [0, 1, 2].map((i) => "http://127.0.0.1:" + (18080 + i));
const requests = [];
const fetchMock = async function(url) {
  const pathname = String(url);
  requests.push(pathname);
  if (pathname === "/stats") {
    return {
      ok: true,
      json: async () => ({
        shard_urls: shardUrls,
        breakers: Object.fromEntries(shardUrls.map((url) =>
          [url, {state: "closed", consecutive_failures: 0}]))
      })
    };
  }
  assert.ok(pathname.startsWith("/search?"));
  const query = new URLSearchParams(pathname.split("?")[1]).get("q");
  const titles = {
    "circuit breaker": "Circuit Breaker Pattern",
    "BM25 ranking": "BM25 Relevance Ranking",
    "vector search": "Vector Similarity Search"
  };
  const results = titles[query] ? [{
    doc_id: 8, score: 2.834953, shard: shardUrls[1], title: titles[query]
  }] : [];
  return { ok: true, json: async () => ({
    results, elapsed_ms: 3.2, shards_queried: 3, shards_failed: [], degraded: false
  }) };
};

vm.runInNewContext(match[1], {
  document, fetch: fetchMock, URLSearchParams, AbortController,
  setTimeout: () => 1, clearTimeout: () => {},
  console, Number, String
}, {filename: "inline-ui.js"});

const sleep = () => new Promise((resolve) => setImmediate(resolve));
async function flush() { await sleep(); await sleep(); }

(async function main() {
  await flush();
  assert.equal(document.getElementById("cluster-state").textContent, "3 shards configured");
  assert.equal(document.getElementById("shard-status-list").children.length, 3);

  chips[0].listeners.click();
  await flush();
  assert.equal(document.getElementById("q").value, "circuit breaker");
  const output = document.getElementById("result-content");
  assert.equal(output.children[0].tagName, "OL");
  assert.match(output.textContent, /Circuit Breaker Pattern/);
  assert.match(output.textContent, /BM25 2.835/);
  assert.match(output.textContent, /Shard 02/);
  assert.doesNotMatch(output.textContent, /127\.0\.0\.1/);
  assert.match(document.getElementById("meta").textContent, /3\/3 shards answered/);

  chips[2].listeners.click();
  await flush();
  assert.match(output.textContent, /Vector Similarity Search/);
  assert.doesNotMatch(output.textContent, /Circuit Breaker Pattern/);

  document.getElementById("q").value = "unknown unindexed phrase";
  document.getElementById("search-form").listeners.submit({preventDefault() {}});
  await flush();
  assert.match(output.textContent, /No matching documents/);

  document.listeners.keydown({
    key: "/", target: new Element(), ctrlKey: false, metaKey: false,
    preventDefault() {}
  });
  assert.equal(document.activeElement, document.getElementById("q"));
  assert.ok(requests.some((url) => url.includes("q=circuit+breaker")));

  console.log("PASS: example chips, real result rendering, no matches, shortcuts, shard telemetry");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
