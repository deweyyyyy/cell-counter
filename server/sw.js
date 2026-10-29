// Cell Counter offline support: keep the page itself available without internet.
// Pages: network first (so updates and Cloudflare Access logins work), cached copy when offline.
// Fonts: cache first. API calls are never cached; the page queues lab entries itself.
const CACHE = 'cellcounter-v1';
const SHELL = ['./', './manifest.webmanifest'];
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {})); self.skipWaiting(); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET') return;
  if (url.origin === location.origin && (url.pathname.includes('/api/') || url.pathname.includes('/reports/'))) return;
  if (/fonts\.(googleapis|gstatic)\.com$/.test(url.hostname)) {
    e.respondWith(caches.open(CACHE).then(async c => (await c.match(req)) || fetch(req).then(r => { c.put(req, r.clone()); return r; })));
    return;
  }
  if (url.origin !== location.origin) return;
  e.respondWith(fetch(req).then(r => {
    // only cache real pages from our server, never a login redirect
    if (r.ok && r.type === 'basic' && !r.redirected) caches.open(CACHE).then(c => c.put(req, r.clone()));
    return r;
  }).catch(() => caches.match(req).then(r => r || caches.match('./'))));
});
