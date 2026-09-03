// Minimal service worker. Its job is to make the page installable as an app; the
// dashboard is regenerated every few minutes, so nothing is cached - a stale
// cached page showing yesterday's levels would be worse than no app at all.
//
// The worker kept no cache of its own, but a bare fetch(e.request) still goes
// through the browser's HTTP cache, and GitHub Pages serves index.html with a
// max-age. On 3 Sep 2026 the page kept showing a discontinued EVZ gauge after the
// corrected build had been published, because the installed app was serving the
// noon copy. Navigations now bypass the HTTP cache outright, which is what the
// comment above always claimed was happening.
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const req = e.request;
  // Only force-revalidate the page itself; icons and the manifest are static and
  // may as well be cached normally.
  const isPage = req.mode === 'navigate' ||
                 (req.method === 'GET' && new URL(req.url).pathname.endsWith('/'));
  if (!isPage) return;
  e.respondWith(
    fetch(req, { cache: 'no-store' }).catch(() => fetch(req))
  );
});
