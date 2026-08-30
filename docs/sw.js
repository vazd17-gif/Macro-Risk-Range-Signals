// Minimal service worker. Its job is to make the page installable as an app; the
// dashboard is regenerated every few minutes, so nothing is cached - a stale
// cached page showing yesterday's levels would be worse than no app at all.
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => e.respondWith(fetch(e.request)));
