// Hosted-site integration. Local servers, forks and downloaded copies stay independent.
(() => {
  const { location } = window;
  if (location.protocol !== 'https:' || location.hostname !== 'saidgurbuz.github.io'
      || !location.pathname.startsWith('/transformer-atlas/')) return;
  if (document.getElementById('atlas-google-tag')) return;

  const measurementId = 'G-8J3PCF3R9G';
  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };
  window.gtag('js', new Date());
  window.gtag('config', measurementId, {
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
  });

  const tag = document.createElement('script');
  tag.id = 'atlas-google-tag';
  tag.async = true;
  tag.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
  document.head.append(tag);

  // Scene visits supplement Google's standard measurement without manual page_view duplicates.
  // Only scene identifiers/titles are included; input text and tensor contents stay in the app.
  window.addEventListener('atlas:view', ({ detail }) => {
    if (!detail || typeof detail.id !== 'string' || typeof detail.title !== 'string') return;
    window.gtag('event', 'atlas_view', {
      send_to: measurementId,
      view_id: detail.id,
      view_name: detail.title,
      page_location: `${location.origin}${location.pathname}#${detail.id}`,
    });
  });
})();
