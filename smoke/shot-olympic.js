const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1100, height: 700 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text().slice(0, 200)); });
  page.on('pageerror', e => errors.push('PAGEERROR: ' + String(e.message).slice(0, 200)));
  const cdp = await page.context().newCDPSession(page);
  const shot = async (n) => {
    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' });
    require('fs').writeFileSync('/home/hatch/workspace/aero3d/smoke/' + n + '.png', Buffer.from(data, 'base64'));
  };
  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2500);
  await shot('shot-olympic-landing');
  const cards = await page.$$('#scene-cards .card');
  console.log('cards:', cards.length, '| first card label:', await cards[0].getAttribute('aria-label'));
  await cards[0].click();
  await page.waitForFunction(() => !document.getElementById('screen-viewer').classList.contains('hidden'), { timeout: 30000 });
  await page.waitForTimeout(4000);
  await shot('shot-olympic-orbit');
  await page.click('[data-mode="drone"]');
  await page.waitForTimeout(2500);
  await shot('shot-olympic-drone');
  console.log('console errors:', errors.length ? errors : 'none');
  await browser.close();
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });
