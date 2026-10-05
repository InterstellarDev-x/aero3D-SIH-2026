const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'],
  });
  const page = await browser.newPage({ viewport: { width: 1100, height: 700 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text().slice(0, 300)); });
  page.on('pageerror', e => errors.push('PAGEERROR: ' + String(e.message).slice(0, 300)));

  const cdp = await page.context().newCDPSession(page);
  const shot = async (n) => {
    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' });
    require('fs').writeFileSync(`/home/hatch/workspace/aero3d/smoke/${n}.png`, Buffer.from(data, 'base64'));
  };

  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2500);
  await shot('shot-landing');

  const cards = await page.$$('#scene-cards .card');
  console.log('scene cards found:', cards.length);
  await cards[0].click();

  // wait for viewer screen
  await page.waitForFunction(
    () => !document.getElementById('screen-viewer').classList.contains('hidden'),
    { timeout: 30000 });
  await page.waitForTimeout(3500);
  await shot('shot-viewer-orbit');
  console.log('viewer visible, webgl info:',
    await page.evaluate(() => {
      const c = document.querySelector('#gl canvas');
      return c ? `${c.width}x${c.height}` : 'NO CANVAS';
    }));

  // drone mode
  await page.click('[data-mode="drone"]');
  await page.waitForTimeout(2500);
  await shot('shot-drone');

  // slope overlay
  await page.selectOption('#overlay-sel', '1');
  await page.waitForTimeout(1200);
  await shot('shot-slope');

  // contours
  await page.click('#btn-contours');
  await page.waitForTimeout(1000);
  await shot('shot-contours');

  // probe: enable + click center
  await page.click('#btn-probe');
  await page.mouse.click(720, 450);
  await page.waitForTimeout(800);
  await shot('shot-probe');
  const probeVisible = await page.evaluate(() =>
    !document.getElementById('probe-pop').classList.contains('hidden'));
  console.log('probe popup visible:', probeVisible);

  // metrics panel content
  const metrics = await page.evaluate(() => document.getElementById('metrics-body').innerText.slice(0, 200));
  console.log('metrics:', JSON.stringify(metrics));

  console.log('CONSOLE ERRORS (' + errors.length + '):');
  errors.slice(0, 15).forEach(e => console.log(' -', e));
  await browser.close();
})().catch(e => { console.error('SMOKE FAILED:', e.message); process.exit(1); });
