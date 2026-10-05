const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'],
  });
  const page = await browser.newPage({ viewport: { width: 1100, height: 700 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text().slice(0, 200)); });
  page.on('pageerror', e => errors.push('PAGEERROR: ' + String(e.message).slice(0, 200)));
  const cdp = await page.context().newCDPSession(page);
  const shot = async (n) => {
    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' });
    require('fs').writeFileSync(`/home/hatch/workspace/aero3d/smoke/${n}.png`, Buffer.from(data, 'base64'));
  };
  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);
  const cards = await page.$$('#scene-cards .card');
  await cards[0].click();
  // wait for processing to reach the classification preview
  await page.waitForTimeout(9000);
  await shot('shot-proc-class');
  await page.waitForFunction(
    () => !document.getElementById('screen-viewer').classList.contains('hidden'), { timeout: 30000 });
  await page.waitForTimeout(2500);
  // switch to land-cover overlay
  await page.selectOption('#overlay-sel', '3');
  await page.waitForTimeout(1500);
  await shot('shot-class3d');
  const legendVisible = await page.evaluate(() =>
    !document.getElementById('class-legend').classList.contains('hidden'));
  const legendText = await page.evaluate(() =>
    document.getElementById('class-legend').innerText.slice(0, 160));
  console.log('legend visible:', legendVisible);
  console.log('legend:', JSON.stringify(legendText));
  console.log('CONSOLE ERRORS (' + errors.length + '):');
  errors.slice(0, 10).forEach(e => console.log(' -', e));
  await browser.close();
})().catch(e => { console.error('SMOKE FAILED:', e.message); process.exit(1); });
