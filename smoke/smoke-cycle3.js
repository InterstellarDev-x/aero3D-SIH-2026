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
  await page.waitForTimeout(2000);
  const cards = await page.$$('#scene-cards .card');
  await cards[0].click();
  await page.waitForFunction(
    () => !document.getElementById('screen-viewer').classList.contains('hidden'),
    { timeout: 30000 });
  await page.waitForTimeout(4000);

  // 1. point-cloud toggle
  await page.click('#btn-points');
  await page.waitForTimeout(2500);
  await shot('shot-cycle3-points');

  // 2. flood: toggle on, set slider to 45%
  await page.click('#btn-flood');
  await page.evaluate(() => {
    const r = document.getElementById('flood-range');
    r.value = '45'; r.dispatchEvent(new Event('input'));
  });
  await page.waitForTimeout(2000);
  const floodVal = await page.textContent('#flood-val');
  await shot('shot-cycle3-flood');
  console.log('flood label:', floodVal.trim());

  // 3. places panel + fly to a preset
  await page.click('#btn-places');
  await page.waitForTimeout(800);
  await shot('shot-cycle3-places');
  const flies = await page.$$('#places-list .pfly');
  console.log('place rows:', flies.length);
  await flies[0].click(); // fly to highest peak
  await page.waitForTimeout(3500); // fly-to duration is 2.4s
  await shot('shot-cycle3-places-fly');

  // 4. save current view -> persisted row appears
  await page.click('#btn-place-save');
  await page.waitForTimeout(600);
  const flies2 = await page.$$('#places-list .pfly');
  console.log('place rows after save:', flies2.length);
  await shot('shot-cycle3-places-saved');

  console.log('console errors:', errors.length);
  errors.forEach(e => console.log('ERR:', e));
  await browser.close();
  process.exit(errors.length ? 1 : 0);
})().catch(e => { console.error('FATAL', e); process.exit(2); });
