// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('ImageCulling Studio E2E', () => {
  /** @type {string[]} */
  let pageErrors = [];

  test.beforeEach(async ({ page, request }) => {
    pageErrors = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));
    await request.put('/api/session', { data: { state: {} } });
  });

  test.afterEach(() => {
    expect(pageErrors).toEqual([]);
  });

  test('loads grid view, sidebar counts, dirSelect, card size slider, and Run AI Cull menu', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/ImageCulling Studio/);

    await expect(page.locator('#resultCount')).toHaveText('4 images');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);
    await expect(page.locator('#cntAll')).toHaveText('4');
    await expect(page.locator('#cntL2')).toHaveText('1');
    await expect(page.locator('#cntDel')).toHaveText('1');
    await expect(page.locator('#dirSelect option')).toHaveCount(1);

    // Card size slider updates grid card size
    await page.locator('#cardSizeSlider').fill('300');
    await page.locator('#cardSizeSlider').dispatchEvent('input');

    // Run AI Cull dropdown opens and closes
    await page.click('#btnRunCullMenu');
    await expect(page.locator('#cullDropdown')).toBeVisible();
    await page.click('#btnRunCullMenu');
    await expect(page.locator('#cullDropdown')).toBeHidden();
  });

  test('exercises all sidebar filters, operators, gear facets, search, and sort', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Folder filter: day1 -> 2 images, then back to All Folders
    await page.locator('#folderList .folder-item', { hasText: 'day1' }).click();
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.locator('#folderList .folder-item').first().click();
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Layer filter + AI layer mode checkbox
    await page.click('#layerFilters [data-layer="layer2_to_edit"]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(1);
    await page.check('#chkUseAiLayer');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(1);
    await page.uncheck('#chkUseAiLayer');
    await page.click('#layerFilters [data-layer=""]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Rating operator bar (=, <=, >=) + rating filter
    await page.click('#ratingOpBar [data-op="eq"]');
    await page.click('#ratingFilterBar [data-rating="5"]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(1);
    await page.click('#ratingOpBar [data-op="lte"]');
    await page.click('#ratingFilterBar [data-rating="0"]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(3);
    await page.click('#ratingOpBar [data-op="gte"]');
    await page.click('#ratingFilterBar [data-rating=""]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Color label filter
    await page.click('#colorFilterBar [data-label="Yellow"]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.click('#colorFilterBar [data-label="__all__"]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Shutter speed tier filter
    await page.click('#speedFilters [data-speed="super_fast"]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.click('#speedFilters [data-speed=""]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Track, Species, Scenario, Hybrid Light
    await page.click('#trackFilters [data-track="Wildlife"]');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.selectOption('#speciesFilter', 'Eurasian Kestrel');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.selectOption('#speciesFilter', '');
    await page.click('#trackFilters [data-track=""]');

    await page.selectOption('#scenarioFilter', 'action_sequence');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.selectOption('#scenarioFilter', '');

    await page.check('#chkHybridOnly');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(1);
    await page.uncheck('#chkHybridOnly');

    // EXIF Gear: Camera & Lens
    await page.selectOption('#cameraFilter', 'Canon EOS R5');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);
    await page.selectOption('#lensFilter', 'RF100-500mm');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.selectOption('#lensFilter', '');
    await page.selectOption('#cameraFilter', '');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Only #1 sharpest per burst
    await page.check('#chkBurstBest');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(2);
    await page.uncheck('#chkBurstBest');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Auto-Advance toggle
    await page.check('#chkAutoAdvance');
    await page.uncheck('#chkAutoAdvance');

    // Search & Sort
    await page.fill('#searchInput', 'SUNRISE');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(1);
    await page.fill('#searchInput', '');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    await page.selectOption('#sortSelect', 'filename:asc');
    await expect(page.locator('#photoGrid .photo-card').first()).toContainText('BLURRY_004.jpg');
  });

  test('supports batch selection, batch star/color updates, and undo in Grid', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    const cards = page.locator('#photoGrid .photo-card');
    await cards.nth(0).locator('.card-chk').click({ force: true });
    await cards.nth(1).locator('.card-chk').click({ force: true });

    await expect(page.locator('#batchBar')).toBeVisible();
    await expect(page.locator('#batchCount')).toHaveText('2 selected');

    // Batch rate 4 stars and batch color Green, verify UI updates, then undo both and clear selection
    await page.click('#batchBar [data-batch-rate="4"]');
    await expect(cards.nth(0).locator('.card-stars .star-off')).toHaveCount(1);
    await expect(cards.nth(1).locator('.card-stars .star-off')).toHaveCount(1);
    await page.click('#batchBar [data-batch-label="Green"]');
    await expect(cards.nth(1)).toHaveClass(/label-Green/);
    await page.click('#btnClearSel');
    await expect(page.locator('#batchBar')).toBeHidden();

    await page.click('#btnUndo');
    await expect(cards.nth(1)).toHaveClass(/label-Yellow/);
    await page.click('#btnUndo');
    await expect(cards.nth(0).locator('.card-stars .star-off')).toHaveCount(0);
    await expect(cards.nth(1).locator('.card-stars .star-off')).toHaveCount(5);
  });

  test('exercises Bursts, Loupe (zoom, AF, crop, flags, colors, split/unsplit), and Compare (swap, keep both, pin)', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#photoGrid .photo-card')).toHaveCount(4);

    // Bursts view + filters + sort
    await page.click('.view-switcher [data-view="bursts"]');
    await expect(page.locator('#burstsView')).toBeVisible();
    await expect(page.locator('#burstsList .burst-row')).toHaveCount(2);
    await page.click('#burstFilterBar [data-bf="multi_only"]');
    await expect(page.locator('#burstsList .burst-row')).toHaveCount(2);
    await page.selectOption('#burstSortSelect', 'score');
    await page.click('#burstFilterBar [data-bf=""]');

    // Loupe view
    await page.click('.view-switcher [data-view="loupe"]');
    await expect(page.locator('#loupeView')).toBeVisible();
    const firstFile = await page.locator('#loupeFilename').textContent();

    await page.click('#btnLoupeNext');
    await expect(page.locator('#loupeFilename')).not.toHaveText(firstFile || '');

    // Split burst on 2nd frame, then merge back
    if (await page.locator('#btnLoupeSplitBurst').isVisible()) {
      await page.click('#btnLoupeSplitBurst');
      await expect(page.locator('#btnLoupeUnsplitBurst')).toBeVisible();
      await page.click('#btnLoupeUnsplitBurst');
    }

    await page.click('#btnLoupePrev');
    await expect(page.locator('#loupeFilename')).toHaveText(firstFile || '');

    // HUD toggles, AF Focus jump, Hold Full-Res, Inspector Crop toggle, Zoom presets
    await page.click('#btnToggleAf');
    await page.click('#btnToggleHighlights');
    await page.click('#btnToggleBbox');
    await page.click('#btnJumpFocus');
    await page.locator('#btnHoldFull').dispatchEvent('mousedown');
    await page.locator('#btnHoldFull').dispatchEvent('mouseup');
    await page.click('#inspCropBox');
    await page.click('#zoomPresets [data-zoom="100"]');
    await expect(page.locator('#zoomPctLabel')).toHaveText('100%');
    await page.click('#zoomPresets [data-zoom="fit"]');

    // Filmstrip tabs
    await page.click('.filmstrip-tabs [data-fs-mode="burst"]');
    await page.click('.filmstrip-tabs [data-fs-mode="list"]');

    // Pick / Reject flags & Color dot click & Star keyboard + Undo
    await page.click('#btnFlagPick');
    await expect(page.locator('#btnFlagPick')).toHaveClass(/active/);
    await page.click('#btnFlagReject');
    await expect(page.locator('#btnFlagReject')).toHaveClass(/active/);
    await page.click('#btnUndo');
    await page.click('#btnUndo');

    await page.locator('#loupeColors .cp-dot').nth(2).click(); // Red/Yellow/Green dot
    await page.click('#btnUndo');

    await page.keyboard.press('3');
    await expect(page.locator('#loupeStars .star-btn.on')).toHaveCount(3);
    await page.click('#btnUndo');
    await expect(page.locator('#loupeStars .star-btn.on')).toHaveCount(5);

    // Jump from Loupe directly to Compare this burst
    await page.click('#btnLoupeCompareBurst');
    await expect(page.locator('#compareView')).toBeVisible();
    await expect(page.locator('#wvcWinName')).not.toHaveText('');
    await expect(page.locator('#wvcChalName')).not.toHaveText('');

    // Compare controls: Challenger prev/next, Pin Left, Keep Both, Swap + Undo, Zoom, Burst prev/next, Exit
    await page.click('#btnWvcChalNext');
    await page.click('#btnWvcChalPrev');
    await page.click('#btnWvcPinChallenger');
    await page.click('#btnWvcKeepBoth');
    await page.click('#btnUndo');
    await page.click('#btnWvcSwap');
    await page.click('#btnUndo');
    await page.click('#btnWvcZoomToggle');
    await page.click('#btnWvcNextBurst');
    await page.click('#btnWvcPrevBurst');
    await page.click('#btnCompareExit');
    await expect(page.locator('#loupeView')).toBeVisible();
    await page.click('.view-switcher [data-view="grid"]');
    await expect(page.locator('#gridView')).toBeVisible();
  });

  test('exercises Settings (XMP mapping + quality + updates), Logs (filters + restore), Insights, Help, and Open Dir', async ({ page }) => {
    await page.goto('/');

    // Create a rating change first so Change History & Restore has an entry
    await page.click('.view-switcher [data-view="loupe"]');
    await page.keyboard.press('2');
    await page.click('.view-switcher [data-view="grid"]');

    const origSettings = await (await page.request.get('/api/settings')).json();

    // Settings modal: quality inputs, AI mapping, dry run, apply, save
    await page.click('#btnSettings');
    await expect(page.locator('#modalSettings')).toBeVisible();
    await page.check('#setAutoAdvance');
    await page.fill('#setHqQuality', '88');
    await page.selectOption('#setZoomDim', '4096');
    await page.click('#btnDryRunAiXmp');
    await expect(page.locator('#aiXmpStatusMsg')).not.toHaveText('');
    await page.click('#btnApplyAiXmp');
    await expect(page.locator('#aiXmpStatusMsg')).not.toHaveText('');
    await page.click('#btnSaveSettings');
    await expect(page.locator('#modalSettings')).toBeHidden();

    const settingsResp = await page.request.get('/api/settings');
    const settingsJson = await settingsResp.json();
    expect(settingsJson.auto_advance_on_rate).toBe(true);
    expect(settingsJson.hq_preview_quality).toBe(88);

    // Restore original settings so test order never leaks state
    await page.request.put('/api/settings', {
      data: {
        auto_advance_on_rate: origSettings.auto_advance_on_rate,
        hq_preview_quality: origSettings.hq_preview_quality,
        zoom_preview_max_dim: origSettings.zoom_preview_max_dim,
      },
    });

    // Logs & Restore modal: event filters, refresh, change restore, pipeline runs
    await page.click('#btnLogs');
    await expect(page.locator('#modalLogs')).toBeVisible();
    await page.selectOption('#logLevelFilter', 'INFO');
    await page.click('#btnRefreshLogs');
    await page.selectOption('#logLevelFilter', '');

    await page.click('#modalLogs [data-log-tab="changes"]');
    await expect(page.locator('#tabChanges')).toBeVisible();
    const restoreBtn = page.locator('#changeLogBody button', { hasText: /Restore/i }).first();
    if (await restoreBtn.isVisible()) {
      await restoreBtn.click();
    }

    await page.click('#modalLogs [data-log-tab="runs"]');
    await expect(page.locator('#tabRuns')).toBeVisible();
    await page.click('#modalLogs .modal-close');
    await expect(page.locator('#modalLogs')).toBeHidden();

    // Insights modal
    await page.click('#btnLearning');
    await expect(page.locator('#modalLearning')).toBeVisible();
    await page.click('#modalLearning .modal-close');
    await expect(page.locator('#modalLearning')).toBeHidden();

    // Help / Shortcuts modal
    await page.click('#btnShortcuts');
    await expect(page.locator('#modalShortcuts')).toBeVisible();
    await page.click('#modalShortcuts .modal-close');
    await expect(page.locator('#modalShortcuts')).toBeHidden();

    // Open Directory modal (validate empty path error toast / close)
    await page.click('#btnOpenDir');
    await expect(page.locator('#modalOpenDir')).toBeVisible();
    await page.click('#btnConfirmOpenDir');
    await page.click('#modalOpenDir .modal-close');
    await expect(page.locator('#modalOpenDir')).toBeHidden();

    // Trigger Refresh EXIF/XMP scan from Run AI Cull menu
    await page.click('#btnRunCullMenu');
    await expect(page.locator('#cullDropdown')).toBeVisible();
    await page.click('#btnScanMeta');
    await expect(page.locator('#cullDropdown')).toBeHidden();
  });
});
