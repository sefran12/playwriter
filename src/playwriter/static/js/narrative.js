/* ───────────────────────────────────────────────────────
   Playwriter — Narrative Engine Frontend
   ─────────────────────────────────────────────────────── */

const API = '/api/narrative';
const PROVIDER_API = '/api/providers';

// ── State ────────────────────────────────────────────
let worldId = null;
let autoAdvanceTimer = null;
let isAdvancing = false;
let providersData = {};  // cached: { "openai": {...}, "anthropic": {...}, ... }

// ── DOM refs ─────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const setupPanel     = $('#setup-panel');
const enginePanel    = $('#engine-panel');
const errorToast     = $('#error-toast');
// setupStatus removed — replaced by generation log
const providerSelect = $('#provider-select');
const modelSelect    = $('#model-select');
const seedInput      = $('#seed-description');
const modeSelect     = $('#mode-select');
const tropePoolInput = $('#trope-pool-size');
const numCharsInput  = $('#num-characters');
const createBtn      = $('#create-world-btn');
const sampleBtn      = $('#load-sample-btn');

const worldTitle     = $('#world-title');
const worldStatus    = $('#world-status');
const modeToggle     = $('#mode-toggle');
const backBtn        = $('#back-btn');
const timeline       = $('#timeline');
const proseReader    = $('#prose-reader');
const diceValue      = $('#dice-value');
const diceDetails    = $('#dice-details');
const tropesDisplay  = $('#tropes-display');
const charStrip      = $('#character-strip');
const threadTracker  = $('#thread-tracker');
const directorCtrls  = $('#director-controls');

// Advance buttons
const advBeatBtn     = $('#advance-beat-btn');
const adv5Btn        = $('#advance-5-btn');
const advSceneBtn    = $('#advance-scene-btn');
const advActBtn      = $('#advance-act-btn');
const autoAdvance    = $('#auto-advance');
const streamBtn      = $('#stream-btn');

// Director inputs
const dirActor       = $('#dir-actor');
const dirAction      = $('#dir-action');
const dirRoll        = $('#dir-roll');
const dirOverrideBtn = $('#dir-override-btn');
const dirEvent       = $('#dir-event');
const dirInjectBtn   = $('#dir-inject-btn');
const dirTrope       = $('#dir-trope');
const dirTropeBtn    = $('#dir-trope-btn');


// ── Toast ────────────────────────────────────────────
function showToast(msg, type = 'error', duration = 4000) {
  errorToast.textContent = msg;
  errorToast.className = 'error-toast' + (type !== 'error' ? ` toast-${type}` : '');
  errorToast.classList.remove('hidden');
  clearTimeout(errorToast._timer);
  errorToast._timer = setTimeout(() => errorToast.classList.add('hidden'), duration);
}

errorToast.addEventListener('click', () => errorToast.classList.add('hidden'));


// ── Provider / Model loading ─────────────────────────
// API returns providers as an OBJECT { "openai": {...}, ... }, not an array.
// We cache it as-is and index by name (same pattern as the arena).
async function loadProviders() {
  try {
    const res = await fetch(PROVIDER_API);
    if (!res.ok) return;
    const data = await res.json();
    providersData = data.providers || {};

    if (data.active) {
      providerSelect.value = data.active;
    }
    populateModels(providerSelect.value);

    if (data.active_model) {
      modelSelect.value = data.active_model;
    }
  } catch (e) {
    console.warn('Could not load providers', e);
  }
}

function populateModels(providerName) {
  modelSelect.innerHTML = '<option value="">(Provider Default)</option>';
  const info = providersData[providerName];
  if (!info || !info.models) return;
  for (const m of info.models) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    if (m === info.strong_model) opt.textContent += ' (strong)';
    if (m === info.fast_model) opt.textContent += ' (fast)';
    modelSelect.appendChild(opt);
  }
}

providerSelect.addEventListener('change', () => {
  populateModels(providerSelect.value);
  switchProvider();
});

modelSelect.addEventListener('change', () => switchProvider());

// API expects { name: ... } not { provider: ... }
async function switchProvider() {
  const body = { name: providerSelect.value };
  if (modelSelect.value) body.model = modelSelect.value;
  try {
    const res = await fetch(PROVIDER_API + '/active', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showToast(`Provider switch failed: ${err.detail || res.statusText}`);
    }
  } catch (e) {
    showToast('Failed to switch provider');
  }
}


// ── Sample seed ──────────────────────────────────────
const SAMPLE_SEED = `A crumbling Mediterranean port city, once the jewel of a maritime empire, now caught between a corrupt merchant oligarchy and a populist revolutionary movement. The old lighthouse keeper guards secrets from the city's founding. A plague ship approaches the harbor. Three factions — the Merchants' Guild, the Dockworkers' Union, and the Temple of Tides — each believe they alone can save the city. But the real threat comes from beneath the waves, where something ancient has been awakened by decades of reckless harbor expansion.`;

sampleBtn.addEventListener('click', () => {
  seedInput.value = SAMPLE_SEED;
});


// ── Generation log helpers ────────────────────────────
const generationLog     = $('#generation-log');
const generationEntries = $('#generation-log-entries');

function showGenLog() {
  generationEntries.innerHTML = '';
  generationLog.classList.remove('hidden');
}

function hideGenLog() {
  generationLog.classList.add('hidden');
}

function appendLogEntry(step, detail, isActive = true) {
  const prev = generationEntries.querySelector('.log-entry.active');
  if (prev) {
    prev.classList.remove('active');
    prev.classList.add('done');
    const spinner = prev.querySelector('.spinner');
    if (spinner) spinner.remove();
  }

  const div = document.createElement('div');
  div.className = `log-entry${isActive ? ' active' : ' done'}`;
  div.innerHTML = `${isActive ? '<span class="spinner"></span>' : ''}`
    + `<span class="log-step">${esc(step)}</span>`
    + (detail ? `<div class="log-detail">${esc(detail)}</div>` : '');
  generationEntries.appendChild(div);
  generationLog.scrollTop = generationLog.scrollHeight;
}

function finalizeGenLog() {
  const prev = generationEntries.querySelector('.log-entry.active');
  if (prev) {
    prev.classList.remove('active');
    prev.classList.add('done');
    const spinner = prev.querySelector('.spinner');
    if (spinner) spinner.remove();
  }
}

// ── Create World (SSE streaming with progress log) ─────────
createBtn.addEventListener('click', async () => {
  const seed = seedInput.value.trim();
  if (!seed) {
    showToast('Please enter a world seed description');
    return;
  }

  await switchProvider();

  createBtn.disabled = true;
  showGenLog();

  const body = {
    seed_description: seed,
    mode: modeSelect.value,
    trope_pool_size: parseInt(tropePoolInput.value) || 30,
  };
  const nc = parseInt(numCharsInput.value);
  if (nc && nc > 0) body.num_characters = nc;

  try {
    const res = await fetch(API + '/worlds/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'World creation failed');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let result = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));

          if (event.step === 'error') {
            throw new Error(event.detail || 'World creation failed');
          }

          if (event.step === 'done') {
            result = event;
          } else {
            appendLogEntry(event.step, event.detail);
          }
        } catch (parseErr) {
          if (parseErr.message && !parseErr.message.includes('JSON')) throw parseErr;
        }
      }
    }

    if (!result || !result.world_id) {
      throw new Error('World creation did not complete');
    }

    finalizeGenLog();
    worldId = result.world_id;
    showToast(`World created: ${result.characters.join(', ')}`, 'success', 3000);

    setupPanel.classList.add('hidden');
    enginePanel.classList.remove('hidden');
    modeToggle.value = modeSelect.value;
    updateDirectorVisibility();
    await refreshWorldState();

  } catch (e) {
    appendLogEntry('error', e.message, false);
    showToast(e.message);
  } finally {
    createBtn.disabled = false;
  }
});


// ── Advance controls ─────────────────────────────────
async function doAdvance(steps) {
  if (isAdvancing || !worldId) return;
  isAdvancing = true;
  setAdvanceButtonsDisabled(true);

  try {
    const res = await fetch(API + `/worlds/${worldId}/advance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ steps }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Advance failed');
    }

    const data = await res.json();
    processEvents(data.events);
    await refreshWorldState();

  } catch (e) {
    showToast(e.message);
  } finally {
    isAdvancing = false;
    setAdvanceButtonsDisabled(false);
  }
}

async function doAdvanceScene() {
  if (isAdvancing || !worldId) return;
  isAdvancing = true;
  setAdvanceButtonsDisabled(true);

  try {
    const res = await fetch(API + `/worlds/${worldId}/advance/scene`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Scene advance failed');
    const data = await res.json();
    processEvents(data.events);
    await refreshWorldState();
  } catch (e) {
    showToast(e.message);
  } finally {
    isAdvancing = false;
    setAdvanceButtonsDisabled(false);
  }
}

async function doAdvanceAct() {
  if (isAdvancing || !worldId) return;
  isAdvancing = true;
  setAdvanceButtonsDisabled(true);

  try {
    const res = await fetch(API + `/worlds/${worldId}/advance/act`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Act advance failed');
    const data = await res.json();
    processEvents(data.events);
    await refreshWorldState();
  } catch (e) {
    showToast(e.message);
  } finally {
    isAdvancing = false;
    setAdvanceButtonsDisabled(false);
  }
}

function setAdvanceButtonsDisabled(disabled) {
  advBeatBtn.disabled = disabled;
  adv5Btn.disabled = disabled;
  advSceneBtn.disabled = disabled;
  advActBtn.disabled = disabled;
  streamBtn.disabled = disabled;
}

advBeatBtn.addEventListener('click', () => doAdvance(1));
adv5Btn.addEventListener('click', () => doAdvance(5));
advSceneBtn.addEventListener('click', () => doAdvanceScene());
advActBtn.addEventListener('click', () => doAdvanceAct());

// Auto-advance
autoAdvance.addEventListener('change', () => {
  if (autoAdvance.checked) {
    autoAdvanceTimer = setInterval(() => {
      if (!isAdvancing) doAdvance(1);
    }, 2000);
  } else {
    clearInterval(autoAdvanceTimer);
    autoAdvanceTimer = null;
  }
});

// SSE Stream
streamBtn.addEventListener('click', async () => {
  if (!worldId) return;
  streamBtn.disabled = true;
  streamBtn.textContent = 'Streaming...';

  try {
    const evtSource = new EventSource(API + `/worlds/${worldId}/stream?steps=20`);
    evtSource.onmessage = (e) => {
      const event = JSON.parse(e.data);
      if (event.type === 'stream_complete' || event.type === 'error') {
        evtSource.close();
        streamBtn.disabled = false;
        streamBtn.textContent = 'Stream';
        if (event.type === 'error') showToast(event.message);
        refreshWorldState();
        return;
      }
      processEvents([event]);
    };
    evtSource.onerror = () => {
      evtSource.close();
      streamBtn.disabled = false;
      streamBtn.textContent = 'Stream';
      refreshWorldState();
    };
  } catch (e) {
    showToast(e.message);
    streamBtn.disabled = false;
    streamBtn.textContent = 'Stream';
  }
});


// ── Process events from advance ──────────────────────
function processEvents(events) {
  for (const ev of events) {
    switch (ev.type) {
      case 'beat_resolved':
        appendBeatProse(ev);
        updateDice(ev);
        updateTropes(ev);
        break;
      case 'scene_composed':
        appendSceneBreak(ev);
        break;
      case 'scene_completed':
        showToast(`Scene ${ev.scene_number} completed (${ev.beats_count} beats)`, 'info', 2000);
        break;
      case 'act_planned':
        appendActBreak(ev);
        break;
      case 'act_completed':
        showToast(`Act ${ev.act_number} completed`, 'success', 3000);
        break;
    }
  }
}


// ── Prose rendering ──────────────────────────────────
function clearProse() {
  proseReader.innerHTML = '';
}

function appendBeatProse(ev) {
  // Remove placeholder
  const ph = proseReader.querySelector('.prose-placeholder');
  if (ph) ph.remove();

  const div = document.createElement('div');
  div.className = 'prose-beat';

  const outcome = ev.dice_outcome || 'mixed';
  div.innerHTML = `
    <div class="prose-beat-header">
      <span class="prose-beat-actor">${esc(ev.actor)}</span>
      <span class="prose-beat-outcome ${outcome}">${outcome.replaceAll('_', ' ')}</span>
      <span>d100: ${ev.raw_roll || '?'} → ${ev.final_value || '?'}</span>
    </div>
    <div class="prose-beat-text">${esc(ev.prose || ev.actual_outcome || '')}</div>
  `;
  proseReader.appendChild(div);
  proseReader.scrollTop = proseReader.scrollHeight;
}

function appendSceneBreak(ev) {
  const ph = proseReader.querySelector('.prose-placeholder');
  if (ph) ph.remove();

  const div = document.createElement('div');
  div.className = 'prose-scene-break';
  const setting = ev.setting || 'Unknown setting';
  const actors = (ev.actors || []).join(', ') || 'Unknown actors';
  div.textContent = `Scene ${ev.scene_number} — ${setting} [${actors}]`;
  proseReader.appendChild(div);
  proseReader.scrollTop = proseReader.scrollHeight;
}

function appendActBreak(ev) {
  const ph = proseReader.querySelector('.prose-placeholder');
  if (ph) ph.remove();

  const div = document.createElement('div');
  div.className = 'prose-act-break';
  div.textContent = `${ev.title || 'Act ' + ev.act_number}`;
  proseReader.appendChild(div);
  proseReader.scrollTop = proseReader.scrollHeight;
}


// ── Dice display ─────────────────────────────────────
function updateDice(ev) {
  const outcome = ev.dice_outcome || '';
  diceValue.textContent = ev.final_value || '--';
  diceValue.className = 'dice-value ' + outcome;
  diceDetails.textContent = `Raw: ${ev.raw_roll || '?'} | ${outcome.replaceAll('_', ' ')}`;
}


// ── Tropes display ───────────────────────────────────
function updateTropes(ev) {
  if (!worldId) return;
  fetch(API + `/worlds/${worldId}/dice-history`)
    .then(r => r.json())
    .then(data => {
      const rolls = data.rolls || [];
      if (!rolls.length) return;

      // Show tropes from the last few rolls
      const recent = rolls.slice(-3);
      const tropeSet = new Map();
      for (const roll of recent) {
        for (const fm of (roll.fate_modifiers || [])) {
          tropeSet.set(fm.trope, `${fm.trope} (${fm.modifier >= 0 ? '+' : ''}${fm.modifier})`);
        }
      }

      if (tropeSet.size === 0) {
        tropesDisplay.innerHTML = '<span class="dim">None yet</span>';
        return;
      }
      tropesDisplay.innerHTML = Array.from(tropeSet.values())
        .map(t => `<span class="trope-tag">${esc(t)}</span>`)
        .join('');
    })
    .catch(() => {});
}


// ── Refresh full world state ─────────────────────────
async function refreshWorldState() {
  if (!worldId) return;

  try {
    const res = await fetch(API + `/worlds/${worldId}/summary`);
    if (!res.ok) return;
    const data = await res.json();

    // Header — show teleology as title
    worldTitle.textContent = data.teleology
      ? data.teleology.substring(0, 60) + (data.teleology.length > 60 ? '...' : '')
      : 'Narrative World';
    worldStatus.textContent = data.status;
    worldStatus.className = 'status-badge' + (data.status.includes('completed') ? '' : ' active');

    // Timeline
    renderTimeline(data.acts);

    // Characters
    await renderCharacters();

    // Threads
    renderThreads(data.threads);

  } catch (e) {
    console.warn('Refresh failed', e);
  }
}


// ── Timeline rendering ───────────────────────────────
function renderTimeline(acts) {
  timeline.innerHTML = '';
  if (!acts || !acts.length) {
    timeline.innerHTML = '<span class="dim">No acts yet</span>';
    return;
  }

  for (const act of acts) {
    const actDiv = document.createElement('div');
    actDiv.className = 'tl-act';

    const title = document.createElement('div');
    title.className = 'tl-act-title';
    title.textContent = `${act.title || 'Act ' + act.number} [${act.status}]`;
    actDiv.appendChild(title);

    for (const scene of (act.scenes || [])) {
      const scDiv = document.createElement('div');
      scDiv.className = 'tl-scene';

      const scTitle = document.createElement('div');
      scTitle.className = 'tl-scene-title';
      const setting = (scene.setting || 'Scene').substring(0, 25);
      scTitle.textContent = `Sc ${scene.number}: ${setting}...`;
      scTitle.addEventListener('click', () => scrollToScene(scene.number));
      scDiv.appendChild(scTitle);

      for (const beat of (scene.beats || [])) {
        const bDiv = document.createElement('div');
        bDiv.className = 'tl-beat';
        // outcome from API is already lowercase (enum value)
        const outcome = beat.outcome || 'mixed';
        bDiv.innerHTML = `
          <span class="tl-beat-dot ${outcome}"></span>
          <span class="tl-beat-label">${esc(beat.actor)}</span>
        `;
        scDiv.appendChild(bDiv);
      }

      actDiv.appendChild(scDiv);
    }

    timeline.appendChild(actDiv);
  }
}

function scrollToScene(sceneNum) {
  const breaks = proseReader.querySelectorAll('.prose-scene-break');
  for (const br of breaks) {
    if (br.textContent.includes(`Scene ${sceneNum}`)) {
      br.scrollIntoView({ behavior: 'smooth', block: 'start' });
      break;
    }
  }
}


// ── Character rendering ──────────────────────────────
async function renderCharacters() {
  if (!worldId) return;
  try {
    const res = await fetch(API + `/worlds/${worldId}/characters`);
    if (!res.ok) return;
    const chars = await res.json();

    charStrip.innerHTML = '';
    for (const [name, char] of Object.entries(chars)) {
      const card = document.createElement('div');
      card.className = 'char-card';
      card.innerHTML = `
        <div class="char-card-name">${esc(name)}</div>
        <div class="char-card-state">${esc((char.internal_state || '').substring(0, 60))}</div>
      `;
      card.addEventListener('click', () => showCharacterModal(name, char));
      charStrip.appendChild(card);
    }
  } catch (e) {
    console.warn('Char render failed', e);
  }
}

function showCharacterModal(name, char) {
  // Remove existing modal
  const existing = document.querySelector('.char-modal-overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.className = 'char-modal-overlay';
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.remove();
  });

  const fields = [
    ['Name', char.name],
    ['Internal State', char.internal_state],
    ['Ambitions', char.ambitions],
    ['Teleology', char.teleology],
    ['Philosophy', char.philosophy],
    ['Physical State', char.physical_state],
    ['Long-Term Memory', (char.long_term_memory || []).join('\n')],
    ['Short-Term Memory', (char.short_term_memory || []).join('\n')],
    ['Internal Contradictions', (char.internal_contradictions || []).join('\n')],
  ];

  let fieldsHtml = fields.map(([label, val]) => `
    <div class="char-modal-field">
      <div class="char-modal-label">${label}</div>
      <div class="char-modal-value">${esc(val || '(empty)')}</div>
    </div>
  `).join('');

  overlay.innerHTML = `
    <div class="char-modal">
      <h3>${esc(name)}</h3>
      ${fieldsHtml}
      <button class="btn-control char-modal-close">Close</button>
    </div>
  `;

  overlay.querySelector('.char-modal-close').addEventListener('click', () => overlay.remove());
  document.body.appendChild(overlay);
}


// ── Thread rendering ─────────────────────────────────
function renderThreads(threads) {
  threadTracker.innerHTML = '';
  if (!threads || !threads.length) {
    threadTracker.innerHTML = '<span class="dim">No threads yet</span>';
    return;
  }

  for (const t of threads) {
    const div = document.createElement('div');
    div.className = 'thread-item';

    const tension = t.tension || 0;
    const tensionClass = tension <= 3 ? 'low' : tension <= 6 ? 'mid' : 'high';
    const tensionPct = tension * 10;

    div.innerHTML = `
      <span class="thread-status ${t.status}">${t.status}</span>
      <span class="thread-text" title="${esc(t.thread || '')}">${esc((t.thread || '').substring(0, 60))}</span>
      <div class="thread-tension">
        <div class="thread-tension-fill ${tensionClass}" style="width: ${tensionPct}%"></div>
      </div>
    `;
    threadTracker.appendChild(div);
  }
}


// ── Director controls ────────────────────────────────
function updateDirectorVisibility() {
  if (modeToggle.value === 'director') {
    directorCtrls.classList.remove('hidden');
  } else {
    directorCtrls.classList.add('hidden');
  }
}

modeToggle.addEventListener('change', async () => {
  updateDirectorVisibility();
  if (!worldId) return;
  try {
    await fetch(API + `/worlds/${worldId}/mode`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: modeToggle.value }),
    });
    showToast(`Mode: ${modeToggle.value}`, 'info', 1500);
  } catch (e) {
    showToast('Failed to switch mode');
  }
});

dirOverrideBtn.addEventListener('click', async () => {
  if (!worldId) return;
  const actor = dirActor.value.trim();
  const action = dirAction.value.trim();
  const roll = parseInt(dirRoll.value);
  if (!actor || !action || !roll) {
    showToast('Fill in actor, action, and roll');
    return;
  }
  try {
    const res = await fetch(API + `/worlds/${worldId}/director/override-dice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor, action, forced_roll: roll }),
    });
    if (!res.ok) throw new Error('Override failed');
    const beat = await res.json();
    processEvents([{
      type: 'beat_resolved',
      actor: beat.actor,
      prose: beat.prose,
      actual_outcome: beat.actual_outcome,
      dice_outcome: beat.dice_roll?.outcome,
      raw_roll: beat.dice_roll?.raw_roll,
      final_value: beat.dice_roll?.final_value,
    }]);
    await refreshWorldState();
    showToast('Dice overridden', 'success', 2000);
  } catch (e) {
    showToast(e.message);
  }
});

dirInjectBtn.addEventListener('click', async () => {
  if (!worldId) return;
  const desc = dirEvent.value.trim();
  if (!desc) return;
  try {
    const res = await fetch(API + `/worlds/${worldId}/director/inject-event`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event_description: desc }),
    });
    if (!res.ok) throw new Error('Inject failed');
    showToast('Event injected', 'success', 2000);
    dirEvent.value = '';
    await refreshWorldState();
  } catch (e) {
    showToast(e.message);
  }
});

dirTropeBtn.addEventListener('click', async () => {
  if (!worldId) return;
  const query = dirTrope.value.trim();
  if (!query) return;
  try {
    const res = await fetch(API + `/worlds/${worldId}/director/force-trope`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trope_query: query }),
    });
    if (!res.ok) throw new Error('Force trope failed');
    const tropes = await res.json();
    showToast(`Injected ${tropes.length} tropes`, 'success', 2000);
    dirTrope.value = '';
  } catch (e) {
    showToast(e.message);
  }
});


// ── Back to setup ────────────────────────────────────
backBtn.addEventListener('click', () => {
  if (!confirm('Create a new world? Current world will be lost.')) return;
  clearInterval(autoAdvanceTimer);
  autoAdvance.checked = false;
  worldId = null;
  enginePanel.classList.add('hidden');
  setupPanel.classList.remove('hidden');
  hideGenLog();
  clearProse();
  proseReader.innerHTML = '<p class="prose-placeholder">Create a world to begin generating narrative...</p>';
});


// ── Utility ──────────────────────────────────────────
function esc(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}


// ── Init ─────────────────────────────────────────────
loadProviders();
