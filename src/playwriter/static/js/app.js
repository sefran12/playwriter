// ─── Playwriter Character Arena — Frontend ──────────────────────────────
const API = '';  // same origin

// ─── State ──────────────────────────────────────────────────────────────
let sessionId = null;
let characterName = '';
let providersData = {};  // cached provider info

// ─── DOM refs ───────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const setupPanel     = $('#setup-panel');
const gamePanel      = $('#game-panel');
const providerSelect = $('#provider-select');
const modelSelect    = $('#model-select');
const tccInput       = $('#tcc-context');
const charDescInput  = $('#char-desc');
const sceneDescInput = $('#scene-desc');
const injectTropes   = $('#inject-tropes');
const tropePreview   = $('#trope-preview');
const startBtn       = $('#start-btn');
const loadSampleBtn  = $('#load-sample-btn');
const setupStatus    = $('#setup-status');
const charNameDisp   = $('#character-name-display');
const sceneLabel     = $('#scene-label');
const profileToggle  = $('#profile-toggle');
const profileSidebar = $('#profile-sidebar');
const profileContent = $('#profile-content');
const worldToggle    = $('#world-toggle');
const worldSidebar   = $('#world-sidebar');
const worldContent   = $('#world-content');
const chatMessages   = $('#chat-messages');
const chatArea       = $('#chat-area');
const userInput      = $('#user-input');
const sendBtn        = $('#send-btn');
const newSessionBtn  = $('#new-session-btn');
const errorToast     = $('#error-toast');

// ─── Init ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Load providers and populate model selector
  await loadProviders();

  // Event listeners
  startBtn.addEventListener('click', startArena);
  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', onInputKeydown);
  profileToggle.addEventListener('click', toggleProfile);
  worldToggle.addEventListener('click', toggleWorld);
  newSessionBtn.addEventListener('click', resetToSetup);
  loadSampleBtn.addEventListener('click', loadSampleContext);
  injectTropes.addEventListener('change', onTropeToggle);
  providerSelect.addEventListener('change', onProviderChange);
  modelSelect.addEventListener('change', switchProvider);
  errorToast.addEventListener('click', () => errorToast.classList.add('hidden'));

  // Pre-fetch tropes if checked
  if (injectTropes.checked) fetchTropePreview();
});

// ─── Toast notifications ─────────────────────────────────────────────────
let toastTimeout = null;

function showToast(message, type = 'error', duration = 5000) {
  errorToast.textContent = message;
  errorToast.className = 'error-toast';
  if (type === 'success') errorToast.classList.add('toast-success');
  else if (type === 'info') errorToast.classList.add('toast-info');
  errorToast.classList.remove('hidden');

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    errorToast.classList.add('hidden');
  }, duration);
}

// ─── Provider & model management ─────────────────────────────────────────
async function loadProviders() {
  try {
    const res = await fetch(`${API}/api/providers`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    providersData = data.providers || {};

    // Set active provider
    if (data.active) {
      providerSelect.value = data.active;
    }

    // Populate model dropdown for current provider
    populateModels(providerSelect.value);

    // Set active model if one was set
    if (data.active_model) {
      modelSelect.value = data.active_model;
    }
  } catch (e) {
    console.warn('Failed to load providers:', e);
  }
}

function populateModels(providerName) {
  const info = providersData[providerName];
  modelSelect.innerHTML = '<option value="">(Provider Default)</option>';

  if (info && info.models) {
    info.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      if (m === info.strong_model) opt.textContent += ' (strong)';
      if (m === info.fast_model) opt.textContent += ' (fast)';
      modelSelect.appendChild(opt);
    });
  }
}

function onProviderChange() {
  populateModels(providerSelect.value);
  switchProvider();
}

async function switchProvider() {
  const body = { name: providerSelect.value };
  if (modelSelect.value) body.model = modelSelect.value;

  try {
    const res = await fetch(`${API}/api/providers/active`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
  } catch (e) {
    showToast(`Provider switch failed: ${e.message}`);
  }
}

// ─── Sample context ─────────────────────────────────────────────────────
function loadSampleContext() {
  tccInput.value = `This modern play unfolds in Lima, a dynamic cityscape where tradition grapples with progress, and prosperity resides alongside hardship. This vibrant city is more than just a setting—it's a character in its own right, offering a rich canvas of diverse experiences that profoundly affect our protagonists.

At the heart of this narrative is Alejandro, a young man with ambitions to radically transform the field of philosophy. In his journey, he's accompanied by a motley crew of individuals, each contributing to his understanding of life's paradoxes. This includes his pragmatic mother Rosa, street-smart friend Carlos, cynical Professor Mendoza, and the captivating Maria who kindles his affections. Additionally, we encounter the influential businessman Don Julio, his superficial daughter Lucia, Pedro the street vendor-cum-philosophical comrade, Inspector Gomez who unexpectedly crosses paths with Alejandro, and a Combi driver who plays a crucial role in the play's tragicomic climax.

This play's ultimate aim is to highlight the absurdity of overreaching ambition, reminding us of the value in appreciating life's simple joys. We journey through a maze of contrasting societal strata, philosophical debates, romantic entanglements, practical anxieties, and chance encounters. Each narrative thread underscores the irony and unpredictability of life, leading to an ending that encapsulates the essence of tragicomedy.`;

  charDescInput.value = 'Alejandro — a young philosophy student burning with the desire to revolutionize modern logic, yet blind to the beauty of everyday life around him';
  sceneDescInput.value = 'A dimly lit cafe in Miraflores, late evening. Rain patters against the windows. The smell of coffee mixes with cigarette smoke. A worn copy of Wittgenstein lies open on the table.';
}

// ─── Trope preview ──────────────────────────────────────────────────────
async function fetchTropePreview() {
  try {
    const res = await fetch(`${API}/api/tropes/random?n=5`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.tropes && data.tropes.length) {
      tropePreview.innerHTML = data.tropes
        .map(t => `<div><strong>${escapeHtml(t.name)}</strong>: ${escapeHtml(t.description.slice(0, 100))}...</div>`)
        .join('');
      tropePreview.classList.remove('hidden');
    }
  } catch (e) {
    tropePreview.classList.add('hidden');
    console.warn('Failed to fetch tropes:', e);
  }
}

function onTropeToggle() {
  if (injectTropes.checked) {
    fetchTropePreview();
  } else {
    tropePreview.classList.add('hidden');
  }
}

// ─── Start arena ────────────────────────────────────────────────────────
async function startArena() {
  const tccContext = tccInput.value.trim();
  const charDesc = charDescInput.value.trim();
  const sceneDesc = sceneDescInput.value.trim();

  if (!tccContext || !charDesc || !sceneDesc) {
    showStatus('Please fill in all fields.', 'error');
    showToast('Please fill in all fields.');
    return;
  }

  startBtn.disabled = true;
  showStatus('<span class="spinner"></span> Generating character and entering scene...', 'loading');

  try {
    // Switch provider if changed
    await switchProvider();

    const res = await fetch(`${API}/api/arena/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tcc_context: tccContext,
        character_description: charDesc,
        scene_description: sceneDesc,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;
    characterName = data.character_name || 'Character';

    // Transition to game panel
    setupPanel.classList.add('hidden');
    gamePanel.classList.remove('hidden');
    charNameDisp.textContent = characterName;
    sceneLabel.textContent = sceneDesc.slice(0, 80) + (sceneDesc.length > 80 ? '...' : '');

    // Render profile and auto-show it
    renderProfile(data.character);
    profileSidebar.classList.remove('hidden');
    profileToggle.classList.add('active');
    profileToggle.textContent = 'Hide Profile';

    // System message
    addMessage('system', null, `${characterName} materializes before you. The scene is set.`);

    userInput.focus();
  } catch (e) {
    showStatus(e.message, 'error');
    showToast(`Failed to start arena: ${e.message}`);
  } finally {
    startBtn.disabled = false;
  }
}

// ─── Chat ───────────────────────────────────────────────────────────────
async function sendMessage() {
  const msg = userInput.value.trim();
  if (!msg || !sessionId) return;

  userInput.value = '';
  sendBtn.disabled = true;

  addMessage('user', 'You', msg);

  // Typing indicator
  const typingId = addMessage('system', null, `${characterName} is responding...`);

  try {
    const res = await fetch(`${API}/api/arena/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: msg }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    removeMessage(typingId);
    addMessage('character', characterName, data.response);
  } catch (e) {
    removeMessage(typingId);
    addMessage('error-msg', null, `Error: ${e.message}`);
    showToast(`Chat error: ${e.message}`);
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}

function onInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

// ─── Messages ───────────────────────────────────────────────────────────
let msgCounter = 0;

function addMessage(type, sender, text) {
  const id = `msg-${++msgCounter}`;
  const div = document.createElement('div');
  div.id = id;
  div.className = `message ${type}`;

  let html = '';
  if (sender) {
    html += `<div class="msg-sender">${escapeHtml(sender)}</div>`;
  }

  // Format character responses: (action) dialogue
  if (type === 'character') {
    html += `<div class="msg-text">${formatCharacterText(text)}</div>`;
  } else {
    html += `<div class="msg-text">${escapeHtml(text)}</div>`;
  }

  div.innerHTML = html;
  chatMessages.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return id;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function formatCharacterText(text) {
  // Highlight (action descriptions) in italics
  return escapeHtml(text).replace(
    /\(([^)]+)\)/g,
    '<span class="msg-action">($1)</span>'
  );
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ─── Profile ────────────────────────────────────────────────────────────
function renderProfile(character) {
  if (!character) {
    profileContent.innerHTML = '<div class="profile-field-value empty">No character data available.</div>';
    return;
  }

  const fields = [
    { label: 'Name',                   value: character.name },
    { label: 'Internal State',         value: character.internal_state },
    { label: 'Ambitions',              value: character.ambitions },
    { label: 'Teleology',              value: character.teleology },
    { label: 'Philosophy',             value: character.philosophy },
    { label: 'Physical State',         value: character.physical_state },
    { label: 'Long-Term Memory',       value: character.long_term_memory, list: true },
    { label: 'Short-Term Memory',      value: character.short_term_memory, list: true },
    { label: 'Internal Contradictions', value: character.internal_contradictions, list: true },
  ];

  profileContent.innerHTML = fields
    .map(f => {
      let valueHtml;
      const isEmpty = !f.value || (Array.isArray(f.value) && f.value.length === 0);

      if (isEmpty) {
        valueHtml = `<div class="profile-field-value empty">(not generated)</div>`;
      } else if (f.list && Array.isArray(f.value)) {
        valueHtml = `<ul class="profile-field-list">${
          f.value.map(v => `<li>${escapeHtml(v)}</li>`).join('')
        }</ul>`;
      } else {
        valueHtml = `<div class="profile-field-value">${escapeHtml(String(f.value))}</div>`;
      }
      return `<div class="profile-field">
        <div class="profile-field-label">${escapeHtml(f.label)}</div>
        ${valueHtml}
      </div>`;
    })
    .join('');
}

function toggleProfile() {
  const isHidden = profileSidebar.classList.toggle('hidden');
  profileToggle.textContent = isHidden ? 'Profile' : 'Hide Profile';
  profileToggle.classList.toggle('active', !isHidden);
}

// ─── World Inspector ────────────────────────────────────────────────────
async function toggleWorld() {
  const isHidden = worldSidebar.classList.toggle('hidden');
  worldToggle.textContent = isHidden ? 'World' : 'Hide World';
  worldToggle.classList.toggle('active', !isHidden);

  if (!isHidden && sessionId) {
    // Fetch fresh world details
    worldContent.innerHTML = '<span class="spinner"></span> Loading world details...';
    try {
      const res = await fetch(`${API}/api/arena/sessions/${sessionId}/world`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderWorldDetails(data);
    } catch (e) {
      worldContent.innerHTML = `<div class="profile-field-value empty">Failed to load: ${escapeHtml(e.message)}</div>`;
      showToast(`Failed to load world details: ${e.message}`);
    }
  }
}

function renderWorldDetails(data) {
  const sections = [
    { title: 'TCC Context (World)', content: data.tcc_context },
    { title: 'Scene Description', content: data.scene_description },
    { title: 'Character Name', content: data.character_name },
  ];

  // Character fields
  if (data.character) {
    const charFields = [
      'internal_state', 'ambitions', 'teleology', 'philosophy', 'physical_state',
    ];
    charFields.forEach(f => {
      if (data.character[f]) {
        const label = f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        sections.push({ title: `Character: ${label}`, content: data.character[f] });
      }
    });

    const listFields = ['long_term_memory', 'short_term_memory', 'internal_contradictions'];
    listFields.forEach(f => {
      if (data.character[f] && data.character[f].length) {
        const label = f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        sections.push({
          title: `Character: ${label}`,
          content: data.character[f].map((v, i) => `${i + 1}. ${v}`).join('\n'),
        });
      }
    });
  }

  // System prompt (the full embodiment prompt sent to the LLM)
  sections.push({ title: 'System Prompt (Embodiment)', content: data.system_prompt });

  // Conversation history
  if (data.history && data.history.length) {
    const historyText = data.history
      .map(m => `[${m.role}] ${m.content}`)
      .join('\n\n');
    sections.push({ title: `Conversation History (${data.history.length} messages)`, content: historyText });
  }

  worldContent.innerHTML = sections
    .map(s => `<div class="world-section">
      <div class="world-section-title">${escapeHtml(s.title)}</div>
      <div class="world-section-content">${escapeHtml(s.content || '(empty)')}</div>
    </div>`)
    .join('');
}

// ─── Reset ──────────────────────────────────────────────────────────────
function resetToSetup() {
  // End current session
  if (sessionId) {
    fetch(`${API}/api/arena/sessions/${sessionId}`, { method: 'DELETE' }).catch(() => {});
  }

  sessionId = null;
  characterName = '';
  chatMessages.innerHTML = '';
  profileContent.innerHTML = '';
  worldContent.innerHTML = '';
  profileSidebar.classList.add('hidden');
  worldSidebar.classList.add('hidden');
  profileToggle.classList.remove('active');
  worldToggle.classList.remove('active');
  profileToggle.textContent = 'Profile';
  worldToggle.textContent = 'World';
  gamePanel.classList.add('hidden');
  setupPanel.classList.remove('hidden');
  hideStatus();
}

// ─── Status helpers ─────────────────────────────────────────────────────
function showStatus(html, type) {
  setupStatus.innerHTML = html;
  setupStatus.className = `status ${type}`;
  setupStatus.classList.remove('hidden');
}

function hideStatus() {
  setupStatus.classList.add('hidden');
}
