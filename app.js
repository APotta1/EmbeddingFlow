/**
 * EmbeddingFlow — Transform embedding process
 * Left: pipeline steps (tokenize → IDs → embedding matrix)
 * Right: matrix visual (rows = tokens, cols = dimensions)
 */

const textInput = document.getElementById('text-input');
const tokensOut = document.getElementById('tokens-out');
const idsOut = document.getElementById('ids-out');
const matrixMeta = document.getElementById('matrix-meta');
const btnRun = document.getElementById('btn-run');
const matrixContainer = document.getElementById('matrix-container');
const matrixPlaceholder = document.getElementById('matrix-placeholder');
const matrixGrid = document.getElementById('matrix-grid');

// Simple tokenizer: split on spaces and punctuation, keep words
function tokenize(text) {
  if (!text || !text.trim()) return [];
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\w\s]/g, ' $& ')
    .split(/\s+/)
    .filter(Boolean);
}

// Deterministic "token ID" from string (simulated vocab)
function tokenToId(token) {
  let h = 0;
  for (let i = 0; i < token.length; i++) {
    h = (h * 31 + token.charCodeAt(i)) >>> 0;
  }
  return h % 1024;
}

// Generate a deterministic embedding vector for a token ID (simulated)
// Returns array of length dim
function embedToken(id, dim = 8) {
  const vec = [];
  for (let d = 0; d < dim; d++) {
    const x = Math.sin(id * 0.1 + d * 0.5) * 0.5 + 0.5;
    vec.push(Math.round(x * 100) / 100);
  }
  return vec;
}

function runPipeline() {
  const text = textInput.value.trim();
  const tokens = tokenize(text);
  const ids = tokens.map(tokenToId);
  const dim = 8;
  const matrix = tokens.map((_, i) => embedToken(ids[i], dim));

  // Update pipeline panel
  tokensOut.textContent = tokens.length ? tokens.join(', ') : '—';
  idsOut.textContent = ids.length ? ids.join(', ') : '—';
  matrixMeta.textContent = matrix.length
    ? `${matrix.length} × ${dim} (tokens × dimensions)`
    : '—';

  // Update matrix visual
  if (!matrix.length) {
    matrixContainer.classList.remove('has-matrix');
    matrixGrid.innerHTML = '';
    return;
  }

  matrixContainer.classList.add('has-matrix');
  renderMatrix(tokens, matrix);
}

function renderMatrix(tokens, matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;

  // Flatten values for heatmap scale
  const allValues = matrix.flat();
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = max - min || 1;

  function heatmapColor(value) {
    const t = (value - min) / range;
    const r = Math.round(125 + (1 - t) * 130);
    const g = Math.round(211 + (1 - t) * 44);
    const b = Math.round(252);
    return `rgb(${r},${g},${b})`;
  }

  matrixGrid.style.gridTemplateColumns = `auto repeat(${cols}, 22px)`;
  matrixGrid.style.gridTemplateRows = `auto repeat(${rows}, 22px)`;
  matrixGrid.innerHTML = '';

  // Corner
  matrixGrid.appendChild(createCell('', 'corner dim-label token-label'));

  // Dimension headers (0 .. dim-1)
  for (let c = 0; c < cols; c++) {
    const cell = createCell(`d${c}`, 'dim-label');
    matrixGrid.appendChild(cell);
  }

  // Rows: token label + values
  for (let r = 0; r < rows; r++) {
    const label = createCell(
      tokens[r].length > 10 ? tokens[r].slice(0, 8) + '…' : tokens[r],
      'token-label'
    );
    matrixGrid.appendChild(label);
    for (let c = 0; c < cols; c++) {
      const value = matrix[r][c];
      const cell = createCell(value.toFixed(2), 'value');
      cell.style.background = heatmapColor(value);
      cell.style.color = value > 0.5 ? 'var(--bg)' : 'var(--text)';
      matrixGrid.appendChild(cell);
    }
  }
}

function createCell(text, className) {
  const cell = document.createElement('div');
  cell.className = 'matrix-cell ' + (className || '');
  cell.textContent = text;
  return cell;
}

btnRun.addEventListener('click', runPipeline);
textInput.addEventListener('input', () => {
  const t = textInput.value.trim();
  if (!t) {
    matrixContainer.classList.remove('has-matrix');
    matrixGrid.innerHTML = '';
  }
});

// Initial run so matrix shows on load
runPipeline();
