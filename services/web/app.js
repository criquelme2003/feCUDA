const form = document.getElementById("effects-form");
const textarea = document.getElementById("tensor");
const statusBox = document.getElementById("status");
const resultsSection = document.getElementById("results");
const summaryBox = document.getElementById("summary");
const tabsBox = document.getElementById("graph-tabs");
const graphsContainer = document.getElementById("graphs-container");
const tablesContainer = document.getElementById("tables-container");
const dictionaryContainer = document.getElementById("dictionary-container");
const dictionaryWrapper = document.getElementById("dictionary-table-wrapper");
const submitBtn = document.getElementById("submit-btn");
const backendInput = document.getElementById("backend-url");
const sampleBtn = document.getElementById("use-sample");

const DEFAULT_TENSOR = [
  [
    [1.0, 0.75, 0.6],
    [0.8, 0.9, 0.55]
  ],
  [
    [0.65, 0.7, 0.85],
    [0.4, 0.95, 0.5]
  ]
];

textarea.value = JSON.stringify(DEFAULT_TENSOR, null, 2);

const SAMPLE_RESPONSE = {
  effects: [
    { order: 0, row: 0, path: [0, 0, 4, 1], value: 0.5 },
    { order: 0, row: 1, path: [0, 0, 5, 2], value: 0.32 },
    { order: 0, row: 2, path: [0, 0, 6, 3], value: 0.41 },
    { order: 0, row: 3, path: [0, 1, 7, 4], value: 0.54 },
    { order: 0, row: 4, path: [1, 0, 2, 5], value: 0.48 },
    { order: 1, row: 0, path: [0, 0, 5, 2, 9], value: 0.39 },
    { order: 1, row: 1, path: [0, 1, 7, 4, 12], value: 0.45 },
    { order: 1, row: 2, path: [1, 0, 2, 5, 11], value: 0.51 },
    { order: 2, row: 0, path: [0, 0, 5, 2, 9, 13], value: 0.31 }
  ],
  total_entries: 9,
  metrics: {
    total_processing_ms: 18.42,
    algorithm_ms: 15.73,
    bootstrap_ms: 0.0,
    bootstrap_replicas: 0,
    gpu_memory_free_before_mb: 992.5,
    gpu_memory_free_after_mb: 990.3,
    gpu_memory_delta_mb: 2.2
  }
};

const storedBackend = localStorage.getItem("effects-backend-url");
backendInput.value = storedBackend || "http://localhost:8000/effects";

sampleBtn.addEventListener("click", () => {
  renderResults(SAMPLE_RESPONSE.effects, SAMPLE_RESPONSE.metrics);
  showStatus("Mostrando datos de ejemplo sin contactar al servicio.", false);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  let tensor;
  try {
    tensor = JSON.parse(textarea.value);
    if (!Array.isArray(tensor)) {
      throw new Error("El tensor debe ser un arreglo tridimensional.");
    }
  } catch (err) {
    showStatus(`Error al interpretar el tensor: ${err.message}`, true);
    return;
  }

  const payload = {
    tensor,
    threshold: parseFloat(form.threshold.value),
    order: parseInt(form.order.value, 10),
    bootstrap_replicas: parseInt(form.bootstrap.value, 10) || 0
  };

  const endpoint = backendInput.value.trim();

  submitBtn.disabled = true;

  if (!endpoint) {
    renderResults(SAMPLE_RESPONSE.effects, SAMPLE_RESPONSE.metrics);
    showStatus("Mostrando datos de ejemplo (sin servicio configurado).", false);
    submitBtn.disabled = false;
    return;
  }

  localStorage.setItem("effects-backend-url", endpoint);
  showStatus("Procesando en GPU...", false);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || response.statusText);
    }

    const data = await response.json();
    renderResults(data.effects, data.metrics);
    const total = data.total_entries ?? (data.effects ? data.effects.length : 0);
    showStatus(`Se generaron ${total} efectos.`, false);
  } catch (error) {
    showStatus(`Error: ${error.message}`, true);
    resultsSection.classList.add("hidden");
  } finally {
    submitBtn.disabled = false;
  }
});

function showStatus(message, isError) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? "#e11d48" : "inherit";
}

function renderResults(effects, metrics) {
  resultsSection.classList.remove("hidden");
  summaryBox.innerHTML = "";
  tabsBox.innerHTML = "";
  graphsContainer.innerHTML = "";
  tablesContainer.innerHTML = "";
  dictionaryWrapper.innerHTML = "";

  const byOrder = new Map();
  for (const entry of effects) {
    if (!byOrder.has(entry.order)) {
      byOrder.set(entry.order, []);
    }
    byOrder.get(entry.order).push(entry);
  }

  const sortedOrders = Array.from(byOrder.keys()).sort((a, b) => a - b);

  if (metrics) {
    const metricChips = [
      ["Total", `${metrics.total_processing_ms.toFixed(2)} ms`],
      ["Algoritmo", `${metrics.algorithm_ms.toFixed(2)} ms`],
      ["Bootstrap", `${metrics.bootstrap_ms.toFixed(2)} ms`],
      ["GPU Δ", `${metrics.gpu_memory_delta_mb.toFixed(2)} MB`],
      ["Replicas", `${metrics.bootstrap_replicas}`]
    ];
    metricChips.forEach(([label, value]) => {
      const chip = document.createElement("span");
      chip.className = "summary-chip";
      chip.textContent = `${label}: ${value}`;
      summaryBox.appendChild(chip);
    });
  }

  for (const order of sortedOrders) {
    const items = byOrder.get(order);
    const chip = document.createElement("span");
    chip.className = "summary-chip";
    chip.textContent = `Orden ${order}: ${items.length} caminos`;
    summaryBox.appendChild(chip);
  }

  sortedOrders.forEach((order, index) => {
    const tab = document.createElement("button");
    tab.className = "tab-button";
    tab.textContent = `Orden ${order}`;
    tab.dataset.order = order;
    tab.addEventListener("click", () => setActiveOrder(order));
    if (index === 0) tab.classList.add("active");
    tabsBox.appendChild(tab);

    const panel = document.createElement("section");
    panel.className = "graph-panel";
    panel.id = `panel-order-${order}`;
    if (index === 0) panel.classList.add("active");

    const meta = document.createElement("p");
    meta.className = "graph-meta";
    meta.textContent = `Caminos encontrados: ${byOrder.get(order).length}`;
    panel.appendChild(meta);

    const graphDiv = document.createElement("div");
    graphDiv.className = "graph-container";
    graphDiv.id = `graph-order-${order}`;
    panel.appendChild(graphDiv);

    graphsContainer.appendChild(panel);

    const orderEntries = byOrder.get(order);
    renderGraph(graphDiv.id, orderEntries);
    renderTable(order, orderEntries);
  });

  if (sortedOrders.length > 0) {
    setActiveOrder(sortedOrders[0]);
  }

  renderDictionary(effects);
}

function setActiveOrder(order) {
  document.querySelectorAll(".tab-button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.order === String(order));
  });

  document.querySelectorAll(".graph-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-order-${order}`);
  });

  document.querySelectorAll(`.order-table`).forEach((table) => {
    table.classList.toggle("hidden", table.dataset.order !== String(order));
  });
}

function renderGraph(containerId, entries) {
  const nodes = new Map();
  const edges = [];

  const addNode = (key, label, group) => {
    if (!nodes.has(key)) {
      nodes.set(key, { id: key, label, group });
    }
  };

  entries.forEach((entry) => {
    const { path } = entry;
    if (!Array.isArray(path) || path.length < 4) return;

    const seq = [];
    const batch = path[0];
    const row = path[1];
    const intermediate = path.slice(2, -1);
    const col = path[path.length - 1];

    const batchKey = `batch-${batch}`;
    addNode(batchKey, `Batch ${batch}`, "batch");
    seq.push(batchKey);

    const rowKey = `row-${batch}-${row}`;
    addNode(rowKey, `Row ${row}`, "row");
    seq.push(rowKey);

    intermediate.forEach((mid, idx) => {
      const key = `mid-${batch}-${entry.order}-${idx}-${mid}`;
      addNode(key, `I${idx + 1}: ${mid}`, "mid");
      seq.push(key);
    });

    const colKey = `col-${batch}-${col}`;
    addNode(colKey, `Col ${col}`, "col");
    seq.push(colKey);

    for (let i = 0; i < seq.length - 1; i++) {
      edges.push({
        from: seq[i],
        to: seq[i + 1],
        arrows: "to",
        color: { color: "#2563eb" }
      });
    }
  });

  const container = document.getElementById(containerId);
  const networkData = {
    nodes: new vis.DataSet(Array.from(nodes.values())),
    edges: new vis.DataSet(edges)
  };

  const options = {
    layout: {
      hierarchical: {
        enabled: true,
        levelSeparation: 130,
        nodeSpacing: 150,
        direction: "LR",
        sortMethod: "hubsize"
      }
    },
    physics: false,
    nodes: {
      shape: "box",
      widthConstraint: { maximum: 120 },
      font: { size: 14 }
    },
    edges: {
      smooth: true
    }
  };

  new vis.Network(container, networkData, options);
}

function renderTable(order, entries) {
  const wrapper = document.createElement("div");
  wrapper.className = "table-wrapper order-table";
  wrapper.dataset.order = order;
  if (order !== 0) wrapper.classList.add("hidden");

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>#</th>
      <th>Camino</th>
      <th>Valor</th>
    </tr>
  `;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  entries.forEach((entry, idx) => {
    const row = document.createElement("tr");
    const pathPretty = entry.path.join(" → ");
    row.innerHTML = `
      <td>${idx + 1}</td>
      <td><span class="badge">${pathPretty}</span></td>
      <td><span class="badge value">${entry.value.toFixed(6)}</span></td>
    `;
    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  wrapper.appendChild(table);
  tablesContainer.appendChild(wrapper);
}

function renderDictionary(effects) {
  if (!effects || effects.length === 0) {
    dictionaryContainer.classList.add("hidden");
    dictionaryWrapper.innerHTML = "";
    return;
  }

  dictionaryContainer.classList.remove("hidden");
  dictionaryWrapper.innerHTML = "";

  const sorted = [...effects].sort((a, b) => {
    if (a.order !== b.order) return a.order - b.order;
    const [batchA, rowA] = a.path;
    const [batchB, rowB] = b.path;
    if (batchA !== batchB) return batchA - batchB;
    if (rowA !== rowB) return rowA - rowB;
    const colA = a.path[a.path.length - 1];
    const colB = b.path[b.path.length - 1];
    if (colA !== colB) return colA - colB;
    return b.value - a.value;
  });

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>#</th>
      <th>Orden</th>
      <th>Batch</th>
      <th>Origen</th>
      <th>Destino</th>
      <th>Valor</th>
    </tr>
  `;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  sorted.forEach((entry, idx) => {
    const path = entry.path || [];
    if (path.length < 4) return;
    const batch = path[0];
    const origin = path[1];
    const destination = path[path.length - 1];

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${idx + 1}</td>
      <td>${entry.order}</td>
      <td>${batch}</td>
      <td><span class="badge">Batch ${batch} · Row ${origin}</span></td>
      <td><span class="badge">Batch ${batch} · Col ${destination}</span></td>
      <td><span class="badge value">${entry.value.toFixed(6)}</span></td>
    `;
    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  dictionaryWrapper.appendChild(table);
}
