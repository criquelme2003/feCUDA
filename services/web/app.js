const form = document.getElementById("effects-form");
const tensorInput = document.getElementById("tensor");
const backendInput = document.getElementById("backend-url");
const submitBtn = document.getElementById("submit-btn");
const sampleBtn = document.getElementById("use-sample");
const statusBox = document.getElementById("status");
const resultsSection = document.getElementById("results");
const summaryBox = document.getElementById("summary");
const tableBody = document.querySelector("#effects-table tbody");

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

const SAMPLE_RESPONSE = {
  effects: [
    { order: 0, row: 0, path: [0, 0, 4, 1], value: 0.5 },
    { order: 0, row: 1, path: [0, 0, 5, 2], value: 0.32 },
    { order: 1, row: 0, path: [0, 1, 7, 4, 12], value: 0.45 },
    { order: 2, row: 0, path: [1, 0, 2, 5, 11, 14], value: 0.31 }
  ],
  total_entries: 4,
  summary: {
    total_processing_ms: 12.5,
    algorithm_ms: 11.1,
    bootstrap_ms: 0.0,
    bootstrap_replicas: 0
  },
  notes: "Ejemplo local sin GPU"
};

tensorInput.value = JSON.stringify(DEFAULT_TENSOR, null, 2);
const storedBackend = localStorage.getItem("effects-backend-url");
backendInput.value = storedBackend || "http://localhost:8000/effects";

sampleBtn.addEventListener("click", () => {
  renderResults(SAMPLE_RESPONSE.effects, SAMPLE_RESPONSE.summary, SAMPLE_RESPONSE.notes);
  showStatus("Mostrando datos de ejemplo locales.", false);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  let tensor;
  try {
    tensor = JSON.parse(tensorInput.value);
    if (!Array.isArray(tensor)) {
      throw new Error("El tensor debe ser un arreglo tridimensional.");
    }
  } catch (error) {
    showStatus(`Tensor inválido: ${error.message}`, true);
    return;
  }

  const payload = {
    tensor,
    threshold: Number(form.threshold.value),
    order: Number(form.order.value),
    bootstrap_replicas: Number(form.bootstrap.value) || 0
  };

  const endpoint = backendInput.value.trim();
  submitBtn.disabled = true;

  if (!endpoint) {
    renderResults(SAMPLE_RESPONSE.effects, SAMPLE_RESPONSE.summary, SAMPLE_RESPONSE.notes);
    showStatus("No se configuró backend, usando ejemplo.", false);
    submitBtn.disabled = false;
    return;
  }

  localStorage.setItem("effects-backend-url", endpoint);
  showStatus("Procesando en GPU…", false);

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
    renderResults(data.effects, data.summary, data.notes);
    const total = data.total_entries ?? data.effects?.length ?? 0;
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
  statusBox.classList.toggle("error", Boolean(isError));
}

function renderResults(effects = [], summary, notes) {
  resultsSection.classList.remove("hidden");
  summaryBox.innerHTML = "";
  tableBody.innerHTML = "";

  if (summary) {
    const chips = [
      `Total ${summary.total_processing_ms.toFixed(2)} ms`,
      `Kernel ${summary.algorithm_ms.toFixed(2)} ms`,
      `Bootstrap ${summary.bootstrap_ms.toFixed(2)} ms`,
      `Réplicas ${summary.bootstrap_replicas}`
    ];
    chips.forEach((label) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = label;
      summaryBox.appendChild(chip);
    });
  }

  if (notes) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = notes;
    summaryBox.appendChild(note);
  }

  effects.forEach((effect, index) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${effect.order}</td>
      <td>${Array.isArray(effect.path) ? effect.path.join(" → ") : "-"}</td>
      <td>${Number(effect.value).toFixed(6)}</td>
    `;
    tableBody.appendChild(row);
  });
}
