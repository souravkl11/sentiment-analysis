/**
 * Sentix ML — Frontend Application Logic
 * Interacts with FastAPI backend for real-time classical ML inference and inspection.
 */

// Preset Examples
const PRESET_TWEETS = {
  pos: "@VirginAmerica amazing flight to SFO! Friendly crew, mood lighting, free WiFi loved every minute! #bestairline",
  neg: "@united flight delayed for 5 hours in Chicago. Gate staff extremely rude, lost my luggage! Horrible customer service.",
  neu: "@SouthwestAir can you please confirm what time terminal 2 baggage drop opens tomorrow morning at BWI?",
  sarcasm: "@AmericanAir thank you for losing my luggage on the day of my keynote speech! Absolutely stellar service."
};

const BATCH_PRESETS = [
  "@VirginAmerica amazing flight to SFO! Friendly crew and smooth landing. Loved it!",
  "@united flight delayed for 5 hours, gate staff rude, lost my luggage! Terrible service.",
  "@SouthwestAir what time does check-in counter open at Dallas Love Field tomorrow?",
  "@Delta outstanding customer care by attendant Sarah on DL452! Thank you!",
  "@USAirways phone lines busy for 3 hours. Completely unacceptable hold times!",
  "@AmericanAir please confirm if flight AA102 has power outlets in economy.",
  "@JetBlue free high-speed wifi and unlimited snacks! Best airline in the sky."
];

// State
let appState = {
  modelsReady: false,
  metricsData: null,
  activeCmModel: "Linear SVM"
};

// DOM Elements
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initPresets();
  initInference();
  initBatchStudio();
  checkServerHealth();
});

/* ==========================================================================
   Tab Navigation
   ========================================================================== */
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      // Auto-fetch metrics if switching to benchmark tab
      if (targetId === "tab-benchmarks" && !appState.metricsData) {
        loadBenchmarkMetrics();
      }
    });
  });
}

/* ==========================================================================
   Presets & Inputs
   ========================================================================== */
function initPresets() {
  const tweetInput = document.getElementById("tweetInput");
  const charCount = document.getElementById("charCount");

  tweetInput.addEventListener("input", () => {
    charCount.textContent = `${tweetInput.value.length} / 600`;
  });

  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-preset");
      if (PRESET_TWEETS[key]) {
        tweetInput.value = PRESET_TWEETS[key];
        charCount.textContent = `${tweetInput.value.length} / 600`;
        runInference(tweetInput.value);
      }
    });
  });

  document.getElementById("clearBtn").addEventListener("click", () => {
    tweetInput.value = "";
    charCount.textContent = "0 / 600";
    document.getElementById("consensusContainer").style.display = "none";
    document.getElementById("modelsGridSection").style.display = "none";
    document.getElementById("activeFeaturesCard").style.display = "none";
  });

  // Shortcut Ctrl + Enter
  tweetInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runInference(tweetInput.value);
    }
  });

  // Batch presets
  document.getElementById("loadBatchPresetBtn").addEventListener("click", () => {
    const batchInput = document.getElementById("batchInput");
    batchInput.value = BATCH_PRESETS.join("\n");
    updateBatchCount();
  });

  const batchInput = document.getElementById("batchInput");
  batchInput.addEventListener("input", updateBatchCount);
}

function updateBatchCount() {
  const text = document.getElementById("batchInput").value.trim();
  const lines = text ? text.split("\n").filter(l => l.trim().length > 0).length : 0;
  document.getElementById("batchLinesCount").textContent = `${lines} lines detected`;
}

/* ==========================================================================
   Server Health & Readiness
   ========================================================================== */
async function checkServerHealth() {
  const badge = document.getElementById("serverStatusBadge");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Server not ready");
    const data = await res.json();
    if (data.models_ready) {
      appState.modelsReady = true;
      badge.innerHTML = `<span class="status-dot pulsing"></span><span>Models Ready (${data.available_models.length}/4)</span>`;
      badge.style.color = "var(--accent-emerald)";
      badge.style.borderColor = "rgba(16, 185, 129, 0.3)";
      // Load starter sample
      if (!document.getElementById("tweetInput").value) {
        document.getElementById("presetPositive").click();
      }
    } else {
      badge.innerHTML = `<span class="status-dot" style="background:#f59e0b"></span><span>Training / Loading...</span>`;
    }
  } catch (err) {
    badge.innerHTML = `<span class="status-dot" style="background:#f43f5e"></span><span>API Offline</span>`;
    badge.style.color = "var(--accent-rose)";
    badge.style.borderColor = "rgba(244, 63, 94, 0.3)";
  }
}

/* ==========================================================================
   Live Inference Execution
   ========================================================================== */
function initInference() {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const tweetInput = document.getElementById("tweetInput");

  analyzeBtn.addEventListener("click", () => {
    runInference(tweetInput.value);
  });
}

async function runInference(text) {
  if (!text || !text.trim()) {
    alert("Please enter a tweet or select one of the test scenarios!");
    return;
  }

  const analyzeBtn = document.getElementById("analyzeBtn");
  const originalBtnContent = analyzeBtn.innerHTML;
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span>Analyzing...</span>`;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.trim() })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Inference failed");
    }

    const data = await res.json();
    renderInferenceResults(data);
    updatePipelineInspector(data);
  } catch (err) {
    alert(`Prediction Error: ${err.message}`);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = originalBtnContent;
  }
}

function renderInferenceResults(data) {
  const consensus = data.consensus;
  const models = data.models;
  const topFeatures = data.top_features;

  // 1. Consensus Hero Box
  const consensusContainer = document.getElementById("consensusContainer");
  const consensusCard = document.getElementById("consensusCard");
  const verdictEmoji = document.getElementById("verdictEmoji");
  const verdictSentiment = document.getElementById("verdictSentiment");
  const verdictDescription = document.getElementById("verdictDescription");
  const agreementPercentage = document.getElementById("agreementPercentage");
  const agreementFill = document.getElementById("agreementFill");
  const votePills = document.getElementById("votePills");

  consensusCard.className = `card glass-card consensus-card ${consensus.sentiment}`;
  verdictSentiment.textContent = consensus.sentiment.toUpperCase();
  
  const emojis = { positive: "🌟", negative: "⚠️", neutral: "ℹ️" };
  verdictEmoji.textContent = emojis[consensus.sentiment] || "✨";

  if (consensus.unanimous) {
    verdictDescription.textContent = `Complete unanimity: All 4 classical algorithms predicted ${consensus.sentiment.toUpperCase()} sentiment with high certainty.`;
  } else {
    verdictDescription.textContent = `Majority consensus: ${consensus.agreement_percentage}% of algorithms agree on ${consensus.sentiment.toUpperCase()} sentiment polarity.`;
  }

  agreementPercentage.textContent = `${consensus.agreement_percentage}%`;
  agreementFill.style.width = `${consensus.agreement_percentage}%`;

  votePills.innerHTML = Object.entries(consensus.votes).map(([sent, count]) => {
    return `<span class="vote-pill ${sent}">${sent}: ${count}/4</span>`;
  }).join("");

  consensusContainer.style.display = "block";

  // 2. 4-Model Breakdown Cards
  const modelsGrid = document.getElementById("modelsGrid");
  modelsGrid.innerHTML = "";

  const modelMeta = {
    "Multinomial Naive Bayes": { param: "α = 1.0 (Laplace)", code: "mnb" },
    "K-Nearest Neighbors": { param: "k = 5, Metric: Cosine", code: "knn" },
    "Logistic Regression": { param: "L2 Penalty, C = 1.0", code: "lr" },
    "Linear SVM": { param: "Kernel: Linear, C = 1.0", code: "svm" }
  };

  for (const [modelName, info] of Object.entries(models)) {
    const meta = modelMeta[modelName] || { param: "Classical ML", code: "gen" };
    const card = document.createElement("div");
    card.className = "card glass-card model-card";

    const probs = info.probabilities || { positive: 0, neutral: 0, negative: 0 };
    const posPct = Math.round((probs.positive || 0) * 100);
    const neuPct = Math.round((probs.neutral || 0) * 100);
    const negPct = Math.round((probs.negative || 0) * 100);

    card.innerHTML = `
      <div>
        <div class="model-card-top">
          <div>
            <div class="model-name">${modelName}</div>
            <div class="model-type-tag">${meta.param}</div>
          </div>
          <span class="model-sentiment-badge ${info.sentiment}">${info.sentiment}</span>
        </div>

        <div class="model-confidence-row">
          <span class="confidence-lbl">Predicted Confidence:</span>
          <span class="confidence-val">${info.confidence}%</span>
        </div>
      </div>

      <div class="probability-bars">
        <div class="prob-row">
          <span class="prob-label">Positive</span>
          <div class="prob-bar-track">
            <div class="prob-bar-fill pos" style="width: ${posPct}%;"></div>
          </div>
          <span class="prob-num">${posPct}%</span>
        </div>
        <div class="prob-row">
          <span class="prob-label">Neutral</span>
          <div class="prob-bar-track">
            <div class="prob-bar-fill neu" style="width: ${neuPct}%;"></div>
          </div>
          <span class="prob-num">${neuPct}%</span>
        </div>
        <div class="prob-row">
          <span class="prob-label">Negative</span>
          <div class="prob-bar-track">
            <div class="prob-bar-fill neg" style="width: ${negPct}%;"></div>
          </div>
          <span class="prob-num">${negPct}%</span>
        </div>
      </div>
    `;
    modelsGrid.appendChild(card);
  }

  document.getElementById("modelsGridSection").style.display = "block";

  // 3. Top Salient TF-IDF Features
  const activeFeaturesCard = document.getElementById("activeFeaturesCard");
  const featuresList = document.getElementById("featuresList");
  if (topFeatures && topFeatures.length > 0) {
    featuresList.innerHTML = topFeatures.map(item => `
      <div class="feature-tag">
        <span>${item.term}</span>
        <span class="feature-weight">${item.tfidf_weight}</span>
      </div>
    `).join("");
    activeFeaturesCard.style.display = "block";
  } else {
    activeFeaturesCard.style.display = "none";
  }
}

/* ==========================================================================
   NLP Pipeline Inspector
   ========================================================================== */
function updatePipelineInspector(data) {
  const steps = data.pipeline_steps || {};
  document.getElementById("pipeRawText").textContent = steps.original || data.original_text || "";
  document.getElementById("pipeNoiseFiltered").textContent = steps.noise_filtered || "";

  // Lemmatized tokens
  const lemmatizedBox = document.getElementById("pipeLemmatizedTokens");
  const tokens = steps.lemmatized_tokens || [];
  if (tokens.length > 0) {
    lemmatizedBox.innerHTML = tokens.map(t => `<span class="token-pill">${t}</span>`).join("");
  } else {
    lemmatizedBox.innerHTML = `<span class="token-pill placeholder-token">No valid tokens</span>`;
  }

  // Active TF-IDF terms
  const tfidfTermsBox = document.getElementById("pipeTfidfTerms");
  const topF = data.top_features || [];
  if (topF.length > 0) {
    tfidfTermsBox.innerHTML = topF.map(f => `<span class="token-pill">${f.term} (w=${f.tfidf_weight})</span>`).join("");
  } else {
    tfidfTermsBox.innerHTML = `<span class="token-pill placeholder-token">Terms out of 2500 feature vocabulary</span>`;
  }
}

/* ==========================================================================
   Model Benchmarks & Metrics
   ========================================================================== */
async function loadBenchmarkMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error("Could not load metrics");
    const data = await res.json();
    appState.metricsData = data;
    renderBenchmarkScorecard(data);
    renderConfusionMatrix(appState.activeCmModel);
  } catch (err) {
    console.warn("Metrics load note:", err.message);
  }
}

function renderBenchmarkScorecard(data) {
  const metrics = data.metrics || {};
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";

  const metaDict = {
    "Multinomial Naive Bayes": { param: "MultinomialNB(alpha=1.0)" },
    "K-Nearest Neighbors": { param: "KNeighborsClassifier(k=5, metric='cosine')" },
    "Logistic Regression": { param: "LogisticRegression(C=1.0, max_iter=1000)" },
    "Linear SVM": { param: "SVC(kernel='linear', C=1.0)" }
  };

  // Sort by F1 macro descending
  const sortedModels = Object.entries(metrics).sort((a, b) => b[1].f1_macro - a[1].f1_macro);

  if (sortedModels.length > 0) {
    const [topName, topM] = sortedModels[0];
    document.getElementById("kpiTopModel").textContent = topName;
    document.getElementById("kpiTopF1").textContent = (topM.f1_macro * 100).toFixed(1) + "%";
    document.getElementById("kpiTopAcc").textContent = (topM.accuracy * 100).toFixed(1) + "%";
  }

  sortedModels.forEach(([name, m], idx) => {
    const param = metaDict[name] ? metaDict[name].param : "Default";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${name}</strong></td>
      <td><code>${param}</code></td>
      <td>${(m.accuracy * 100).toFixed(2)}%</td>
      <td>${(m.precision_macro * 100).toFixed(2)}%</td>
      <td>${(m.recall_macro * 100).toFixed(2)}%</td>
      <td><strong style="color:var(--accent-emerald)">${(m.f1_macro * 100).toFixed(2)}%</strong></td>
      <td><span class="rank-pill ${idx === 0 ? 'top' : ''}">#${idx + 1}</span></td>
    `;
    tbody.appendChild(tr);
  });

  // Setup CM Selector tabs
  const cmSelector = document.getElementById("cmModelSelector");
  cmSelector.innerHTML = Object.keys(metrics).map(mName => `
    <button class="cm-tab-btn ${mName === appState.activeCmModel ? 'active' : ''}" data-model="${mName}">
      ${mName}
    </button>
  `).join("");

  cmSelector.querySelectorAll(".cm-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      cmSelector.querySelectorAll(".cm-tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      appState.activeCmModel = btn.getAttribute("data-model");
      renderConfusionMatrix(appState.activeCmModel);
    });
  });
}

function renderConfusionMatrix(modelName) {
  if (!appState.metricsData || !appState.metricsData.metrics) return;
  const modelData = appState.metricsData.metrics[modelName];
  if (!modelData) return;

  const labels = appState.metricsData.class_labels || ["negative", "neutral", "positive"];
  const cm = modelData.confusion_matrix;
  const grid = document.getElementById("cmHeatmapGrid");
  grid.innerHTML = "";

  document.getElementById("cmModelTitle").textContent = `${modelName} Classification Metrics`;

  // Render 3x3 cells
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const cell = document.createElement("div");
      const isDiagonal = (r === c);
      cell.className = `cm-cell ${isDiagonal ? 'diagonal' : ''}`;
      cell.innerHTML = `
        <span class="cm-cell-count">${cm[r][c]}</span>
        <span class="cm-cell-label">${labels[r].slice(0,3)}→${labels[c].slice(0,3)}</span>
      `;
      grid.appendChild(cell);
    }
  }

  // Render per-class report details
  const detailsBox = document.getElementById("cmModelDetails");
  const report = modelData.classification_report || {};
  detailsBox.innerHTML = `
    <div><strong>Overall Accuracy:</strong> ${(modelData.accuracy * 100).toFixed(2)}%</div>
    <div><strong>Macro F1-Score:</strong> ${(modelData.f1_macro * 100).toFixed(2)}%</div>
    <div style="margin-top:6px; font-weight:600; color:var(--text-primary)">Per-Class F1 Breakdown:</div>
    <div>• Negative: ${report['negative'] ? (report['negative']['f1-score'] * 100).toFixed(1) + '%' : 'N/A'}</div>
    <div>• Neutral: ${report['neutral'] ? (report['neutral']['f1-score'] * 100).toFixed(1) + '%' : 'N/A'}</div>
    <div>• Positive: ${report['positive'] ? (report['positive']['f1-score'] * 100).toFixed(1) + '%' : 'N/A'}</div>
  `;
}

/* ==========================================================================
   Batch Studio
   ========================================================================== */
function initBatchStudio() {
  const runBtn = document.getElementById("runBatchBtn");
  runBtn.addEventListener("click", runBatchAnalysis);
}

async function runBatchAnalysis() {
  const text = document.getElementById("batchInput").value.trim();
  if (!text) {
    alert("Please enter at least one tweet to analyze!");
    return;
  }

  const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return;

  const runBtn = document.getElementById("runBatchBtn");
  const origText = runBtn.innerHTML;
  runBtn.disabled = true;
  runBtn.innerHTML = `<span>Processing ${lines.length} texts...</span>`;

  try {
    const res = await fetch("/api/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: lines })
    });

    if (!res.ok) throw new Error("Batch processing failed");
    const data = await res.json();
    renderBatchResults(data);
  } catch (err) {
    alert(`Batch Analysis Error: ${err.message}`);
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = origText;
  }
}

function renderBatchResults(data) {
  document.getElementById("batchTotalCount").textContent = data.total_analyzed;
  const dist = data.distribution;

  const posPct = dist.positive ? dist.positive.percentage : 0;
  const neuPct = dist.neutral ? dist.neutral.percentage : 0;
  const negPct = dist.negative ? dist.negative.percentage : 0;

  document.getElementById("batchPosCount").textContent = `${posPct}% (${dist.positive ? dist.positive.count : 0})`;
  document.getElementById("batchNeuCount").textContent = `${neuPct}% (${dist.neutral ? dist.neutral.count : 0})`;
  document.getElementById("batchNegCount").textContent = `${negPct}% (${dist.negative ? dist.negative.count : 0})`;

  document.getElementById("barFillPos").style.width = `${posPct}%`;
  document.getElementById("barFillNeu").style.width = `${neuPct}%`;
  document.getElementById("barFillNeg").style.width = `${negPct}%`;

  const tbody = document.getElementById("batchTableBody");
  tbody.innerHTML = data.results.map((item, idx) => `
    <tr>
      <td>${idx + 1}</td>
      <td style="max-width:320px; font-size:0.85rem;">${item.text}</td>
      <td><span class="model-sentiment-badge ${item.sentiment}">${item.sentiment}</span></td>
      <td><strong>${item.agreement}%</strong></td>
      <td style="font-size:0.78rem; color:var(--text-secondary);">
        MNB: ${item.models["Multinomial Naive Bayes"]} | 
        KNN: ${item.models["K-Nearest Neighbors"]} | 
        LR: ${item.models["Logistic Regression"]} | 
        SVM: ${item.models["Linear SVM"]}
      </td>
    </tr>
  `).join("");

  document.getElementById("batchResultsWrapper").style.display = "block";
}
