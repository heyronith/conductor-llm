/**
 * CCPT Research Article Interactive Visualizations & Data Hydration Engine
 * Vanilla JS, Zero External Dependencies.
 *
 * ALL scientific numbers rendered across the manuscript, tables, and SVG charts
 * are hydrated dynamically from 'data/ccpt-results.json'.
 */

(function () {
  'use strict';

  let CCPT_DATA = null;

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    try {
      const resp = await fetch('data/ccpt-results.json');
      if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
      CCPT_DATA = await resp.json();

      hydrateProseMetrics(CCPT_DATA);
      initArchitectureExplorer();
      initPersistenceExplorer();
      initControllerExplorer();
    } catch (err) {
      console.warn('Could not load data/ccpt-results.json. Fallback SVGs remain visible.', err);
    }
  }

  /* =========================================================================
     0. PROSE DATA HYDRATION (Eliminates Hardcoded Numbers in HTML)
     ========================================================================= */
  function hydrateProseMetrics(data) {
    const tA = data.behavior.table_a_primary_persistence;
    const agg = data.behavior.aggregate_summary;
    const exp = data.experiment;
    const sens = data.ablations.model_c_active_vs_off;
    const drift = data.mechanistic.model_c_drift;

    const s1 = tA["20260821"];
    const s2 = tA["20260823"];
    const s3 = tA["20260824"];

    const s1_sens = sens["20260821"];
    const s2_sens = sens["20260823"];
    const s3_sens = sens["20260824"];

    const s1_l4 = drift["20260821"].layer_4;
    const s2_l4 = drift["20260823"].layer_4;
    const s3_l4 = drift["20260824"].layer_4;

    const bindings = {
      // Model Parameters
      "model_c_total_params": `${(exp.models_compared.model_c.total_parameters / 1e6).toFixed(1)}M`,
      "model_c_cap_params": `${(exp.models_compared.model_c.capability_parameters / 1e6).toFixed(1)}M`,
      "model_c_norm_params": `${(exp.models_compared.model_c.normative_parameters / 1e6).toFixed(1)}M`,
      "model_d_total_params": `${(exp.models_compared.model_d.total_parameters / 1e6).toFixed(1)}M`,
      "model_d_adapter_params": `${(exp.models_compared.model_d.adapter_parameters / 1e6).toFixed(1)}M`,

      // Primary Behavior (Table A)
      "s1_c_pre": `${(s1.c_pre_refusal_rate * 100).toFixed(1)}%`,
      "s1_c_post": `${(s1.c_post_refusal_rate * 100).toFixed(1)}%`,
      "s1_c_delta": `${s1.c_retention_delta_pp.toFixed(2)} pp`,
      "s1_d_pre": `${(s1.d_pre_refusal_rate * 100).toFixed(1)}%`,
      "s1_d_post": `${(s1.d_post_refusal_rate * 100).toFixed(1)}%`,
      "s1_d_delta": `${s1.d_retention_delta_pp.toFixed(2)} pp`,
      "s1_primary_effect": `+${s1.primary_effect_pp.toFixed(2)} percentage points`,

      "s2_c_pre": `${(s2.c_pre_refusal_rate * 100).toFixed(1)}%`,
      "s2_c_post": `${(s2.c_post_refusal_rate * 100).toFixed(1)}%`,
      "s2_c_delta": `${s2.c_retention_delta_pp.toFixed(2)} pp`,
      "s2_d_pre": `${(s2.d_pre_refusal_rate * 100).toFixed(1)}%`,
      "s2_d_post": `${(s2.d_post_refusal_rate * 100).toFixed(1)}%`,
      "s2_d_delta": `${s2.d_retention_delta_pp.toFixed(2)} pp`,
      "s2_primary_effect": `${s2.primary_effect_pp.toFixed(2)} percentage points`,

      "s3_c_pre": `${(s3.c_pre_refusal_rate * 100).toFixed(1)}%`,
      "s3_c_post": `${(s3.c_post_refusal_rate * 100).toFixed(1)}%`,
      "s3_c_delta": `+${s3.c_retention_delta_pp.toFixed(2)} pp`,
      "s3_d_pre": `${(s3.d_pre_refusal_rate * 100).toFixed(1)}%`,
      "s3_d_post": `${(s3.d_post_refusal_rate * 100).toFixed(1)}%`,
      "s3_d_delta": `${s3.d_retention_delta_pp.toFixed(2)} pp`,
      "s3_primary_effect": `+${s3.primary_effect_pp.toFixed(2)} percentage points`,

      // Aggregates
      "mean_primary_effect": `+${agg.mean_primary_effect_pp.toFixed(2)} pp`,
      "sd_primary_effect": `${agg.sd_primary_effect_pp.toFixed(2)} pp`,

      // Benign Ranges
      "post_c_benign_range": `${(agg.min_c_post_benign_rate * 100).toFixed(1)}% to ${(agg.max_c_post_benign_rate * 100).toFixed(1)}%`,
      "post_d_benign_range": `${(agg.min_d_post_benign_rate * 100).toFixed(1)}% to ${(agg.max_d_post_benign_rate * 100).toFixed(1)}%`,

      // Ablations
      "s1_pre_gap": `+${(s1_sens.pre_ablation_gap_determinate * 100).toFixed(2)} pp`,
      "s1_post_gap": `+${(s1_sens.post_ablation_gap_determinate * 100).toFixed(2)} pp`,
      "s1_delta_gap": `+${(s1_sens.ablation_gap_change_determinate * 100).toFixed(2)} pp`,

      "s2_pre_gap": `+${(s2_sens.pre_ablation_gap_determinate * 100).toFixed(2)} pp`,
      "s2_post_gap": `+${(s2_sens.post_ablation_gap_determinate * 100).toFixed(2)} pp`,
      "s2_delta_gap": `${(s2_sens.ablation_gap_change_determinate * 100).toFixed(2)} pp`,

      "s3_pre_gap": `+${(s3_sens.pre_ablation_gap_determinate * 100).toFixed(2)} pp`,
      "s3_post_gap": `+${(s3_sens.post_ablation_gap_determinate * 100).toFixed(2)} pp`,
      "s3_delta_gap": `+${(s3_sens.ablation_gap_change_determinate * 100).toFixed(2)} pp`,

      // Mechanistic
      "s1_l4_gate": s1_l4.gate_absolute_change_mean.toFixed(4),
      "s2_l4_gate": s2_l4.gate_absolute_change_mean.toFixed(4),
      "s3_l4_gate": s3_l4.gate_absolute_change_mean.toFixed(4),
      "s2_l4_norm_cka": s2_l4.normative_linear_cka.toFixed(4),
      "s2_l4_steer_cka": s2_l4.steering_linear_cka.toFixed(4),
    };

    document.querySelectorAll('[data-bind]').forEach(el => {
      const key = el.getAttribute('data-bind');
      if (bindings[key] !== undefined) {
        el.textContent = bindings[key];
      }
    });
  }

  /* =========================================================================
     1. ARCHITECTURE EXPLORER
     ========================================================================= */
  function initArchitectureExplorer() {
    const stage = document.getElementById('arch-stage');
    const controls = document.querySelectorAll('[data-arch-mode]');
    if (!stage || !controls.length || !CCPT_DATA) return;

    const mC = CCPT_DATA.experiment.models_compared.model_c;
    const cParams = (mC.capability_parameters / 1e6).toFixed(1) + 'M';
    const nParams = (mC.normative_parameters / 1e6).toFixed(1) + 'M';

    function renderArch(mode) {
      const isLM = mode === 'lm_training';
      const isSafety = mode === 'safety_training';

      const capStroke = isSafety ? '#94a3b8' : '#2563eb';
      const capFill = isSafety ? '#f8fafc' : '#ffffff';
      const capBadge = isSafety ? 'FROZEN (∇θ_C = 0)' : (isLM ? 'TRAINABLE (∇θ_C > 0)' : 'ACTIVE (Inference)');
      const capBadgeColor = isSafety ? '#64748b' : (isLM ? '#1d4ed8' : '#334155');
      const capBadgeBg = isSafety ? '#f1f5f9' : (isLM ? '#dbeafe' : '#e2e8f0');

      const normStroke = isLM ? '#94a3b8' : '#059669';
      const normFill = isLM ? '#f8fafc' : '#ffffff';
      const normBadge = isLM ? 'FROZEN / DETACHED (∇θ_N = 0)' : (isSafety ? 'TRAINABLE (∇θ_N > 0)' : 'ACTIVE (Inference)');
      const normBadgeColor = isLM ? '#64748b' : (isSafety ? '#047857' : '#334155');
      const normBadgeBg = isLM ? '#f1f5f9' : (isSafety ? '#dcfce7' : '#e2e8f0');

      const obsEdgeOpacity = isLM ? '0.3' : '1.0';
      const ctrlEdgeOpacity = isLM ? '0.2' : '1.0';

      const modeTitle = isLM 
        ? 'Phase 1: Ordinary Language Modeling (Next-Token Cross-Entropy on 1B Tokens)' 
        : (isSafety 
            ? 'Phase 2: Normative Control-Plane Optimization (20M Tokens on WildGuard)' 
            : 'Standard Autoregressive Generation with Normative Interventions');

      const modeDesc = isLM
        ? `Capability parameters (θ_C: ${cParams}) update on general text. Normative parameters (θ_N: ${nParams}) are frozen and untouched by LM loss gradients.`
        : (isSafety
            ? 'Capability parameters (θ_C) are completely frozen. The normative stream reads capability activations via detached observation and learns to steer output via θ_N updates.'
            : 'Capability tokens stream through C while N monitors safety state and applies multiplicative gating & residual steering vectors at controlled layers.');

      stage.innerHTML = `
        <svg viewBox="0 0 900 480" width="100%" height="100%" aria-label="CCPT Architecture Diagram in ${mode} mode">
          <defs>
            <marker id="arr-main" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#475569"/>
            </marker>
            <marker id="arr-obs" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#d97706"/>
            </marker>
            <marker id="arr-ctrl" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#059669"/>
            </marker>
          </defs>

          <!-- Capability Column -->
          <rect x="40" y="30" width="380" height="390" rx="12" fill="#f8fafc" stroke="${isSafety ? '#cbd5e1' : '#bfdbfe'}" stroke-width="2"/>
          <rect x="55" y="45" width="350" height="32" rx="6" fill="#e2e8f0"/>
          <text x="70" y="66" font-size="13" font-weight="700" fill="#1e293b">Capability Stream (θ_C: ${cParams} params)</text>
          
          <rect x="250" y="50" width="145" height="22" rx="4" fill="${capBadgeBg}"/>
          <text x="322" y="65" font-size="10" font-weight="700" fill="${capBadgeColor}" text-anchor="middle">${capBadge}</text>

          <rect x="80" y="95" width="300" height="38" rx="6" fill="${capFill}" stroke="#94a3b8" stroke-width="1.5"/>
          <text x="145" y="119" font-size="12" font-weight="600" fill="#334155">Input Embeddings (512-dim)</text>

          <rect x="80" y="155" width="300" height="48" rx="6" fill="${capFill}" stroke="${capStroke}" stroke-width="${isSafety ? 1.5 : 2.5}"/>
          <text x="110" y="184" font-size="12" font-weight="600" fill="${isSafety ? '#64748b' : '#1d4ed8'}">Capability Layer 1-2 (Causal Attn + MLP)</text>

          <rect x="80" y="235" width="300" height="48" rx="6" fill="${capFill}" stroke="${capStroke}" stroke-width="${isSafety ? 1.5 : 2.5}"/>
          <text x="110" y="264" font-size="12" font-weight="600" fill="${isSafety ? '#64748b' : '#1d4ed8'}">Capability Layer 3-4 (Causal Attn + MLP)</text>

          <rect x="80" y="315" width="300" height="42" rx="6" fill="${capFill}" stroke="#0f172a" stroke-width="1.5"/>
          <text x="145" y="341" font-size="12" font-weight="600" fill="#0f172a">LM Head &amp; Output Logits</text>

          <!-- Normative Column -->
          <rect x="480" y="30" width="380" height="390" rx="12" fill="#f0fdf4" stroke="${isLM ? '#cbd5e1' : '#86efac'}" stroke-width="2"/>
          <rect x="495" y="45" width="350" height="32" rx="6" fill="#dcfce7"/>
          <text x="510" y="66" font-size="13" font-weight="700" fill="#065f46">Normative Control Stream (θ_N: ${nParams} params)</text>
          
          <rect x="670" y="50" width="165" height="22" rx="4" fill="${normBadgeBg}"/>
          <text x="752" y="65" font-size="10" font-weight="700" fill="${normBadgeColor}" text-anchor="middle">${normBadge}</text>

          <rect x="520" y="155" width="300" height="48" rx="6" fill="${normFill}" stroke="${normStroke}" stroke-width="${isLM ? 1.5 : 2.5}"/>
          <text x="555" y="184" font-size="12" font-weight="600" fill="${isLM ? '#64748b' : '#047857'}">Normative Layer 1 (Causal Attn + MLP)</text>

          <rect x="520" y="235" width="300" height="48" rx="6" fill="${normFill}" stroke="${normStroke}" stroke-width="${isLM ? 1.5 : 2.5}"/>
          <text x="555" y="264" font-size="12" font-weight="600" fill="${isLM ? '#64748b' : '#047857'}">Normative Layer 2 + Risk Recognition Head</text>

          <!-- Asymmetric Edges -->
          <g opacity="${obsEdgeOpacity}">
            <path d="M 380 179 L 512 179" fill="none" stroke="#d97706" stroke-width="2.5" stroke-dasharray="4,4" marker-end="url(#arr-obs)"/>
            <rect x="405" y="163" width="86" height="18" rx="4" fill="#fef3c7" stroke="#f59e0b" stroke-width="1"/>
            <text x="410" y="176" font-size="9" font-weight="700" fill="#b45309">stop_gradient(C)</text>
          </g>

          <g opacity="${ctrlEdgeOpacity}">
            <path d="M 520 259 L 388 259" fill="none" stroke="#059669" stroke-width="2.5" marker-end="url(#arr-ctrl)"/>
            <rect x="400" y="243" width="98" height="20" rx="4" fill="#dcfce7" stroke="#10b981" stroke-width="1"/>
            <text x="406" y="257" font-size="9" font-weight="700" fill="#047857">g_l · ΔC_l + s_l (Steer)</text>
          </g>

          <!-- Internal Arrows -->
          <line x1="230" y1="133" x2="230" y2="150" stroke="#475569" stroke-width="1.5" marker-end="url(#arr-main)"/>
          <line x1="230" y1="203" x2="230" y2="230" stroke="#475569" stroke-width="1.5" marker-end="url(#arr-main)"/>
          <line x1="230" y1="283" x2="230" y2="310" stroke="#475569" stroke-width="1.5" marker-end="url(#arr-main)"/>

          <!-- Mode Summary Box -->
          <rect x="40" y="430" width="820" height="42" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
          <text x="55" y="448" font-size="12" font-weight="700" fill="#0f172a">${modeTitle}</text>
          <text x="55" y="463" font-size="11.5" fill="#475569">${modeDesc}</text>
        </svg>
      `;
    }

    controls.forEach(btn => {
      btn.addEventListener('click', () => {
        controls.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderArch(btn.getAttribute('data-arch-mode'));
      });
    });

    renderArch('lm_training');
  }

  /* =========================================================================
     2. THREE-SEED PERSISTENCE EXPLORER
     ========================================================================= */
  function initPersistenceExplorer() {
    const stage = document.getElementById('persistence-stage');
    const metricBtns = document.querySelectorAll('[data-persistence-metric]');
    const dataBox = document.getElementById('persistence-data-box');
    if (!stage || !CCPT_DATA) return;

    let currentMetric = 'harmful';

    function renderPersistence() {
      const tableA = CCPT_DATA.behavior.table_a_primary_persistence;
      const benign = CCPT_DATA.behavior.benign_over_refusal;
      const agg = CCPT_DATA.behavior.aggregate_summary;

      const s1 = tableA["20260821"];
      const s2 = tableA["20260823"];
      const s3 = tableA["20260824"];

      if (currentMetric === 'harmful') {
        stage.innerHTML = `
          <svg viewBox="0 0 820 440" width="100%" height="100%" aria-label="Three-Seed Harmful Refusal Retention Chart">
            <g transform="translate(60, 40)">
              <line x1="40" y1="40" x2="700" y2="40" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="44" font-size="11" fill="#94a3b8" text-anchor="end">+40 pp</text>

              <line x1="40" y1="100" x2="700" y2="100" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="104" font-size="11" fill="#94a3b8" text-anchor="end">+20 pp</text>

              <line x1="40" y1="160" x2="700" y2="160" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>
              <text x="25" y="164" font-size="11" font-weight="600" fill="#475569" text-anchor="end">0 pp</text>

              <line x1="40" y1="220" x2="700" y2="220" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="224" font-size="11" fill="#94a3b8" text-anchor="end">-20 pp</text>

              <line x1="40" y1="280" x2="700" y2="280" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="284" font-size="11" fill="#94a3b8" text-anchor="end">-40 pp</text>

              <line x1="40" y1="20" x2="40" y2="300" stroke="#cbd5e1" stroke-width="1.5"/>

              <!-- Seed 1 -->
              <rect x="90" y="160" width="36" height="${Math.max(2, Math.abs(s1.c_retention_delta_pp) * 3)}" rx="2" fill="#2563eb"/>
              <text x="108" y="152" font-size="10.5" font-weight="600" fill="#2563eb" text-anchor="middle">${s1.c_retention_delta_pp.toFixed(1)} pp</text>

              <rect x="132" y="160" width="36" height="${Math.abs(s1.d_retention_delta_pp) * 3}" rx="2" fill="#94a3b8"/>
              <text x="150" y="298" font-size="10.5" font-weight="600" fill="#64748b" text-anchor="middle">${s1.d_retention_delta_pp.toFixed(1)} pp</text>

              <rect x="85" y="320" width="90" height="24" rx="4" fill="#dcfce7" stroke="#10b981" stroke-width="1"/>
              <text x="130" y="336" font-size="11" font-weight="700" fill="#047857" text-anchor="middle">+${s1.primary_effect_pp.toFixed(2)} pp</text>
              <text x="130" y="362" font-size="12" font-weight="600" fill="#1e293b" text-anchor="middle">Seed 20260821</text>
              <text x="130" y="377" font-size="10.5" fill="#64748b" text-anchor="middle">Pre: ${(s1.c_pre_refusal_rate*100).toFixed(1)}% → ${(s1.c_post_refusal_rate*100).toFixed(1)}%</text>

              <!-- Seed 2 -->
              <rect x="310" y="160" width="36" height="${Math.abs(s2.c_retention_delta_pp) * 3}" rx="2" fill="#2563eb"/>
              <text x="328" y="226" font-size="10.5" font-weight="600" fill="#2563eb" text-anchor="middle">${s2.c_retention_delta_pp.toFixed(1)} pp</text>

              <rect x="352" y="160" width="36" height="${Math.abs(s2.d_retention_delta_pp) * 3}" rx="2" fill="#94a3b8"/>
              <text x="370" y="184" font-size="10.5" font-weight="600" fill="#64748b" text-anchor="middle">${s2.d_retention_delta_pp.toFixed(1)} pp</text>

              <rect x="305" y="320" width="90" height="24" rx="4" fill="#fee2e2" stroke="#ef4444" stroke-width="1"/>
              <text x="350" y="336" font-size="11" font-weight="700" fill="#b91c1c" text-anchor="middle">${s2.primary_effect_pp.toFixed(2)} pp</text>
              <text x="350" y="362" font-size="12" font-weight="600" fill="#1e293b" text-anchor="middle">Seed 20260823</text>
              <text x="350" y="377" font-size="10.5" fill="#64748b" text-anchor="middle">Pre: ${(s2.c_pre_refusal_rate*100).toFixed(1)}% → ${(s2.c_post_refusal_rate*100).toFixed(1)}%</text>

              <!-- Seed 3 -->
              <rect x="530" y="${160 - s3.c_retention_delta_pp * 3}" width="36" height="${s3.c_retention_delta_pp * 3}" rx="2" fill="#2563eb"/>
              <text x="548" y="117" font-size="10.5" font-weight="600" fill="#2563eb" text-anchor="middle">+${s3.c_retention_delta_pp.toFixed(1)} pp</text>

              <rect x="572" y="160" width="36" height="${Math.abs(s3.d_retention_delta_pp) * 3}" rx="2" fill="#94a3b8"/>
              <text x="590" y="202" font-size="10.5" font-weight="600" fill="#64748b" text-anchor="middle">${s3.d_retention_delta_pp.toFixed(1)} pp</text>

              <rect x="525" y="320" width="90" height="24" rx="4" fill="#dcfce7" stroke="#10b981" stroke-width="1"/>
              <text x="570" y="336" font-size="11" font-weight="700" fill="#047857" text-anchor="middle">+${s3.primary_effect_pp.toFixed(2)} pp</text>
              <text x="570" y="362" font-size="12" font-weight="600" fill="#1e293b" text-anchor="middle">Seed 20260824</text>
              <text x="570" y="377" font-size="10.5" fill="#64748b" text-anchor="middle">Pre: ${(s3.c_pre_refusal_rate*100).toFixed(1)}% → ${(s3.c_post_refusal_rate*100).toFixed(1)}%</text>
            </g>

            <g transform="translate(100, 415)">
              <rect x="0" y="0" width="12" height="12" rx="2" fill="#2563eb"/>
              <text x="18" y="10" font-size="11" font-weight="600" fill="#334155">CCPT (Model C) Retention</text>

              <rect x="180" y="0" width="12" height="12" rx="2" fill="#94a3b8"/>
              <text x="198" y="10" font-size="11" font-weight="600" fill="#334155">Frozen-Adapter Control (Model D) Retention</text>

              <rect x="440" y="0" width="12" height="12" rx="2" fill="#dcfce7" stroke="#10b981" stroke-width="1"/>
              <text x="458" y="10" font-size="11" font-weight="600" fill="#047857">Primary C-vs-D Effect (ΔC - ΔD)</text>
            </g>
          </svg>
        `;

        dataBox.innerHTML = `
          <div class="ccpt-stat-box">
            <div class="ccpt-stat-label">Direction Consistency</div>
            <div class="ccpt-stat-value">2 / 3 Seeds</div>
            <div class="ccpt-stat-sub">${agg.direction_consistency}</div>
          </div>
          <div class="ccpt-stat-box">
            <div class="ccpt-stat-label">Mean Primary Effect</div>
            <div class="ccpt-stat-value pos">+${agg.mean_primary_effect_pp.toFixed(2)} pp</div>
            <div class="ccpt-stat-sub">Sample SD: ${agg.sd_primary_effect_pp.toFixed(2)} pp (Heterogeneous)</div>
          </div>
          <div class="ccpt-stat-box">
            <div class="ccpt-stat-label">Model C Mean Retention</div>
            <div class="ccpt-stat-value">${agg.mean_c_retention_pp.toFixed(2)} pp</div>
            <div class="ccpt-stat-sub">vs. Adapter Model D: ${agg.mean_d_retention_pp.toFixed(2)} pp</div>
          </div>
        `;
      } else {
        const s1_b = benign["20260821"];
        const s2_b = benign["20260823"];
        const s3_b = benign["20260824"];

        stage.innerHTML = `
          <svg viewBox="0 0 820 400" width="100%" height="100%" aria-label="Benign Over-Refusal Comparison Chart">
            <g transform="translate(60, 40)">
              <line x1="40" y1="40" x2="700" y2="40" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="44" font-size="11" fill="#94a3b8" text-anchor="end">100%</text>

              <line x1="40" y1="100" x2="700" y2="100" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="104" font-size="11" fill="#94a3b8" text-anchor="end">75%</text>

              <line x1="40" y1="160" x2="700" y2="160" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="164" font-size="11" fill="#94a3b8" text-anchor="end">50%</text>

              <line x1="40" y1="220" x2="700" y2="220" stroke="#f1f5f9" stroke-width="1.5"/>
              <text x="25" y="224" font-size="11" fill="#94a3b8" text-anchor="end">25%</text>

              <line x1="40" y1="280" x2="700" y2="280" stroke="#cbd5e1" stroke-width="1.5"/>
              <text x="25" y="284" font-size="11" fill="#94a3b8" text-anchor="end">0%</text>
              <line x1="40" y1="20" x2="40" y2="280" stroke="#cbd5e1" stroke-width="1.5"/>

              <!-- Seed 1 -->
              <rect x="80" y="${280 - s1_b.model_c_pre_over_refusal_rate * 240}" width="28" height="${s1_b.model_c_pre_over_refusal_rate * 240}" rx="2" fill="#d97706"/>
              <rect x="112" y="${280 - s1_b.model_c_post_over_refusal_rate * 240}" width="28" height="${s1_b.model_c_post_over_refusal_rate * 240}" rx="2" fill="#f59e0b"/>
              <rect x="144" y="${280 - s1_b.model_d_post_over_refusal_rate * 240}" width="28" height="${s1_b.model_d_post_over_refusal_rate * 240}" rx="2" fill="#94a3b8"/>
              <text x="126" y="305" font-size="12" font-weight="600" fill="#1e293b" text-anchor="middle">Seed 20260821</text>
              <text x="126" y="322" font-size="10.5" fill="#64748b" text-anchor="middle">C Post: ${(s1_b.model_c_post_over_refusal_rate * 100).toFixed(1)}%</text>

              <!-- Seed 2 -->
              <rect x="290" y="${280 - s2_b.model_c_pre_over_refusal_rate * 240}" width="28" height="${s2_b.model_c_pre_over_refusal_rate * 240}" rx="2" fill="#d97706"/>
              <rect x="322" y="${280 - s2_b.model_c_post_over_refusal_rate * 240}" width="28" height="${s2_b.model_c_post_over_refusal_rate * 240}" rx="2" fill="#f59e0b"/>
              <rect x="354" y="${280 - s2_b.model_d_post_over_refusal_rate * 240}" width="28" height="${s2_b.model_d_post_over_refusal_rate * 240}" rx="2" fill="#94a3b8"/>
              <text x="336" y="305" font-size="12" font-weight="600" fill="#1e293b" text-anchor="middle">Seed 20260823</text>
              <text x="336" y="322" font-size="10.5" fill="#64748b" text-anchor="middle">C Post: ${(s2_b.model_c_post_over_refusal_rate * 100).toFixed(1)}%</text>

              <!-- Seed 3 -->
              <rect x="500" y="${280 - s3_b.model_c_pre_over_refusal_rate * 240}" width="28" height="${s3_b.model_c_pre_over_refusal_rate * 240}" rx="2" fill="#d97706"/>
              <rect x="532" y="${280 - s3_b.model_c_post_over_refusal_rate * 240}" width="28" height="${s3_b.model_c_post_over_refusal_rate * 240}" rx="2" fill="#f59e0b"/>
              <rect x="564" y="${280 - s3_b.model_d_post_over_refusal_rate * 240}" width="28" height="${s3_b.model_d_post_over_refusal_rate * 240}" rx="2" fill="#94a3b8"/>
              <text x="546" y="305" font-size="12" font-weight="600" fill="#1e293b" text-anchor="middle">Seed 20260824</text>
              <text x="546" y="322" font-size="10.5" fill="#64748b" text-anchor="middle">C Post: ${(s3_b.model_c_post_over_refusal_rate * 100).toFixed(1)}%</text>
            </g>

            <g transform="translate(100, 375)">
              <rect x="0" y="0" width="12" height="12" rx="2" fill="#d97706"/>
              <text x="18" y="10" font-size="11" font-weight="600" fill="#334155">CCPT PRE Over-Refusal</text>

              <rect x="180" y="0" width="12" height="12" rx="2" fill="#f59e0b"/>
              <text x="198" y="10" font-size="11" font-weight="600" fill="#334155">CCPT POST Over-Refusal</text>

              <rect x="360" y="0" width="12" height="12" rx="2" fill="#94a3b8"/>
              <text x="378" y="10" font-size="11" font-weight="600" fill="#334155">Adapter Model D POST Over-Refusal</text>
            </g>
          </svg>
        `;

        dataBox.innerHTML = `
          <div class="ccpt-stat-box">
            <div class="ccpt-stat-label">Post-Persistence Over-Refusal (CCPT)</div>
            <div class="ccpt-stat-value neg">${(agg.min_c_post_benign_rate * 100).toFixed(1)}% – ${(agg.max_c_post_benign_rate * 100).toFixed(1)}%</div>
            <div class="ccpt-stat-sub">Range across the 3 independent seeds</div>
          </div>
          <div class="ccpt-stat-box">
            <div class="ccpt-stat-label">Utility Tradeoff</div>
            <div class="ccpt-stat-value">Substantial</div>
            <div class="ccpt-stat-sub">Safety steering remains coarse at small scale</div>
          </div>
          <div class="ccpt-stat-box">
            <div class="ccpt-stat-label">Model D Post Over-Refusal</div>
            <div class="ccpt-stat-value">${(agg.min_d_post_benign_rate * 100).toFixed(1)}% – ${(agg.max_d_post_benign_rate * 100).toFixed(1)}%</div>
            <div class="ccpt-stat-sub">Shows baseline safety tuning also over-refuses</div>
          </div>
        `;
      }
    }

    metricBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        metricBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentMetric = btn.getAttribute('data-persistence-metric');
        renderPersistence();
      });
    });

    renderPersistence();
  }

  /* =========================================================================
     3. CONTROLLER EFFICACY & ABLATION EXPLORER
     ========================================================================= */
  function initControllerExplorer() {
    const stage = document.getElementById('controller-stage');
    const seedBtns = document.querySelectorAll('[data-controller-seed]');
    const dataBox = document.getElementById('controller-data-box');
    const mechBox = document.getElementById('controller-mech-box');
    if (!stage || !CCPT_DATA) return;

    let currentSeed = "20260823"; // Default to Seed 2

    function renderController() {
      const sens = CCPT_DATA.ablations.model_c_active_vs_off[currentSeed];
      const drift = CCPT_DATA.mechanistic.model_c_drift[currentSeed];

      const preAct = sens.pre_active_rate * 100;
      const preOff = sens.pre_off_rate_determinate * 100;
      const postAct = sens.post_active_rate * 100;
      const postOff = sens.post_off_rate_determinate * 100;

      const preGap = sens.pre_ablation_gap_determinate * 100;
      const postGap = sens.post_ablation_gap_determinate * 100;
      const deltaGap = sens.ablation_gap_change_determinate * 100;

      const isNegative = deltaGap < 0;
      const badgeClass = isNegative ? 'neg' : 'pos';
      const badgeColor = isNegative ? '#b91c1c' : '#047857';

      stage.innerHTML = `
        <svg viewBox="0 0 760 360" width="100%" height="100%" aria-label="Controller Ablation for Seed ${currentSeed}">
          <g transform="translate(60, 30)">
            <line x1="40" y1="20" x2="640" y2="20" stroke="#f1f5f9" stroke-width="1.5"/>
            <text x="25" y="24" font-size="11" fill="#94a3b8" text-anchor="end">100%</text>

            <line x1="40" y1="80" x2="640" y2="80" stroke="#f1f5f9" stroke-width="1.5"/>
            <text x="25" y="84" font-size="11" fill="#94a3b8" text-anchor="end">75%</text>

            <line x1="40" y1="140" x2="640" y2="140" stroke="#f1f5f9" stroke-width="1.5"/>
            <text x="25" y="144" font-size="11" fill="#94a3b8" text-anchor="end">50%</text>

            <line x1="40" y1="200" x2="640" y2="200" stroke="#f1f5f9" stroke-width="1.5"/>
            <text x="25" y="204" font-size="11" fill="#94a3b8" text-anchor="end">25%</text>

            <line x1="40" y1="260" x2="640" y2="260" stroke="#cbd5e1" stroke-width="1.5"/>
            <text x="25" y="264" font-size="11" fill="#94a3b8" text-anchor="end">0%</text>
            <line x1="40" y1="10" x2="40" y2="260" stroke="#cbd5e1" stroke-width="1.5"/>

            <!-- PRE Bars -->
            <rect x="140" y="${260 - preAct * 2.4}" width="38" height="${preAct * 2.4}" rx="3" fill="#1d4ed8"/>
            <text x="159" y="${250 - preAct * 2.4}" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">${preAct.toFixed(1)}%</text>

            <rect x="186" y="${260 - preOff * 2.4}" width="38" height="${preOff * 2.4}" rx="3" fill="#93c5fd"/>
            <text x="205" y="${250 - preOff * 2.4}" font-size="11" font-weight="700" fill="#3b82f6" text-anchor="middle">${preOff.toFixed(1)}%</text>

            <text x="182" y="285" font-size="13" font-weight="700" fill="#1e293b" text-anchor="middle">PRE-PERSISTENCE</text>
            <text x="182" y="302" font-size="11" fill="#047857" text-anchor="middle">Ablation Gap: +${preGap.toFixed(1)} pp</text>

            <!-- POST Bars -->
            <rect x="380" y="${260 - postAct * 2.4}" width="38" height="${postAct * 2.4}" rx="3" fill="#047857"/>
            <text x="399" y="${250 - postAct * 2.4}" font-size="11" font-weight="700" fill="#047857" text-anchor="middle">${postAct.toFixed(1)}%</text>

            <rect x="426" y="${260 - postOff * 2.4}" width="38" height="${postOff * 2.4}" rx="3" fill="#6ee7b7"/>
            <text x="445" y="${250 - postOff * 2.4}" font-size="11" font-weight="700" fill="#059669" text-anchor="middle">${postOff.toFixed(1)}%</text>

            <text x="422" y="285" font-size="13" font-weight="700" fill="#1e293b" text-anchor="middle">POST-PERSISTENCE</text>
            <text x="422" y="302" font-size="11" fill="${badgeColor}" text-anchor="middle">Ablation Gap: +${postGap.toFixed(1)} pp</text>
          </g>

          <g transform="translate(140, 340)">
            <rect x="0" y="0" width="10" height="10" rx="2" fill="#1d4ed8"/>
            <text x="15" y="9" font-size="10" fill="#334155">PRE Active</text>

            <rect x="100" y="0" width="10" height="10" rx="2" fill="#93c5fd"/>
            <text x="115" y="9" font-size="10" fill="#334155">PRE Off (Ablated)</text>

            <rect x="230" y="0" width="10" height="10" rx="2" fill="#047857"/>
            <text x="245" y="9" font-size="10" fill="#334155">POST Active</text>

            <rect x="330" y="0" width="10" height="10" rx="2" fill="#6ee7b7"/>
            <text x="345" y="9" font-size="10" fill="#334155">POST Off (Ablated)</text>
          </g>
        </svg>
      `;

      dataBox.innerHTML = `
        <div class="ccpt-stat-box">
          <div class="ccpt-stat-label">Causal Ablation Gap Shift</div>
          <div class="ccpt-stat-value ${badgeClass}">${deltaGap > 0 ? '+' : ''}${deltaGap.toFixed(2)} pp</div>
          <div class="ccpt-stat-sub">Pre: +${preGap.toFixed(1)} pp → Post: +${postGap.toFixed(1)} pp</div>
        </div>
        <div class="ccpt-stat-box">
          <div class="ccpt-stat-label">NA Sensitivity Range</div>
          <div class="ccpt-stat-value">${sens.sensitivity_a_all_na_refusal.gap_change.toFixed(1)} pp to ${sens.sensitivity_b_all_na_nonrefusal.gap_change.toFixed(1)} pp</div>
          <div class="ccpt-stat-sub">Direction unchanged under extreme bounds</div>
        </div>
        <div class="ccpt-stat-box">
          <div class="ccpt-stat-label">Functional Finding</div>
          <div class="ccpt-stat-value" style="font-size:15px; font-family:var(--font-sans); font-weight:600;">
            ${isNegative ? 'Marked Efficacy Reduction' : 'Efficacy Expansion'}
          </div>
          <div class="ccpt-stat-sub">${isNegative ? 'Controller loses causal grip on capability stream' : 'Controller reinforces steering persistence'}</div>
        </div>
      `;

      const l4 = drift.layer_4;
      const allDrift = CCPT_DATA.mechanistic.model_c_drift;
      mechBox.innerHTML = `
        <div class="ccpt-table-wrap">
          <table class="ccpt-table">
            <thead>
              <tr>
                <th>Diagnostic Metric (Layer 4)</th>
                <th>Value for Seed ${currentSeed}</th>
                <th>Cross-Seed Context</th>
                <th>Scientific Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Gate Absolute Change (Mean)</strong></td>
                <td><code>${l4.gate_absolute_change_mean.toFixed(4)}</code></td>
                <td>Seed 1: ${allDrift["20260821"].layer_4.gate_absolute_change_mean.toFixed(4)} | Seed 3: ${allDrift["20260824"].layer_4.gate_absolute_change_mean.toFixed(4)}</td>
                <td>Elevated in Seed 2 (~2x higher shift in multiplicative gate).</td>
              </tr>
              <tr>
                <td><strong>Linear CKA: Normative State (N)</strong></td>
                <td><code>${l4.normative_linear_cka.toFixed(4)}</code></td>
                <td>Seed 1: ${allDrift["20260821"].layer_4.normative_linear_cka.toFixed(4)} | Seed 3: ${allDrift["20260824"].layer_4.normative_linear_cka.toFixed(4)}</td>
                <td>Higher CKA = greater subspace similarity (Seed 2 did NOT suffer extreme N drift).</td>
              </tr>
              <tr>
                <td><strong>Linear CKA: Steering Vector (s)</strong></td>
                <td><code>${l4.steering_linear_cka.toFixed(4)}</code></td>
                <td>Seed 1: ${allDrift["20260821"].layer_4.steering_linear_cka.toFixed(4)} | Seed 3: ${allDrift["20260824"].layer_4.steering_linear_cka.toFixed(4)}</td>
                <td>Steering subspace remained globally aligned in Seed 2.</td>
              </tr>
              <tr>
                <td><strong>Linear CKA: Capability Proposal (c_tilde)</strong></td>
                <td><code>${l4.capability_linear_cka.toFixed(4)}</code></td>
                <td>Seed 1: ${allDrift["20260821"].layer_4.capability_linear_cka.toFixed(4)} | Seed 3: ${allDrift["20260824"].layer_4.capability_linear_cka.toFixed(4)}</td>
                <td>Capability drift in Seed 2 is comparable to positive seeds.</td>
              </tr>
              <tr>
                <td><strong>Capability Relative L2 Drift</strong></td>
                <td><code>${l4.capability_relative_l2_mean.toFixed(4)}</code></td>
                <td>Seed 1: ${allDrift["20260821"].layer_4.capability_relative_l2_mean.toFixed(4)} | Seed 3: ${allDrift["20260824"].layer_4.capability_relative_l2_mean.toFixed(4)}</td>
                <td>Seed 2 relative L2 is lower than Seed 1, confirming H1 is Inconclusive.</td>
              </tr>
            </tbody>
          </table>
        </div>
      `;
    }

    seedBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        seedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentSeed = btn.getAttribute('data-controller-seed');
        renderController();
      });
    });

    renderController();
  }

})();
