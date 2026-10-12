import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Resolve absolute paths based on the location of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_JSON_PATH = os.path.join(ROOT_DIR, "figure_sources", "data", "ecephys-qc-dashboard.json")
OUT_HTML_PATH = os.path.join(ROOT_DIR, "interactive", "openscope_qc_dashboard.html")
OUT_SVG_PATH = os.path.join(ROOT_DIR, "images", "figures", "generated", "openscope_qc_dashboard_static.svg")

# Ensure output directories exist
os.makedirs(os.path.dirname(OUT_HTML_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUT_SVG_PATH), exist_ok=True)

# ==========================================
# 1. GENERATE STATIC SVG PLACEHOLDER (POLISHED + MOUSE SPREAD)
# ==========================================
def generate_static_svg(units_data):
    print("Generating polished 4-panel static SVG counterpart with individual mouse spread...")
    df = pd.DataFrame(units_data)
    
    # Filter to default strict QC rules (excluding 'void')
    qc_df = df[
        (df['isi_violations_ratio'] < 0.01) & 
        (df['presence_ratio'] > 0.90) & 
        (df['amplitude_cutoff'] < 0.10) & 
        (df['snr'] >= 3.0) & 
        (df['depth_from_cortex_um'] >= 0.0) &
        (df['area'] != 'void')
    ]

    # Create 2x2 subplots (Titles handled manually via annotations now)
    fig = make_subplots(
        rows=2, cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.18
    )

    annotations = []

    def add_ecdf(metric, row, col, xref, yref):
        mice = sorted(qc_df['mouse_id'].dropna().unique())
        
        # 1. Plot individual mice as thin, semi-transparent lines
        for mouse in mice:
            mouse_df = qc_df[qc_df['mouse_id'] == mouse]
            first_m = mouse_df[mouse_df['session_stage'] == 'First'][metric].dropna().sort_values()
            last_m = mouse_df[mouse_df['session_stage'] == 'Last'][metric].dropna().sort_values()

            # Handle exact 0s for Log Scales to prevent visual clipping
            if metric == 'isi_violations_ratio':
                first_m = first_m.replace(0, 1e-6)
                last_m = last_m.replace(0, 1e-6)

            if len(first_m) > 0:
                y_f = np.arange(1, len(first_m) + 1) / len(first_m) * 100
                fig.add_trace(go.Scatter(x=first_m, y=y_f, mode='lines', line=dict(color='rgba(44, 160, 44, 0.25)', width=1, shape='hv'), showlegend=False, hoverinfo='skip'), row=row, col=col)
            if len(last_m) > 0:
                y_l = np.arange(1, len(last_m) + 1) / len(last_m) * 100
                fig.add_trace(go.Scatter(x=last_m, y=y_l, mode='lines', line=dict(color='rgba(214, 39, 40, 0.25)', width=1, shape='hv'), showlegend=False, hoverinfo='skip'), row=row, col=col)

        # 2. Plot the aggregate dataset as a thinner solid line on top
        first_vals = qc_df[qc_df['session_stage'] == 'First'][metric].dropna().sort_values()
        last_vals = qc_df[qc_df['session_stage'] == 'Last'][metric].dropna().sort_values()

        if metric == 'isi_violations_ratio':
            first_vals = first_vals.replace(0, 1e-6)
            last_vals = last_vals.replace(0, 1e-6)

        y_first = np.arange(1, len(first_vals) + 1) / len(first_vals) * 100
        y_last = np.arange(1, len(last_vals) + 1) / len(last_vals) * 100
        
        fig.add_trace(go.Scatter(x=first_vals, y=y_first, mode='lines', line=dict(color='#2ca02c', width=2.0, shape='hv'), showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=last_vals, y=y_last, mode='lines', line=dict(color='#d62728', width=2.0, shape='hv'), showlegend=False), row=row, col=col)

        # Add the 'within-plot' legend to the bottom right whitespace
        annotations.append(dict(
            x=0.96, y=0.06, xref=f"{xref} domain", yref=f"{yref} domain",
            text="<span style='color:#2ca02c;'><b>──</b></span> First Session<br><span style='color:#d62728;'><b>──</b></span> Last Session",
            showarrow=False,
            xanchor="right", yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(150, 150, 150, 0.5)",
            borderwidth=1,
            borderpad=6,
            font=dict(size=13, color="#222")
        ))

    # Panel A: SNR (Log Scale)
    add_ecdf('snr', 1, 1, 'x', 'y')
    fig.update_xaxes(title_text="SNR", type="log", tickvals=[1, 3, 10, 30, 100], ticktext=['1', '3', '10', '30', '100'], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative % of Units", range=[0, 105], row=1, col=1)

    # Panel B: ISI Violations (Log Scale)
    add_ecdf('isi_violations_ratio', 1, 2, 'x2', 'y2')
    fig.update_xaxes(title_text="ISI Violations Ratio", type="log", tickvals=[1e-5, 1e-4, 1e-3, 1e-2], ticktext=['10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²'], row=1, col=2)
    fig.update_yaxes(title_text="Cumulative % of Units", range=[0, 105], row=1, col=2)

    # Panel C: Drift (Clipped to 99th percentile to remove extreme outliers)
    add_ecdf('drift_ptp', 2, 1, 'x3', 'y3')
    drift_99 = qc_df['drift_ptp'].quantile(0.99)
    if pd.isna(drift_99) or drift_99 == 0: drift_99 = 100
    fig.update_xaxes(title_text="Peak-to-Peak Drift (µm)", range=[0, drift_99], row=2, col=1)
    fig.update_yaxes(title_text="Cumulative % of Units", range=[0, 105], row=2, col=1)

    # Panel D: Heatmap
    bin_size = 50
    max_depth = int(np.ceil(qc_df['depth_from_cortex_um'].max() / bin_size) * bin_size)
    y_bins = np.arange(0, max_depth + bin_size, bin_size)
    mice = sorted(qc_df['mouse_id'].dropna().unique())

    z_matrix = np.full((len(y_bins) - 1, len(mice)), np.nan)
    for m_idx, mouse in enumerate(mice):
        mouse_df = qc_df[qc_df['mouse_id'] == mouse]
        for b_idx in range(len(y_bins) - 1):
            bin_df = mouse_df[(mouse_df['depth_from_cortex_um'] >= y_bins[b_idx]) & (mouse_df['depth_from_cortex_um'] < y_bins[b_idx+1])]
            if not bin_df.empty:
                z_matrix[b_idx, m_idx] = bin_df['snr'].mean()

    z_flat = z_matrix[~np.isnan(z_matrix)]
    vmax = np.percentile(z_flat, 99) if len(z_flat) > 0 else 25
    y_centers = y_bins[:-1] + (bin_size / 2)

    fig.add_trace(go.Heatmap(
        z=z_matrix, x=mice, y=y_centers,
        colorscale='Viridis',
        zmin=0, zmax=vmax,
        colorbar=dict(
            title=dict(text="Mean SNR", side="right", font=dict(size=16)), 
            x=1.02, y=0.22, len=0.45, thickness=18
        ),
        showscale=True
    ), row=2, col=2)

    fig.update_xaxes(title_text="Mouse ID", type='category', tickangle=-45, row=2, col=2)
    fig.update_yaxes(title_text="Depth from Cortical Entry (µm)", autorange="reversed", row=2, col=2)

    # Global Axis Styling for Bigger Fonts
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

    panels = [
        ("A", "SNR Degradation", "x", "y"),
        ("B", "ISI Violations", "x2", "y2"),
        ("C", "Physical Drift", "x3", "y3"),
        ("D", "SNR Depth Profile", "x4", "y4")
    ]
    
    for letter, title, xref, yref in panels:
        # Panel Letter (Large, Bold, top-left anchored)
        annotations.append(dict(
            x=0, y=1.12, xref=f"{xref} domain", yref=f"{yref} domain",
            text=f"<b>{letter}</b>", showarrow=False,
            font=dict(size=28, color="#111"), xanchor="left", yanchor="bottom"
        ))
        # Panel Title (Centered slightly lower)
        annotations.append(dict(
            x=0.5, y=1.04, xref=f"{xref} domain", yref=f"{yref} domain",
            text=title, showarrow=False,
            font=dict(size=18, color="#222"), xanchor="center", yanchor="bottom"
        ))

    # Final Layout Adjustments
    fig.update_layout(
        template='simple_white',
        font=dict(family="Arial", size=14),
        title=dict(
            text='<b>Neuropixels Unit Quality Control and Stability</b>', 
            font=dict(size=26),
            x=0.01, y=0.98
        ),
        width=1150,
        height=950,
        showlegend=False,
        margin=dict(t=150, b=80, l=90, r=60),
        annotations=annotations
    )
    
    # Add subtle gridlines back to line plots for readability
    for i in range(1, 5):
        axis_name = 'xaxis' if i == 1 else f'xaxis{i}'
        yaxis_name = 'yaxis' if i == 1 else f'yaxis{i}'
        fig.layout[axis_name].showgrid = True
        fig.layout[axis_name].gridcolor = 'rgba(200, 200, 200, 0.3)'
        fig.layout[yaxis_name].showgrid = True
        fig.layout[yaxis_name].gridcolor = 'rgba(200, 200, 200, 0.3)'

    # Save as SVG
    fig.write_image(OUT_SVG_PATH)
    print(f"Static SVG saved to: {OUT_SVG_PATH}")


# ==========================================
# 2. GENERATE INTERACTIVE HTML
# ==========================================
def generate_html(data_dict):
    print("Injecting data into HTML template...")
    
    units_json = json.dumps(data_dict.get("units", []), separators=(',', ':'))
    boundaries_json = json.dumps(data_dict.get("boundaries", []), separators=(',', ':'))
    csd_json = json.dumps(data_dict.get("csd", []), separators=(',', ':'))
    psd_json = json.dumps(data_dict.get("psd", []), separators=(',', ':'))

    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenScope Ecephys Exploratory Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        :root { --accent: #283185; --ink: #303536; --muted: #707674; --line: #d0d4d2; --wash: #f6f7f7; --paper: #fff; --box-bg: #eef0f2; }
        body { font-family: 'Myriad Pro', Arial, sans-serif; margin: 0; padding: 20px; background: var(--wash); color: var(--ink); }
        .viewer { max-width: 1450px; margin: 0 auto; background: var(--paper); border: 1px solid var(--line); border-radius: 5px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .header { border-bottom: 2px solid var(--accent); padding-bottom: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end;}
        h1 { margin: 0; color: var(--accent); font-size: 1.6rem; }
        p.subtitle { margin: 5px 0 0 0; color: var(--muted); font-size: 0.95rem; }

        .layout-grid { display: grid; grid-template-columns: 340px 1fr; gap: 20px; }
        .layout-grid.csd-mode { grid-template-columns: 1fr; }
        .sidebar { background: var(--wash); padding: 15px; border-radius: 5px; border: 1px solid var(--line); height: fit-content; max-height: 85vh; overflow-y: auto;}

        .control-group { display: flex; flex-direction: column; gap: 5px; margin-bottom: 12px; }
        label { font-size: 0.85rem; font-weight: bold; color: var(--ink); }
        select, input[type="number"] { padding: 6px; border: 1px solid #c5c9c7; border-radius: 4px; font-size: 0.85rem; width: 100%; box-sizing: border-box;}
        optgroup { font-weight: bold; color: var(--accent); }
        option { color: var(--ink); font-weight: normal; }

        .qc-panel { background: #fff; border: 1px solid var(--accent); border-radius: 4px; padding: 10px; margin-bottom: 15px; }
        .qc-panel h3 { margin: 0 0 10px 0; font-size: 0.9rem; color: var(--accent); }
        .qc-dynamic-row { display: flex; gap: 5px; margin-bottom: 6px; align-items: center; }
        .qc-dynamic-row select { padding: 4px; font-size: 0.75rem; }
        .qc-dynamic-row select:first-child { flex: 2; }
        .qc-dynamic-row select:nth-child(2) { flex: 1; }
        .qc-dynamic-row input { flex: 1; padding: 4px; font-size: 0.75rem; min-width: 0; }
        .qc-dynamic-row button { background: #d1495b; color: white; border: none; border-radius: 3px; cursor: pointer; padding: 4px 8px; font-weight: bold; }
        .qc-dynamic-row button:hover { background: #a83242; }
        .add-rule-btn { width: 100%; padding: 6px; margin-top: 5px; background: var(--wash); border: 1px dashed var(--accent); color: var(--accent); cursor: pointer; border-radius: 4px; font-weight: bold;}
        .add-rule-btn:hover { background: var(--box-bg); }
        .qc-row-void { display: flex; justify-content: flex-start; gap: 10px; align-items: center; margin-top: 10px; font-size: 0.8rem; border-top: 1px dashed var(--line); padding-top: 10px;}
        .qc-row-void input[type="checkbox"] { transform: scale(1.2); }

        .log-toggle-row { display: flex; gap: 15px; margin-top: 5px; font-size: 0.85rem; align-items: center; font-weight: bold; color: var(--accent); }
        .log-toggle-row input[type="checkbox"] { transform: scale(1.1); margin-right: 4px; }

        .tabs { display: flex; gap: 5px; border-bottom: 1px solid var(--line); margin-bottom: 15px; flex-wrap: wrap;}
        .tab-btn { padding: 8px 15px; background: transparent; border: 1px solid transparent; border-bottom: none; cursor: pointer; font-size: 0.95rem; font-weight: bold; color: var(--muted); }
        .tab-btn.active { color: var(--ink); border: 1px solid var(--line); border-bottom: 1px solid var(--paper); border-radius: 5px 5px 0 0; margin-bottom: -1px; background: var(--paper); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .warning-box { padding: 40px; text-align: center; color: var(--muted); font-size: 1.1rem; border: 2px dashed var(--line); margin-top: 50px; background: #fff;}

        .plot-area { width: 100%; height: 700px; }
        .stats { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--muted); margin-top: 10px; text-align: right; border-top: 1px solid var(--line); padding-top: 10px;}

        hr { border: none; border-top: 1px dashed #ccc; margin: 15px 0; }

        /* CSD Styles */
        .csd-control-grid { display:grid; grid-template-columns:repeat(3,minmax(180px,1fr)); gap:10px 14px; margin:8px 0 6px 0; }
        .csd-control-grid .control-group { margin-bottom:0; }
        .csd-status { padding:9px 11px; margin:10px 0 8px 0; background:#f2f4f5; border-left:3px solid var(--accent); font-size:0.82rem; color:var(--muted); line-height:1.45; }
        .csd-overlay-note { padding:8px 11px; margin:8px 0 10px 0; background:#fafafa; border:1px solid #dedede; border-radius:4px; font-size:0.80rem; line-height:1.45; color:var(--muted); }
        .csd-panel-title { font-weight:bold; margin:14px 0 3px 0; color:var(--ink); }
        .csd-plot { height:640px; width:100%; overflow:visible; }
        .csd-spectrum-plot { height:540px; width:100%; overflow:visible; }
        .csd-band-plot { height:500px; width:100%; overflow:visible; }
    </style>
</head>
<body>
    <div class="viewer">
        <div class="header">
            <div>
                <h1>OpenScope Ecephys Exploratory Dashboard</h1>
                <p class="subtitle">Interactive QC filtering and visualization across ~192k units</p>
            </div>
            <div style="font-size: 0.8rem; color: var(--muted); text-align: right;">
                <b>Tip:</b> Hover over tools in the top-right corner of plots for instructions.<br>Click and drag to zoom. Double-click to reset axes.
            </div>
        </div>

        <div class="layout-grid">
            <div class="sidebar">
                
                <button id="reset-filters-btn" class="add-rule-btn" style="border-color: #d1495b; color: #d1495b; margin-top: 0; margin-bottom: 15px; background: #fff;" onclick="resetAllFilters()">↺ Reset All Filters & Options</button>

                <div class="qc-panel">
                    <h3>Custom QC Filter Builder</h3>
                    <div id="qc-rules-container"></div>
                    <button class="add-rule-btn" onclick="addRule()">+ Add Metric Filter</button>

                    <div class="qc-row-void">
                        <input type="checkbox" id="qc-void" checked onchange="renderAll()">
                        <label for="qc-void">Exclude 'void' (noise/unlabeled areas)</label>
                    </div>
                </div>

                <div class="control-group">
                    <label>Filter Cohort</label>
                    <select id="global-cohort" onchange="updateCascadingFilters()"><option value="All">All Cohorts</option><option value="Motor">Motor Cohort</option><option value="Sequence">Sequence Cohort</option></select>
                </div>

                <div class="control-group" id="context-group">
                    <label>Filter Context</label>
                    <select id="global-context" onchange="updateCascadingFilters()"><option value="All">All Contexts</option><option value="STANDARD">Standard Oddball</option><option value="SENSORYMOTOR">Sensorimotor Mismatch</option><option value="SEQUENCE">Sequence Mismatch</option><option value="DURATION">Duration Mismatch</option></select>
                </div>

                <div class="control-group"><label>Filter Mouse</label><select id="global-mouse" onchange="renderAll()"><option value="All">All Mice</option></select></div>
                
                <div class="control-group">
                    <label>Filter Probe</label>
                    <select id="global-probe" onchange="renderAll()"><option value="All">All Probes</option><option value="ProbeA">Probe A (MOp, MOs, ACA, Hippo)</option><option value="ProbeB">Probe B (VISa, Hippo)</option><option value="ProbeC">Probe C (VISp, Hippo, LGd)</option><option value="ProbeD">Probe D (VISl, Hippo)</option><option value="ProbeE">Probe E (MOp, CP)</option><option value="ProbeF">Probe F (MOs, ACA, PL, ILA)</option></select>
                </div>

                <hr>

                <div class="control-group"><label>Major Division</label><select id="global-division" onchange="updateCascadingFilters()"><option value="All">All Major Divisions</option></select></div>
                <div class="control-group"><label>Region Group</label><select id="global-group" onchange="updateCascadingFilters()"><option value="All">All Region Groups</option></select></div>
                <div class="control-group"><label>Area</label><select id="global-family" onchange="updateCascadingFilters()"><option value="All">All Areas</option></select></div>
                <div class="control-group"><label>Subarea / Subfield</label><select id="global-subregion" onchange="updateCascadingFilters()"><option value="All">All Subareas / Subfields</option></select></div>
                <div class="control-group"><label>Layer / Compartment</label><select id="global-layer" onchange="updateCascadingFilters()"><option value="All">All Layers / Compartments</option></select></div>
                <div class="control-group"><label>Exact Allen CCF Acronym (optional)</label><select id="global-area" onchange="renderAll()"><option value="All">All Exact CCF Acronyms</option></select></div>

                <hr>

                <div class="control-group">
                    <label>Anatomy Overlay (Heatmaps)</label>
                    <select id="boundary-overlay" onchange="renderAll()"><option value="area_layers" selected>Area + layers</option><option value="area_only">Area only</option><option value="off">Off</option></select>
                </div>

                <div class="control-group">
                    <label>Depth Alignment</label>
                    <select id="depth-alignment" onchange="renderAll()"><option value="cortex_entry" selected>Cortical entry (0 = Surface)</option><option value="l1_l23">L1 ↔ L2/3 (0 = L1/L2/3 Boundary)</option></select>
                </div>

                <div class="control-group">
                    <label>Heatmap Color Scale</label>
                    <select id="color-scale-mode" onchange="renderAll()"><option value="shared" selected>Shared across contexts</option><option value="autoscale">Autoscale each context</option></select>
                </div>

                <div class="control-group" id="y-axis-group">
                    <label>Target Metric (Y-Axis / Color)</label>
                    <select id="target-metric" onchange="renderAll()">
                        <optgroup label="Isolation & Contamination">
                            <option value="isi_violations_ratio">ISI Violations Ratio</option>
                            <option value="presence_ratio">Presence Ratio</option>
                            <option value="isolation_distance">Isolation Distance</option>
                            <option value="d_prime">D-Prime</option>
                            <option value="silhouette">Silhouette Score</option>
                            <option value="l_ratio">L-Ratio</option>
                        </optgroup>
                        <optgroup label="Signal Quality & Stability">
                            <option value="snr" selected>Signal-to-Noise Ratio (SNR)</option>
                            <option value="amplitude_cutoff">Amplitude Cutoff</option>
                            <option value="amplitude_cv_median">Amplitude CV Median</option>
                            <option value="drift_ptp">Drift Peak-to-Peak (µm)</option>
                            <option value="drift_std">Drift Std Dev (µm)</option>
                        </optgroup>
                        <optgroup label="Waveform Anatomy">
                            <option value="amplitude">Spike Amplitude (µV)</option>
                            <option value="half_width">Half Width</option>
                            <option value="peak_to_valley">Peak-to-Valley Time</option>
                            <option value="repolarization_slope">Repolarization Slope</option>
                        </optgroup>
                        <optgroup label="Spatial & Activity">
                            <option value="depth_from_cortex_um">Depth from Cortical Entry (µm)</option>
                            <option value="depth_from_l1_l23_um">Depth from L1/L2/3 (µm)</option>
                            <option value="estimated_y">Estimated Y (Probe Depth µm)</option>
                            <option value="firing_rate">Overall Firing Rate (Hz)</option>
                        </optgroup>
                    </select>
                </div>

                <div class="control-group" id="x-axis-group" style="display:none;">
                    <label>X-Axis (Custom Explorer Only)</label>
                    <select id="custom-x" onchange="renderAll()">
                        <optgroup label="Isolation & Contamination">
                            <option value="isi_violations_ratio">ISI Violations Ratio</option>
                            <option value="presence_ratio">Presence Ratio</option>
                            <option value="isolation_distance">Isolation Distance</option>
                            <option value="d_prime">D-Prime</option>
                            <option value="silhouette">Silhouette Score</option>
                            <option value="l_ratio">L-Ratio</option>
                        </optgroup>
                        <optgroup label="Signal Quality & Stability">
                            <option value="snr">Signal-to-Noise Ratio (SNR)</option>
                            <option value="amplitude_cutoff">Amplitude Cutoff</option>
                            <option value="amplitude_cv_median">Amplitude CV Median</option>
                            <option value="drift_ptp">Drift Peak-to-Peak (µm)</option>
                            <option value="drift_std">Drift Std Dev (µm)</option>
                        </optgroup>
                        <optgroup label="Waveform Anatomy">
                            <option value="amplitude">Spike Amplitude (µV)</option>
                            <option value="half_width">Half Width</option>
                            <option value="peak_to_valley">Peak-to-Valley Time</option>
                            <option value="repolarization_slope">Repolarization Slope</option>
                        </optgroup>
                        <optgroup label="Spatial & Activity">
                            <option value="depth_from_cortex_um" selected>Depth from Cortical Entry (µm)</option>
                            <option value="depth_from_l1_l23_um">Depth from L1/L2/3 (µm)</option>
                            <option value="estimated_y">Estimated Y (Probe Depth µm)</option>
                            <option value="firing_rate">Overall Firing Rate (Hz)</option>
                        </optgroup>
                    </select>
                </div>

                <div id="log-toggles-container" style="display:none; margin-bottom: 15px; padding-left: 10px; border-left: 3px solid var(--accent); background-color: var(--box-bg); padding-top: 5px; padding-bottom: 5px;">
                    <div class="log-toggle-row">
                        <label><input type="checkbox" id="log-x-axis" onchange="renderAll()"> Log Scale X</label>
                        <label id="log-y-axis-container" style="display:none; margin-left: 10px;"><input type="checkbox" id="log-y-axis" onchange="renderAll()"> Log Scale Y</label>
                    </div>
                </div>

            </div>

            <div class="main-content">
                <div class="tabs">
                    <button class="tab-btn active" onclick="switchTab(event, 'tab-yield')">Unit Yield</button>
                    <button class="tab-btn" onclick="switchTab(event, 'tab-ecdf')">Degradation (ECDFs)</button>
                    <button class="tab-btn" onclick="switchTab(event, 'tab-depth')">Depth Profile</button>
                    <button class="tab-btn" onclick="switchTab(event, 'tab-layers')">Layer Profile</button>
                    <button class="tab-btn" onclick="switchTab(event, 'tab-timeseries')">Session Time-Series</button>
                    <button class="tab-btn" onclick="switchTab(event, 'tab-custom')">Custom Explorer</button>
                    <button class="tab-btn" onclick="switchTab(event, 'tab-csd')">LFP / CSD</button>
                </div>

                <div id="tab-yield" class="tab-content active">
                    <div id="plot-yield" class="plot-area"></div>
                </div>

                <div id="tab-ecdf" class="tab-content">
                    <div id="plot-ecdf" class="plot-area"></div>
                </div>

                <div id="tab-depth" class="tab-content">
                    <div id="plot-depth" class="plot-area"></div>
                </div>

                <div id="tab-layers" class="tab-content">
                    <div id="plot-layers" class="plot-area"></div>
                </div>

                <div id="tab-timeseries" class="tab-content">
                    <div id="plot-timeseries" class="plot-area"></div>
                </div>

                <div id="tab-custom" class="tab-content">
                    <div id="plot-custom" class="plot-area"></div>
                </div>

                <div id="tab-csd" class="tab-content">
                    <div class="csd-overlay-note"><b>Overlay key:</b> <b>CSD colour:</b> blue = sink (negative), almost-white grey = zero, red = source (positive). The narrow <b>CCF anatomy track</b> immediately to the right of the heatmap is depth-aligned one-to-one with the heatmap: every labelled box occupies the true CCF span of that structure. Very thin spans remain visible as boxes/boundaries but their text is suppressed rather than displaced. Thick black lines = CCF L2/3↔L4 and L4↔L5; black dashed lines = recognised hippocampal stratum boundaries. Blue dash-dot = data-derived earliest strong visual sink, used as a physiological L4 landmark; blue dotted edges/translucent band = heuristic CSD-derived L4 candidate extent. <b>The physiological band is not used to redefine the CCF anatomy.</b></div>
                    <div class="csd-control-grid">
                        <div class="control-group"><label>Session</label><select id="csd-session"></select></div>
                        <div class="control-group"><label>Probe</label><select id="csd-probe"></select></div>
                        <div class="control-group"><label>Stimulus table</label><select id="csd-stim"></select></div>
                        <div class="control-group"><label>Condition</label><select id="csd-condition"></select></div>
                        <div class="control-group"><label>Depth alignment</label><select id="csd-align"><option value="probe">Probe top = 0</option><option value="cortex">Cortical entry = 0</option><option value="l1">L1 ↔ L2/3 = 0</option></select></div>
                        <div class="control-group"><label>View</label><select id="csd-view"><option value="full">Full probe</option><option value="visual">Visual cortex zoom</option><option value="hipp">Hippocampus zoom</option></select></div>
                        <div class="control-group"><label>Spectral band overlay (Heatmap)</label><select id="csd-band"></select></div>
                        <div class="control-group"><label>Band power metric</label><select id="csd-band-metric"><option value="relative" selected>Relative (% of 1–200 Hz)</option><option value="absolute">Absolute (dB mV²)</option></select></div>
                    </div>
                    <div id="csd-status" class="csd-status"></div>
                    <div class="csd-panel-title">Stimulus-triggered LFP</div>
                    <div id="plot-static-lfp" class="plot-area csd-plot"></div>
                    <div class="csd-panel-title">CSD / spatial second derivative</div>
                    <div id="plot-static-csd" class="plot-area csd-plot"></div>

                    <div class="csd-panel-title">Session-sampled LFP spectrum</div>
                    <div class="csd-overlay-note"><b>Spectral interpretation:</b> PSD is estimated from longer LFP windows, preferentially from <code>spontaneous_presentations</code>, rather than from the 400-ms oddball snippets. Band power is plotted against the same CCF-aligned depth/anatomy track. Use <b>relative power</b> to compare spectral composition across depth, or <b>absolute power</b> when asking where LFP activity is strongest. The 65–100 Hz profile is included as a descriptive proxy for high local population activity; it is not a direct firing-rate measurement. Theta/delta are not conditioned on running, sleep, pupil, or other behavioural state.</div>
                    <div id="spectral-status" class="csd-status"></div>
                    <div id="plot-static-psd" class="plot-area csd-spectrum-plot"></div>

                    <div class="csd-panel-title">Selected band power by depth</div>
                    <div style="position: relative;">
                        <div id="plot-static-band" class="plot-area csd-band-plot"></div>
                        <div id="csd-band-checkboxes" style="position: absolute; right: 5px; top: 35px; width: 210px; background: rgba(255,255,255,0.95); border: 1px solid var(--line); border-radius: 4px; padding: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); max-height: 440px; overflow-y: auto; z-index: 10;">
                            <div style="font-weight: bold; margin-bottom: 10px; font-size: 0.9rem; color: var(--accent);">Compare Bands</div>
                            <div id="band-checkbox-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
                        </div>
                    </div>
                </div>

                <div class="stats" id="unit-count">Total Units Displayed: 0</div>
            </div>
        </div>
    </div>

    <script id="unit-data" type="application/json">
{{DATABASE_PAYLOAD}}
    </script>
    <script id="boundary-data" type="application/json">
{{BOUNDARY_PAYLOAD}}
    </script>
    <script id="csd-data" type="application/json">
{{CSD_PAYLOAD}}
    </script>
    <script id="spectral-data" type="application/json">
{{SPECTRAL_PAYLOAD}}
    </script>

    <script>
        let rawData = [];
        try {
            rawData = JSON.parse(document.getElementById("unit-data").textContent);
            console.log("Parsed " + rawData.length + " unit rows.");
        } catch (error) {
            console.error("JSON Parsing Error: ", error);
        }

        let boundaryData = [];
        try {
            boundaryData = JSON.parse(document.getElementById("boundary-data").textContent);
            console.log("Parsed " + boundaryData.length + " CCF boundary records.");
        } catch (error) {
            console.error("Boundary JSON Parsing Error: ", error);
            boundaryData = [];
        }

        let csdData = [];
        try {
            csdData = JSON.parse(document.getElementById("csd-data").textContent);
            console.log("Parsed " + csdData.length + " compact LFP/CSD records.");
        } catch (error) {
            console.error("CSD JSON Parsing Error: ", error);
            csdData = [];
        }

        let spectralData = [];
        try {
            spectralData = JSON.parse(document.getElementById("spectral-data").textContent);
            console.log("Parsed " + spectralData.length + " compact PSD records.");
        } catch (error) {
            console.error("PSD JSON Parsing Error: ", error);
            spectralData = [];
        }

        const corticalLayerOrder = ["1", "2", "2/3", "3", "4", "5", "6a", "6b"];
        const preferredDivisionOrder = ["Isocortex", "Hippocampal formation", "Thalamus", "Basal ganglia", "Other / unclassified"];
        const preferredGroupOrder = ["Visual cortex", "Motor cortex", "Medial frontal cortex", "Retrosplenial cortex", "Somatosensory cortex", "Auditory cortex", "Hippocampus", "Dentate gyrus", "Subicular complex", "Visual thalamus", "Striatum", "Other / unclassified"];
        const preferredFamilyOrder = ["VISp", "VISa", "VISrl", "VISl", "MOp", "MOs", "ACA", "PL", "ILA", "RSP", "Hippocampus", "LGd", "CP"];

        const PLOTLY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

        let currentTab = 'tab-yield';

        const numericMetricNames = {
            'amplitude_cutoff': 'Amp Cutoff', 'isi_violations_ratio': 'ISI Violations',
            'presence_ratio': 'Presence Ratio', 'snr': 'SNR', 'firing_rate': 'Firing Rate (Hz)',
            'isolation_distance': 'Isolation Dist', 'd_prime': 'D-Prime', 'silhouette': 'Silhouette',
            'l_ratio': 'L-Ratio', 'amplitude': 'Amplitude (µV)', 'half_width': 'Half Width (ms)',
            'peak_to_valley': 'Peak-to-Valley (ms)', 'repolarization_slope': 'Repol Slope',
            'drift_ptp': 'Drift PtP (µm)', 'drift_std': 'Drift Std (µm)',
            'amplitude_cv_median': 'Amp CV Median',
            'depth_from_cortex_um': 'Depth from Cortical Entry (µm)',
            'depth_from_l1_l23_um': 'Depth from L1/L2/3 (µm)',
            'estimated_y': 'Est Y (µm)'
        };

        // --- HELPER TO INJECT PSEUDOCOUNTS FOR LOG SCALES ---
        function safeLogVal(metric, val) {
            const logSafe = ['isi_violations_ratio', 'amplitude_cutoff', 'drift_ptp', 'drift_std', 'snr', 'firing_rate'];
            if (val === 0 && logSafe.includes(metric)) {
                return 1e-6;
            }
            return val;
        }

        const BASE_PLOTLY_CONFIG = {
            responsive: true, 
            displayModeBar: true, 
            displaylogo: false
        };

        function enhancePlotlyTooltips() {
            const tooltipMap = {
                'Zoom': 'Zoom (Click and drag to draw a bounding box)',
                'Pan': 'Pan (Click and drag to move the plot)',
                'Zoom in': 'Zoom In (Click to zoom in)',
                'Zoom out': 'Zoom Out (Click to zoom out)',
                'Autoscale': 'Autoscale (Double-click plot to reset)',
                'Reset axes': 'Reset Axes (Double-click plot to reset)'
            };
            document.querySelectorAll('.modebar-btn').forEach(btn => {
                const title = btn.getAttribute('data-title');
                if (tooltipMap[title]) btn.setAttribute('data-title', tooltipMap[title]);
            });
        }

        const _csdDecodedCache = new Map();
        
        const _CSD_HEATMAP_DOMAIN_END = 0.70;
        const _CSD_ANATOMY_X0 = 0.715;
        const _CSD_ANATOMY_X1 = 0.815;
        const _CSD_ANATOMY_LABEL_X = 0.765;
        const _CSD_COLORBAR_X = 0.93;
        
        const _CSD_PLOT_HEIGHT_PX = 580;
        const _CSD_MIN_LABEL_HEIGHT_PX = 17;
        const STATIC_PSD_FMIN = 1.0;
        const STATIC_PSD_FMAX = 200.0;
        const _CSD_MAX_LABELS = 14;

        const PARULA_COLORSCALE = [
            [0.0000, 'rgb(53,42,135)'], [0.0625, 'rgb(50,67,186)'], [0.1250, 'rgb(3,99,225)'],
            [0.1875, 'rgb(13,117,220)'], [0.2500, 'rgb(20,133,212)'], [0.3125, 'rgb(9,152,209)'],
            [0.3750, 'rgb(6,167,198)'], [0.4375, 'rgb(21,177,180)'], [0.5000, 'rgb(56,185,158)'],
            [0.5625, 'rgb(101,190,134)'], [0.6250, 'rgb(146,191,115)'], [0.6875, 'rgb(183,189,100)'],
            [0.7500, 'rgb(217,186,86)'], [0.8125, 'rgb(248,187,68)'], [0.8750, 'rgb(252,206,46)'],
            [0.9375, 'rgb(245,228,29)'], [1.0000, 'rgb(249,251,14)']
        ];

        const _CSD_BLUE_GREY_RED = [
            [0.00, 'rgb(5,48,97)'], [0.12, 'rgb(33,102,172)'], [0.27, 'rgb(103,169,207)'],
            [0.40, 'rgb(209,229,240)'], [0.47, 'rgb(239,243,245)'], [0.50, 'rgb(250,250,250)'],
            [0.53, 'rgb(245,239,238)'], [0.60, 'rgb(253,219,199)'], [0.73, 'rgb(239,138,98)'],
            [0.88, 'rgb(178,24,43)'], [1.00, 'rgb(103,0,31)']
        ];

        const _LFP_BLUE_GREY_RED = [
            [0.00, 'rgb(33,102,172)'], [0.20, 'rgb(103,169,207)'], [0.40, 'rgb(214,230,238)'],
            [0.50, 'rgb(250,250,250)'], [0.60, 'rgb(242,221,215)'], [0.80, 'rgb(214,96,77)'],
            [1.00, 'rgb(178,24,43)']
        ];

        function finiteDepthOrNaN(value) {
            if (value === null || value === undefined || value === '') return NaN;
            const n = Number(value);
            return Number.isFinite(n) ? n : NaN;
        }

        const _allCorticalDepthValues = [];
        const _allL1L23DepthValues = [];

        for(let i=0; i<rawData.length; i++) {
            const c = finiteDepthOrNaN(rawData[i].depth_from_cortex_um);
            const l = finiteDepthOrNaN(rawData[i].depth_from_l1_l23_um);
            if (Number.isFinite(c)) _allCorticalDepthValues.push(c);
            if (Number.isFinite(l)) _allL1L23DepthValues.push(l);
        }

        boundaryData.forEach(b => {
            const c = finiteDepthOrNaN(b.depth_from_cortex_um);
            const l = finiteDepthOrNaN(b.depth_from_l1_l23_um);
            if (Number.isFinite(c)) _allCorticalDepthValues.push(c);
            if (Number.isFinite(l)) _allL1L23DepthValues.push(l);
        });

        let maxCortical = 2000;
        for(let i=0; i<_allCorticalDepthValues.length; i++) {
            if(_allCorticalDepthValues[i] > maxCortical) maxCortical = _allCorticalDepthValues[i];
        }

        const DEPTH_TICK_UM = 250;
        const CORTICAL_DEPTH_MAX_UM = Math.max(DEPTH_TICK_UM, Math.ceil(maxCortical / DEPTH_TICK_UM) * DEPTH_TICK_UM);

        let L1L23_DEPTH_MIN_UM = -500;
        let L1L23_DEPTH_MAX_UM = 2000;
        if (_allL1L23DepthValues.length > 0) {
            let lo = Infinity;
            let hi = -Infinity;
            for(let i=0; i<_allL1L23DepthValues.length; i++) {
                if(_allL1L23DepthValues[i] < lo) lo = _allL1L23DepthValues[i];
                if(_allL1L23DepthValues[i] > hi) hi = _allL1L23DepthValues[i];
            }
            if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
                L1L23_DEPTH_MIN_UM = Math.floor(lo / DEPTH_TICK_UM) * DEPTH_TICK_UM;
                L1L23_DEPTH_MAX_UM = Math.ceil(hi / DEPTH_TICK_UM) * DEPTH_TICK_UM;
                if (L1L23_DEPTH_MAX_UM <= L1L23_DEPTH_MIN_UM) { L1L23_DEPTH_MIN_UM = -500; L1L23_DEPTH_MAX_UM = 2000; }
            }
        }

        function depthAlignmentMode() { return document.getElementById('depth-alignment').value; }
        function unitDisplayDepth(d) { return depthAlignmentMode() === 'l1_l23' ? finiteDepthOrNaN(d.depth_from_l1_l23_um) : finiteDepthOrNaN(d.depth_from_cortex_um); }
        function boundaryDisplayDepth(b) { return depthAlignmentMode() === 'l1_l23' ? finiteDepthOrNaN(b.depth_from_l1_l23_um) : finiteDepthOrNaN(b.depth_from_cortex_um); }

        function depthAxisForMode() {
            if (depthAlignmentMode() === 'l1_l23') {
                return {
                    title: 'Depth from superficial L1 ↔ L2/3 transition (µm; 0 = Boundary)',
                    range: [L1L23_DEPTH_MAX_UM, L1L23_DEPTH_MIN_UM],
                    tickmode: 'linear', tick0: 0, dtick: DEPTH_TICK_UM,
                    zeroline: true, zerolinewidth: 2.2, zerolinecolor: '#111111'
                };
            }
            return {
                title: 'Depth from cortical entry (µm; 0 = Surface)',
                range: [CORTICAL_DEPTH_MAX_UM, 0],
                tickmode: 'linear', tick0: 0, dtick: DEPTH_TICK_UM,
                zeroline: true, zerolinewidth: 1.5, zerolinecolor: '#111111'
            };
        }

        let qcRules = [
            { id: 1, metric: 'isi_violations_ratio', op: '<', value: 0.01 },
            { id: 2, metric: 'presence_ratio', op: '>', value: 0.90 },
            { id: 3, metric: 'amplitude_cutoff', op: '<', value: 0.10 },
            { id: 4, metric: 'snr', op: '>=', value: 3.0 },
            { id: 5, metric: 'depth_from_cortex_um', op: '>=', value: 0.0 }
        ];
        let nextRuleId = 6;

        function renderQCRules() {
            const container = document.getElementById('qc-rules-container');
            container.innerHTML = '';
            qcRules.forEach(rule => {
                const row = document.createElement('div');
                row.className = 'qc-dynamic-row';

                const metricSelect = `<select onchange="updateRule(${rule.id}, 'metric', this.value)">
                    ${Object.keys(numericMetricNames).map(m => `<option value="${m}" ${m===rule.metric?'selected':''}>${numericMetricNames[m]}</option>`).join('')}
                </select>`;

                const opSelect = `<select onchange="updateRule(${rule.id}, 'op', this.value)">
                    <option value="<" ${rule.op==='<'?'selected':''}>&lt;</option>
                    <option value=">" ${rule.op==='>'?'selected':''}>&gt;</option>
                    <option value="<=" ${rule.op==='<='?'selected':''}>&le;</option>
                    <option value=">=" ${rule.op==='>='?'selected':''}>&ge;</option>
                </select>`;

                row.innerHTML = `
                    ${metricSelect}
                    ${opSelect}
                    <input type="number" step="any" value="${rule.value}" onchange="updateRule(${rule.id}, 'value', this.value)">
                    <button title="Remove Filter" onclick="removeRule(${rule.id})">✖</button>
                `;
                container.appendChild(row);
            });
        }

        window.updateRule = function(id, field, value) {
            const rule = qcRules.find(r => r.id === id);
            if (rule) rule[field] = value;
            renderAll();
        };

        window.removeRule = function(id) {
            qcRules = qcRules.filter(r => r.id !== id);
            renderQCRules();
            renderAll();
        };

        window.addRule = function() {
            qcRules.push({ id: nextRuleId++, metric: 'snr', op: '>', value: 3.0 });
            renderQCRules();
            renderAll();
        };

        window.resetAllFilters = function() {
            qcRules = [
                { id: 1, metric: 'isi_violations_ratio', op: '<', value: 0.01 },
                { id: 2, metric: 'presence_ratio', op: '>', value: 0.90 },
                { id: 3, metric: 'amplitude_cutoff', op: '<', value: 0.10 },
                { id: 4, metric: 'snr', op: '>=', value: 3.0 },
                { id: 5, metric: 'depth_from_cortex_um', op: '>=', value: 0.0 }
            ];
            nextRuleId = 6;
            document.getElementById('qc-void').checked = true;
            renderQCRules();

            const selects = [
                'global-cohort', 'global-context', 'global-mouse', 'global-probe',
                'global-division', 'global-group', 'global-family', 'global-subregion',
                'global-layer', 'global-area', 'boundary-overlay', 'depth-alignment',
                'color-scale-mode', 'target-metric', 'custom-x'
            ];
            
            selects.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    if (id === 'boundary-overlay') el.value = 'area_layers';
                    else if (id === 'depth-alignment') el.value = 'cortex_entry';
                    else if (id === 'color-scale-mode') el.value = 'shared';
                    else if (id === 'target-metric') el.value = 'snr';
                    else if (id === 'custom-x') el.value = 'depth_from_cortex_um';
                    else el.value = 'All';
                }
            });
            
            const lx = document.getElementById('log-x-axis');
            if (lx) lx.checked = false;
            const ly = document.getElementById('log-y-axis');
            if (ly) ly.checked = false;

            updateCascadingFilters(); 
        };

        renderQCRules();

        function setSelectOptions(selectId, values, allLabel, preferredOrder=null) {
            const el = document.getElementById(selectId);
            const current = el.value;
            el.innerHTML = `<option value="All">${allLabel}</option>`;
            let arr = [...values].filter(v => v != null && String(v).trim() !== '');
            if (preferredOrder) {
                const rank = new Map(preferredOrder.map((v, i) => [v, i]));
                arr.sort((a,b) => {
                    const ra = rank.has(a) ? rank.get(a) : 999;
                    const rb = rank.has(b) ? rank.get(b) : 999;
                    return ra !== rb ? ra-rb : String(a).localeCompare(String(b));
                });
            } else {
                arr.sort((a,b) => String(a).localeCompare(String(b), undefined, {numeric:true}));
            }
            arr.forEach(v => el.add(new Option(v, v)));
            el.value = arr.includes(current) ? current : 'All';
        }

        function updateCascadingFilters() {
            const cohort = document.getElementById('global-cohort').value;
            const context = document.getElementById('global-context').value;

            const base = rawData.filter(d =>
                (cohort === 'All' || d.cohort === cohort) &&
                (context === 'All' || d.context === context)
            );

            setSelectOptions('global-mouse', new Set(base.map(d => d.mouse_id)), 'All Mice');
            setSelectOptions('global-division', new Set(base.map(d => d.major_division)), 'All Major Divisions', preferredDivisionOrder);

            const division = document.getElementById('global-division').value;
            const divisionData = base.filter(d => division === 'All' || d.major_division === division);
            setSelectOptions('global-group', new Set(divisionData.map(d => d.region_group)), 'All Region Groups', preferredGroupOrder);

            const group = document.getElementById('global-group').value;
            const groupData = divisionData.filter(d => group === 'All' || d.region_group === group);
            setSelectOptions('global-family', new Set(groupData.map(d => d.area_family)), 'All Areas', preferredFamilyOrder);

            const family = document.getElementById('global-family').value;
            const familyData = groupData.filter(d => family === 'All' || d.area_family === family);
            setSelectOptions('global-subregion', new Set(familyData.map(d => d.subarea)), 'All Subareas / Subfields');

            const subregion = document.getElementById('global-subregion').value;
            const subData = familyData.filter(d => subregion === 'All' || d.subarea === subregion);
            setSelectOptions('global-layer', new Set(subData.map(d => d.layer_or_compartment)), 'All Layers / Compartments', corticalLayerOrder);

            const layer = document.getElementById('global-layer').value;
            const layerData = subData.filter(d => layer === 'All' || d.layer_or_compartment === layer);
            setSelectOptions('global-area', new Set(layerData.map(d => d.ccf_acronym || d.area)), 'All Exact CCF Acronyms');

            renderAll();
        }

        function getFilteredData(options = {}) {
            const ignoreContext = Boolean(options.ignoreContext);
            const mouse = document.getElementById('global-mouse').value;
            const cohort = document.getElementById('global-cohort').value;
            const context = document.getElementById('global-context').value;
            const probe = document.getElementById('global-probe').value;
            const division = document.getElementById('global-division').value;
            const group = document.getElementById('global-group').value;
            const family = document.getElementById('global-family').value;
            const subregion = document.getElementById('global-subregion').value;
            const layer = document.getElementById('global-layer').value;
            const area = document.getElementById('global-area').value;
            const excludeVoid = document.getElementById('qc-void').checked;

            return rawData.filter(d => {
                for(let rule of qcRules) {
                    const val = d[rule.metric];
                    if (val == null) return false;
                    const num = parseFloat(rule.value);
                    if (isNaN(num)) continue;

                    if (rule.op === '<' && !(val < num)) return false;
                    if (rule.op === '>' && !(val > num)) return false;
                    if (rule.op === '<=' && !(val <= num)) return false;
                    if (rule.op === '>=' && !(val >= num)) return false;
                }
                if (excludeVoid && d.area === 'void') return false;

                const cleanProbe = probe === 'All' ? 'All' : probe.split(' ')[0];
                return (mouse === 'All' || d.mouse_id === mouse) &&
                       (cohort === 'All' || d.cohort === cohort) &&
                       (ignoreContext || context === 'All' || d.context === context) &&
                       (cleanProbe === 'All' || d.probe === cleanProbe) &&
                       (division === 'All' || d.major_division === division) &&
                       (group === 'All' || d.region_group === group) &&
                       (family === 'All' || d.area_family === family) &&
                       (subregion === 'All' || d.subarea === subregion) &&
                       (layer === 'All' || d.layer_or_compartment === layer) &&
                       (area === 'All' || (d.ccf_acronym || d.area) === area);
            });
        }

        function median(values) {
            if (!values.length) return null;
            const x = [...values].sort((a,b)=>a-b);
            const mid = Math.floor(x.length/2);
            return x.length % 2 ? x[mid] : (x[mid-1] + x[mid]) / 2;
        }

        function robustPercentile(values, q=0.99) {
            const clean = values.map(Number).filter(Number.isFinite).sort((a,b)=>a-b);
            if (!clean.length) return 1;
            const idx = Math.min(clean.length - 1, Math.max(0, Math.floor((clean.length - 1) * q)));
            const v = clean[idx];
            return (Number.isFinite(v) && v > 0) ? v : 1;
        }

        function getColorScaleMode() { return document.getElementById('color-scale-mode').value; }
        function scaleModeLabel() { return getColorScaleMode() === 'shared' ? 'shared across contexts' : 'autoscaled to context'; }

        function sharedScalarMetricVmax(metric, fallbackValues=[]) {
            if (getColorScaleMode() !== 'shared') return robustPercentile(fallbackValues, 0.99);
            const reference = getFilteredData({ignoreContext: true});
            const vals = reference.map(d => Number(d[metric])).filter(Number.isFinite);
            return robustPercentile(vals.length ? vals : fallbackValues, 0.99);
        }

        function sharedTimeSeriesVmax(fallbackValues=[]) {
            if (getColorScaleMode() !== 'shared') return robustPercentile(fallbackValues, 0.98);
            const reference = getFilteredData({ignoreContext: true});
            const vals = [];
            reference.forEach(d => {
                if (Array.isArray(d.fr_time_series)) {
                    d.fr_time_series.forEach(v => { const n = Number(v); if (Number.isFinite(n)) vals.push(n); });
                }
            });
            return robustPercentile(vals.length ? vals : fallbackValues, 0.98);
        }

        function anatomyOverlayMode() { return document.getElementById('boundary-overlay').value; }
        function isLayerBoundary(b) { return String(b.boundary_type || '').toLowerCase() === 'layer'; }

        function boundariesForDisplay(bounds, multiMouse=false) {
            const mode = anatomyOverlayMode();
            if (mode === 'off') return [];
            if (mode === 'area_only' || multiMouse) return bounds.filter(b => !isLayerBoundary(b));
            return bounds;
        }

        function anatomyBoundaryLineStyle(b) {
            const layer = isLayerBoundary(b);
            return layer
                ? { color: 'rgba(115,115,115,0.42)', width: 1.0, dash: 'dash' }
                : { color: 'rgba(75,75,75,0.68)', width: 1.8, dash: 'dash' };
        }

        function relevantBoundaries(mouse=null) {
            if (anatomyOverlayMode() === 'off') return [];
            const probe = document.getElementById('global-probe').value;
            if (probe === 'All') return [];
            const cohort = document.getElementById('global-cohort').value;
            const context = document.getElementById('global-context').value;
            return boundaryData.filter(b => {
                return b.probe === probe &&
                       (mouse == null || b.mouse_id === mouse) &&
                       (cohort === 'All' || b.cohort === cohort) &&
                       (context === 'All' || b.context === context);
            });
        }

        function aggregateBoundaryRecords(records) {
            const groups = new Map();
            records.forEach(b => {
                const key = `${b.boundary_type}|${b.boundary_label}|${b.from_area}|${b.to_area}`;
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(b);
            });
            return [...groups.entries()].map(([key, rows]) => {
                const r0 = rows[0];
                return {
                    depth_from_cortex_um: median(rows.map(r => Number(r.depth_from_cortex_um)).filter(Number.isFinite)),
                    depth_from_l1_l23_um: median(rows.map(r => Number(r.depth_from_l1_l23_um)).filter(Number.isFinite)),
                    boundary_type: r0.boundary_type,
                    boundary_label: r0.boundary_label,
                    from_area: r0.from_area, to_area: r0.to_area,
                    n_sessions: new Set(rows.map(r => r.session_key)).size,
                    state_from: { major_division: r0.from_major_division, region_group: r0.from_region_group, area_family: r0.from_area_family, subarea: r0.from_subarea, layer: r0.from_layer, compartment: r0.from_compartment },
                    state_to: { major_division: r0.to_major_division, region_group: r0.to_region_group, area_family: r0.to_area_family, subarea: r0.to_subarea, layer: r0.to_layer, compartment: r0.to_compartment }
                };
            }).filter(b => (depthAlignmentMode() === 'l1_l23' ? b.depth_from_l1_l23_um : b.depth_from_cortex_um) != null).sort((a,b) => (depthAlignmentMode() === 'l1_l23' ? a.depth_from_l1_l23_um - b.depth_from_l1_l23_um : a.depth_from_cortex_um - b.depth_from_cortex_um));
        }

        function compactAreaName(state) {
            if (!state) return '';
            const all = `${state.major_division || ''} ${state.region_group || ''} ${state.area_family || ''} ${state.subarea || ''}`.toLowerCase();
            if (all.includes('hippocamp') || all.includes('dentate') || all.includes('subiculum')) return 'Hippocampus';
            if (all.includes('thalam') || all.includes('lgd')) return 'Thalamus';
            if (all.includes('striat') || all.includes('caudoputamen') || /(^|\s)cp($|\s)/.test(all)) return 'Striatum';
            if (all.includes('isocortex') || all.includes('visual cortex') || all.includes('motor cortex') || all.includes('retrosplenial')) return 'Cortex';
            if (all.includes('midbrain') || all.includes('superior colliculus')) return 'Midbrain';
            if (all.includes('white matter') || all.includes('fiber tract')) return 'White matter';
            return '';
        }

        function compactLayerName(state) {
            if (!state) return '';
            if (state.layer) return `L${String(state.layer).trim()}`;
            return state.compartment ? String(state.compartment).trim() : '';
        }

        function mergeAdjacentSegments(segments, labelFn) {
            const out = [];
            segments.forEach(seg => {
                const label = labelFn(seg.state);
                if (!label) return;
                const prev = out[out.length - 1];
                if (prev && prev.label === label && Math.abs(prev.high - seg.low) < 1e-6) {
                    prev.high = seg.high;
                } else {
                    out.push({low: seg.low, high: seg.high, label});
                }
            });
            return out;
        }

        function boundsForCurrentDepthMode(bounds) {
            return bounds.map(b => ({...b, display_depth: boundaryDisplayDepth(b)}))
                .filter(b => Number.isFinite(Number(b.display_depth)))
                .sort((a,b) => a.display_depth - b.display_depth);
        }

        function anatomySegmentsFromBounds(bounds, yMin, yMax) {
            if (!bounds.length) return {areaSegments: [], layerSegments: []};
            const sorted = bounds;
            const segments = [];
            let low = yMin;
            let state = sorted[0].state_from;
            sorted.forEach(b => {
                const high = Math.max(yMin, Math.min(yMax, Number(b.display_depth)));
                if (high > low) segments.push({low, high, state});
                low = high;
                state = b.state_to;
            });
            if (low < yMax) segments.push({low, high: yMax, state});

            return {
                areaSegments: mergeAdjacentSegments(segments, compactAreaName),
                layerSegments: mergeAdjacentSegments(segments, compactLayerName)
            };
        }

        function applyPreservedZoom(graphId, layout, currentXKey, currentYKey) {
            const graphDiv = document.getElementById(graphId);

            if (graphDiv && graphDiv.layout &&
                graphDiv.dataset.xKey === currentXKey &&
                graphDiv.dataset.yKey === currentYKey) {

                if (graphDiv.layout.xaxis && graphDiv.layout.xaxis.autorange === false && graphDiv.layout.xaxis.range) {
                    if(!layout.xaxis) layout.xaxis = {};
                    layout.xaxis.range = graphDiv.layout.xaxis.range;
                    layout.xaxis.autorange = false;
                }
                if (graphDiv.layout.yaxis && graphDiv.layout.yaxis.autorange === false && graphDiv.layout.yaxis.range) {
                    if(!layout.yaxis) layout.yaxis = {};
                    layout.yaxis.range = graphDiv.layout.yaxis.range;
                    layout.yaxis.autorange = false;
                }
            }
            if (graphDiv) {
                graphDiv.dataset.xKey = currentXKey;
                graphDiv.dataset.yKey = currentYKey;
            }
        }

        function renderYield() {
            document.getElementById('context-group').style.display = 'flex';
            const data = getFilteredData();
            document.getElementById('unit-count').textContent = `Total Units Displayed: ${data.length.toLocaleString()}`;

            if(data.length === 0) { Plotly.react('plot-yield', [], {title: 'No Data Available'}, BASE_PLOTLY_CONFIG); return; }

            const mice = [...new Set(data.map(d => d.mouse_id))].sort();
            const probes = ["ProbeA", "ProbeB", "ProbeC", "ProbeD", "ProbeE", "ProbeF"];
            const colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00'];

            const traces = probes.map((probe, i) => {
                const counts = mice.map(mouse => data.filter(d => d.mouse_id === mouse && d.probe === probe).length);
                return { x: mice, y: counts, name: probe, type: 'bar', marker: { color: colors[i] } };
            });

            const layout = { title: 'QC-Passed Unit Yield by Mouse and Probe', barmode: 'stack', xaxis: { title: 'Mouse ID', type: 'category' }, yaxis: { title: 'Number of Units' } };
            applyPreservedZoom('plot-yield', layout, 'yield', 'yield');
            Plotly.react('plot-yield', traces, layout, BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());
        }

        function renderECDF() {
            document.getElementById('context-group').style.display = 'none';
            document.getElementById('global-context').value = 'All';

            const data = getFilteredData();
            const metric = document.getElementById('target-metric').value;

            const firstData = data.filter(d => d.session_stage === 'First' && d[metric] != null).map(d => safeLogVal(metric, d[metric])).sort((a,b)=>a-b);
            const lastData = data.filter(d => d.session_stage === 'Last' && d[metric] != null).map(d => safeLogVal(metric, d[metric])).sort((a,b)=>a-b);

            const buildCumulative = arr => ({ x: arr, y: arr.map((_, i) => (i + 1) / arr.length * 100) });

            const trFirst = buildCumulative(firstData);
            const trLast = buildCumulative(lastData);

            const traces = [
                { x: trFirst.x, y: trFirst.y, mode: 'lines', line: {shape: 'hv', width: 3, color: '#2ca02c'}, name: 'First Session' },
                { x: trLast.x, y: trLast.y, mode: 'lines', line: {shape: 'hv', width: 3, color: '#d62728'}, name: 'Last Session' }
            ];

            const useLogX = document.getElementById('log-x-axis') && document.getElementById('log-x-axis').checked;

            const layout = { 
                title: `Degradation Analysis: ${numericMetricNames[metric]||metric}`, 
                xaxis: { title: numericMetricNames[metric]||metric, type: useLogX ? 'log' : 'linear' }, 
                yaxis: { title: 'Cumulative % of Units', range: [0, 100] }
            };
            
            applyPreservedZoom('plot-ecdf', layout, metric, 'cumulative');
            Plotly.react('plot-ecdf', traces, layout, BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());
            document.getElementById('unit-count').textContent = `Total Units Displayed: ${(firstData.length + lastData.length).toLocaleString()} (First/Last stages only)`;
        }

        function renderDepth() {
            document.getElementById('context-group').style.display = 'flex';
            let spatialMetric = 'depth_from_cortex_um';
            const targetMet = document.getElementById('target-metric').value;
            if (targetMet === 'estimated_y' || targetMet === 'depth_from_cortex_um' || targetMet === 'depth_from_l1_l23_um') spatialMetric = targetMet;
            const isDepthMode = (spatialMetric === 'depth_from_cortex_um' || spatialMetric === 'depth_from_l1_l23_um');

            let data = getFilteredData();
            if (isDepthMode) {
                data = data.map(d => ({...d, _display_depth: unitDisplayDepth(d)})).filter(d => Number.isFinite(d._display_depth));
            } else {
                data = data.filter(d => d[spatialMetric] != null);
            }
            document.getElementById('unit-count').textContent = `Total Units Displayed: ${data.length.toLocaleString()}`;

            if(data.length === 0) {
                const msg = (isDepthMode && depthAlignmentMode() === 'l1_l23')
                    ? 'No valid L1 ↔ L2/3 reference for these filters' : `No ${numericMetricNames[spatialMetric]||spatialMetric} Data Available`;
                Plotly.react('plot-depth', [], {title: msg}, BASE_PLOTLY_CONFIG); return;
            }

            const metric = document.getElementById('target-metric').value;
            const binSize = 50;

            let minDepth, maxDepth;
            if (isDepthMode) {
                if (depthAlignmentMode() === 'l1_l23') { minDepth = L1L23_DEPTH_MIN_UM; maxDepth = L1L23_DEPTH_MAX_UM; }
                else { minDepth = 0; maxDepth = CORTICAL_DEPTH_MAX_UM; }
            } else {
                let minVal = Infinity, maxVal = -Infinity;
                for(let i=0; i<data.length; i++) {
                    const v = Number(data[i][spatialMetric]);
                    if(Number.isFinite(v)) {
                        if(v < minVal) minVal = v;
                        if(v > maxVal) maxVal = v;
                    }
                }
                minDepth = Math.floor(minVal / binSize) * binSize;
                maxDepth = Math.ceil(maxVal / binSize) * binSize;
            }

            const yBins = [];
            for(let d = minDepth; d <= maxDepth; d += binSize) { yBins.push(d); }
            if (yBins.length < 2) { Plotly.react('plot-depth', [], {title: 'Not enough spatial variation'}, BASE_PLOTLY_CONFIG); return; }

            const mice = [...new Set(data.map(d => d.mouse_id))].sort();
            const zMatrix = Array(yBins.length - 1).fill(0).map(() => Array(mice.length).fill(null));

            mice.forEach((mouse, m_idx) => {
                const mouseData = data.filter(d => d.mouse_id === mouse && d[metric] != null);
                for(let b_idx = 0; b_idx < yBins.length - 1; b_idx++) {
                    const binData = mouseData.filter(d => (isDepthMode ? d._display_depth : d[spatialMetric]) >= yBins[b_idx] && (isDepthMode ? d._display_depth : d[spatialMetric]) < yBins[b_idx+1]);
                    if(binData.length > 0) zMatrix[b_idx][m_idx] = binData.reduce((sum, d) => sum + d[metric], 0) / binData.length;
                }
            });

            const hoverText = zMatrix.map((row, b_idx) => {
                return row.map((val, m_idx) => {
                    const v = val !== null ? val.toFixed(3) : 'No Data';
                    return `Mouse: ${mice[m_idx]}<br>${isDepthMode ? (depthAlignmentMode() === 'l1_l23' ? 'Depth from L1/L2/3' : 'Depth from Surface') : (numericMetricNames[spatialMetric]||spatialMetric)}: ${yBins[b_idx]}–${yBins[b_idx+1]} µm<br>Mean ${numericMetricNames[metric]||metric}: ${v}`;
                });
            });

            const flatZ = [];
            for(let i=0; i<zMatrix.length; i++) {
                for(let j=0; j<zMatrix[i].length; j++) {
                    if(zMatrix[i][j] !== null) flatZ.push(zMatrix[i][j]);
                }
            }
            const vmax = sharedScalarMetricVmax(metric, flatZ);

            const trace = {
                z: zMatrix, x: mice, y: yBins.slice(0, -1).map(y => y + binSize/2),
                type: 'heatmap', colorscale: PARULA_COLORSCALE,
                zmin: 0, zmax: vmax, hoverongaps: false,
                text: hoverText, hoverinfo: 'text',
                colorbar: { title: {text: `Mean ${numericMetricNames[metric]||metric} · ${scaleModeLabel()}`, side: 'bottom'}, orientation: 'h', x: 0.5, y: -0.18, yanchor: 'top', len: 0.62, thickness: 14 }
            };

            const layout = { title: `Depth Profile Heatmap: ${numericMetricNames[metric]||metric}`, xaxis: { title: 'Mouse ID', type: 'category' }, yaxis: isDepthMode ? depthAxisForMode() : { title: numericMetricNames[spatialMetric]||spatialMetric, autorange: 'reversed' }};

            if (isDepthMode && document.getElementById('global-probe').value !== 'All') {
                layout.shapes = []; layout.annotations = [];
                mice.forEach(mouse => {
                    const allBoundsRaw = aggregateBoundaryRecords(relevantBoundaries(mouse));
                    const allBounds = boundsForCurrentDepthMode(allBoundsRaw);
                    const bounds = boundariesForDisplay(allBounds, mice.length > 1);
                    if (!bounds.length) return;

                    if (mice.length === 1) {
                        bounds.forEach(b => {
                            layout.shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: b.display_depth, y1: b.display_depth, line: anatomyBoundaryLineStyle(b), layer: 'above' });
                        });
                        const anatomy = anatomySegmentsFromBounds(allBounds, minDepth, maxDepth);
                        layout.annotations.push({ xref: 'paper', x: 1.025, yref: 'paper', y: 1.02, text: '<b>Area</b>', showarrow: false, xanchor: 'left', font: {size: 11, color: '#555'} });
                        if (anatomyOverlayMode() === 'area_layers') layout.annotations.push({ xref: 'paper', x: 1.145, yref: 'paper', y: 1.02, text: '<b>Layer</b>', showarrow: false, xanchor: 'left', font: {size: 11, color: '#555'} });

                        anatomy.areaSegments.forEach(seg => {
                            if ((seg.high - seg.low) < 35) return;
                            layout.annotations.push({ xref: 'paper', x: 1.025, yref: 'y', y: (seg.low + seg.high)/2, text: `<b>${seg.label}</b>`, showarrow: false, xanchor: 'left', yanchor: 'middle', font: {size: 11, color: '#444'}, bgcolor: 'rgba(255,255,255,0.92)', bordercolor: '#ddd', borderwidth: 0.5 });
                        });
                        if (anatomyOverlayMode() === 'area_layers') {
                            anatomy.layerSegments.forEach(seg => {
                                if ((seg.high - seg.low) < 35) return;
                                layout.annotations.push({ xref: 'paper', x: 1.145, yref: 'y', y: (seg.low + seg.high)/2, text: seg.label, showarrow: false, xanchor: 'left', yanchor: 'middle', font: {size: 10, color: '#666'}, bgcolor: 'rgba(255,255,255,0.90)', bordercolor: '#e3e3e3', borderwidth: 0.5 });
                            });
                        }
                    }
                });
                layout.margin = layout.annotations.length ? { l: 75, r: anatomyOverlayMode() === 'area_layers' ? 190 : 120, t: 85, b: 115 } : { l: 75, r: 40, t: 85, b: 115 };
            } else {
                layout.margin = { l: 75, r: 40, t: 85, b: 115 };
            }

            applyPreservedZoom('plot-depth', layout, 'mouse', spatialMetric);
            Plotly.react('plot-depth', [trace], layout, BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());
        }

        function renderLayerProfile() {
            document.getElementById('context-group').style.display = 'flex';
            const family = document.getElementById('global-family').value;
            const metric = document.getElementById('target-metric').value;
            const plotDiv = document.getElementById('plot-layers');

            if (family === 'All') {
                Plotly.purge(plotDiv);
                plotDiv.innerHTML = '<div class="warning-box"><b>Select an Area.</b><br><br>Use the biological hierarchy in the sidebar (e.g. Isocortex → Visual cortex → VISp).</div>';
                document.getElementById('unit-count').textContent = 'Total Units Displayed: 0';
                return;
            }
            if (plotDiv.querySelector('.warning-box')) plotDiv.innerHTML = '';

            const data = getFilteredData().filter(d => d[metric] != null);
            document.getElementById('unit-count').textContent = `Total Units Displayed: ${data.length.toLocaleString()}`;
            if (!data.length) { Plotly.react('plot-layers', [], {title: 'No data for selected anatomy'}, BASE_PLOTLY_CONFIG); return; }

            const isHippo = family === 'Hippocampus';
            const isLGd = family === 'LGd';
            const hasCorticalLayers = data.some(d => d.anatomy_class === 'cortical' && d.layer != null && String(d.layer).trim() !== '');
            let xOrder = [];
            let xField = 'layer';
            let xTitle = 'Cortical layer';
            let xFormatter = x => `L${x}`;

            if (isHippo) {
                xField = 'subarea';
                xTitle = 'Hippocampal CA subfield';
                xFormatter = x => x;
                const present = [...new Set(data.map(d => d.subarea).filter(x => x != null))];
                const hipOrder = ['CA1','CA2','CA3'];
                xOrder = hipOrder.filter(x => present.includes(x)).concat(present.filter(x => !hipOrder.includes(x)).sort());
            } else if (hasCorticalLayers) {
                const present = new Set(data.map(d => d.layer).filter(x => x != null));
                xOrder = corticalLayerOrder.filter(x => present.has(x));
            } else {
                xField = 'subarea';
                xTitle = isLGd ? 'LGd subdivision' : 'Subarea / subfield';
                xFormatter = x => x;
                const present = [...new Set(data.map(d => d.subarea).filter(x => x != null))];
                xOrder = present.sort((a,b)=>String(a).localeCompare(String(b), undefined, {numeric:true}));
            }

            const subareas = [...new Set(data.map(d => d.subarea).filter(x => x != null))].sort();
            const traces = [];
            const grouped = (hasCorticalLayers && subareas.length > 1 && !isHippo) ? subareas : [null];

            grouped.forEach(sub => {
                const subData = sub == null ? data : data.filter(d => d.subarea === sub);
                const means = [];
                const counts = [];
                xOrder.forEach(x => {
                    const rows = subData.filter(d => d[xField] === x && d[metric] != null);
                    counts.push(rows.length);
                    means.push(rows.length ? rows.reduce((acc,d)=>acc + Number(d[metric]),0)/rows.length : null);
                });
                traces.push({
                    x: xOrder.map(xFormatter),
                    y: means,
                    type: 'bar',
                    name: sub == null ? family : sub,
                    text: counts.map(n => `n=${n}`),
                    hovertemplate: `%{x}<br>Mean ${numericMetricNames[metric]||metric}: %{y:.3f}<br>%{text}<extra>${sub == null ? family : sub}</extra>`
                });
            });

            const selectedGroup = document.getElementById('global-group').value;
            const hierarchyPrefix = selectedGroup === 'All' ? family : `${selectedGroup} → ${family}`;
            const profileLevel = isHippo ? 'CA subfield' : (hasCorticalLayers ? 'cortical layer' : 'subarea');

            const layerShapes = [];
            for (let i = 0; i < xOrder.length; i++) {
                if (i % 2 === 0) {
                    layerShapes.push({
                        type: 'rect', xref: 'paper', x0: i / xOrder.length, x1: (i + 1) / xOrder.length,
                        yref: 'paper', y0: 0, y1: 1,
                        fillcolor: 'rgba(0,0,0,0.025)', line: { width: 0 }, layer: 'below'
                    });
                }
                if (i > 0) {
                    layerShapes.push({
                        type: 'line', xref: 'paper', x0: i / xOrder.length, x1: i / xOrder.length,
                        yref: 'paper', y0: 0, y1: 1,
                        line: { color: '#000000', width: 1.5, dash: 'dash' }, layer: 'above'
                    });
                }
            }

            const layerAnnotations = [];
            const selectedMouse = document.getElementById('global-mouse').value;
            const selectedProbe = document.getElementById('global-probe').value;
            if (hasCorticalLayers && selectedMouse !== 'All' && selectedProbe !== 'All') {
                const physicalLayerBounds = aggregateBoundaryRecords(relevantBoundaries(selectedMouse))
                    .filter(b => b.boundary_type === 'layer')
                    .sort((a,b) => (depthAlignmentMode() === 'l1_l23' ? a.depth_from_l1_l23_um - b.depth_from_l1_l23_um : a.depth_from_cortex_um - b.depth_from_cortex_um));
                if (physicalLayerBounds.length) {
                    const boundaryText = physicalLayerBounds
                        .slice(0, 10)
                        .map(b => `${b.boundary_label} @ ${(depthAlignmentMode() === 'l1_l23' ? b.depth_from_l1_l23_um : b.depth_from_cortex_um).toFixed(0)} µm`)
                        .join(' &nbsp;•&nbsp; ');
                    layerAnnotations.push({
                        xref: 'paper', x: 0, yref: 'paper', y: 1.16,
                        text: `<b>Probe-specific CCF layer boundaries:</b> ${boundaryText}`,
                        showarrow: false, xanchor: 'left', align: 'left',
                        bgcolor: '#ffffff',
                        bordercolor: '#000000', borderwidth: 1, borderpad: 4, opacity: 1,
                        font: { size: 10, color: '#000000', family: 'Arial, sans-serif' }
                    });
                }
            }

            const layout = {
                title: `${hierarchyPrefix}: ${numericMetricNames[metric]||metric} by ${profileLevel}`,
                xaxis: { title: xTitle, categoryorder: 'array', categoryarray: xOrder.map(xFormatter) },
                yaxis: { title: `Mean ${numericMetricNames[metric]||metric}` },
                barmode: grouped.length > 1 ? 'group' : 'relative',
                shapes: layerShapes,
                annotations: layerAnnotations,
                margin: layerAnnotations.length ? { l: 60, r: 35, t: 125, b: 60 } : { l: 60, r: 35, t: 80, b: 60 }
            };
            applyPreservedZoom('plot-layers', layout, 'layers', metric);
            Plotly.react('plot-layers', traces, layout, BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());
        }

        function renderTimeSeries() {
            document.getElementById('context-group').style.display = 'flex';
            const mouse = document.getElementById('global-mouse').value;
            const context = document.getElementById('global-context').value;
            const plotDiv = document.getElementById('plot-timeseries');

            if (mouse === 'All' || context === 'All') {
                Plotly.purge(plotDiv);
                plotDiv.innerHTML = '<div class="warning-box"><b>Please select a specific Mouse ID and Context.</b></div>';
                document.getElementById('unit-count').textContent = 'Total Units Displayed: 0';
                return;
            }
            if (plotDiv.querySelector('.warning-box')) plotDiv.innerHTML = '';

            let spatialMetric = 'depth_from_cortex_um';
            const targetMet = document.getElementById('target-metric').value;
            if (targetMet === 'estimated_y' || targetMet === 'depth_from_cortex_um' || targetMet === 'depth_from_l1_l23_um') spatialMetric = targetMet;
            const isDepthMode = (spatialMetric === 'depth_from_cortex_um' || spatialMetric === 'depth_from_l1_l23_um');

            let data = getFilteredData().filter(d => d.fr_time_series && d.fr_time_series.length > 0);
            if (isDepthMode) {
                data = data.map(d => ({...d, _display_depth: unitDisplayDepth(d)})).filter(d => Number.isFinite(d._display_depth));
            } else {
                data = data.filter(d => d[spatialMetric] != null);
            }
            document.getElementById('unit-count').textContent = `Total Units Displayed: ${data.length.toLocaleString()}`;
            if (data.length === 0) { Plotly.react('plot-timeseries', [], {title: 'No Time-Series Data Available'}, BASE_PLOTLY_CONFIG); return; }

            const binSize = 50;
            let minDepth, maxDepth;
            if (isDepthMode) {
                if (depthAlignmentMode() === 'l1_l23') { minDepth = L1L23_DEPTH_MIN_UM; maxDepth = L1L23_DEPTH_MAX_UM; }
                else { minDepth = 0; maxDepth = CORTICAL_DEPTH_MAX_UM; }
            } else {
                let minVal = Infinity, maxVal = -Infinity;
                for(let i=0; i<data.length; i++) {
                    const v = Number(data[i][spatialMetric]);
                    if(Number.isFinite(v)) {
                        if(v < minVal) minVal = v;
                        if(v > maxVal) maxVal = v;
                    }
                }
                minDepth = Math.floor(minVal / binSize) * binSize;
                maxDepth = Math.ceil(maxVal / binSize) * binSize;
            }

            const yBins = [];
            for(let d = minDepth; d <= maxDepth; d += binSize) { yBins.push(d); }

            let maxBins = 0;
            for(let i=0; i<data.length; i++) { if(data[i].fr_time_series.length > maxBins) maxBins = data[i].fr_time_series.length; }

            const xBins = Array.from({length: maxBins}, (_, i) => i + 0.5);
            const zMatrix = Array(yBins.length - 1).fill(0).map(() => Array(maxBins).fill(null));

            for(let b_idx = 0; b_idx < yBins.length - 1; b_idx++) {
                const binData = data.filter(d => (isDepthMode ? d._display_depth : d[spatialMetric]) >= yBins[b_idx] && (isDepthMode ? d._display_depth : d[spatialMetric]) < yBins[b_idx+1]);
                if(binData.length > 0) {
                    for(let t = 0; t < maxBins; t++) {
                        let sum = 0, count = 0;
                        binData.forEach(d => {
                            if(t < d.fr_time_series.length) { sum += d.fr_time_series[t]; count++; }
                        });
                        zMatrix[b_idx][t] = count > 0 ? sum / count : null;
                    }
                }
            }

            const vmax = sharedTimeSeriesVmax(zMatrix.flat());
            const trace = {
                z: zMatrix, x: xBins, y: yBins.slice(0, -1).map(y => y + binSize/2),
                type: 'heatmap', colorscale: PARULA_COLORSCALE, zmin: 0, zmax: vmax,
                colorbar: { title: 'Mean Firing Rate (Hz)' }
            };

            const layout = { title: `Within-Session Firing Rate: Mouse ${mouse} | ${context}`, xaxis: { title: 'Time (Minutes)', range: [0, maxBins] }, yaxis: isDepthMode ? depthAxisForMode() : { title: numericMetricNames[spatialMetric]||spatialMetric, autorange: 'reversed' }, margin: { l: 60 } };

            if (isDepthMode && document.getElementById('global-probe').value !== 'All') {
                layout.shapes = []; layout.annotations = [];
                const allBoundsRaw = aggregateBoundaryRecords(relevantBoundaries(mouse));
                const allBounds = boundsForCurrentDepthMode(allBoundsRaw);
                const bounds = boundariesForDisplay(allBounds, false);
                bounds.forEach(b => {
                    layout.shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: b.display_depth, y1: b.display_depth, line: anatomyBoundaryLineStyle(b), layer: 'above' });
                });
                const anatomy = anatomySegmentsFromBounds(allBounds, minDepth, maxDepth);
                layout.annotations.push({ xref: 'paper', x: 1.015, yref: 'paper', y: 1.02, text: '<b>Area</b>', showarrow: false, xanchor: 'left', font: {size: 11, color: '#555'} });
                anatomy.areaSegments.forEach(seg => {
                    if ((seg.high - seg.low) < 35) return;
                    layout.annotations.push({ xref: 'paper', x: 1.015, yref: 'y', y: (seg.low + seg.high)/2, text: `<b>${seg.label}</b>`, showarrow: false, xanchor: 'left', yanchor: 'middle', font: {size: 10, color: '#444'}, bgcolor: 'rgba(255,255,255,0.92)', bordercolor: '#ddd', borderwidth: 0.5 });
                });
                if (anatomyOverlayMode() === 'area_layers') {
                    layout.annotations.push({ xref: 'paper', x: 1.115, yref: 'paper', y: 1.02, text: '<b>Layer</b>', showarrow: false, xanchor: 'left', font: {size: 11, color: '#555'} });
                    anatomy.layerSegments.forEach(seg => {
                        if ((seg.high - seg.low) < 35) return;
                        layout.annotations.push({ xref: 'paper', x: 1.115, yref: 'y', y: (seg.low + seg.high)/2, text: seg.label, showarrow: false, xanchor: 'left', yanchor: 'middle', font: {size: 10, color: '#666'}, bgcolor: 'rgba(255,255,255,0.90)', bordercolor: '#e3e3e3', borderwidth: 0.5 });
                    });
                }
                layout.margin = { l: 60, r: anatomyOverlayMode() === 'area_layers' ? 180 : 120, t: 80 };
            }

            applyPreservedZoom('plot-timeseries', layout, 'time', spatialMetric);
            Plotly.react('plot-timeseries', [trace], layout, BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());
        }

        function renderCustom() {
            document.getElementById('context-group').style.display = 'flex';
            const xMet = document.getElementById('custom-x').value;
            const yMet = document.getElementById('target-metric').value;

            const data = getFilteredData().filter(d => d[xMet] != null && d[yMet] != null);
            document.getElementById('unit-count').textContent = `Total Units Displayed: ${data.length.toLocaleString()}`;

            const buildTrace = (dGroup, name, color) => ({
                x: dGroup.map(d => safeLogVal(xMet, d[xMet])), 
                y: dGroup.map(d => safeLogVal(yMet, d[yMet])),
                mode: 'markers', type: 'scattergl', name: name,
                marker: { color: color, size: 4, opacity: 0.5 },
                text: dGroup.map(d => `Mouse: ${d.mouse_id}<br>Probe: ${d.probe}<br>Area: ${d.ccf_acronym || d.area}`), hoverinfo: 'x+y+text'
            });

            const traces = [
                buildTrace(data.filter(d => d.cohort === 'Motor'), 'Motor', '#1f77b4'),
                buildTrace(data.filter(d => d.cohort === 'Sequence'), 'Sequence', '#d62728')
            ].filter(t => t.x.length > 0);

            const useLogX = document.getElementById('log-x-axis') && document.getElementById('log-x-axis').checked;
            const useLogY = document.getElementById('log-y-axis') && document.getElementById('log-y-axis').checked;

            const layout = { 
                title: `${numericMetricNames[yMet]||yMet} vs ${numericMetricNames[xMet]||xMet}`, 
                xaxis: { title: numericMetricNames[xMet]||xMet, type: useLogX ? 'log' : 'linear' }, 
                yaxis: { title: numericMetricNames[yMet]||yMet, type: useLogY ? 'log' : 'linear' }
            };
            
            if (yMet === 'depth_from_cortex_um' || yMet === 'depth_from_l1_l23_um') { layout.yaxis.autorange = 'reversed'; }
            if (xMet === 'depth_from_cortex_um' || xMet === 'depth_from_l1_l23_um') { layout.xaxis.autorange = 'reversed'; }

            applyPreservedZoom('plot-custom', layout, xMet, yMet);
            Plotly.react('plot-custom', traces, layout, BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());
        }

        function _csdUnique(values) {
            return [...new Set(values.filter(v => v !== null && v !== undefined && String(v) !== ''))]
                .sort((a,b)=>String(a).localeCompare(String(b)));
        }

        function _csdSetOptions(id, values, labels=null) {
            const el=document.getElementById(id);
            if (!el) return;
            const old=el.value;
            el.innerHTML='';
            values.forEach((v,i)=>{
                const o=document.createElement('option');
                o.value=String(v);
                o.textContent=labels?labels[i]:String(v);
                el.appendChild(o);
            });
            if(values.map(String).includes(old)) el.value=old;
        }

        function _decodeInt16Matrix(rec, kind) {
            const key=[rec.session_key,rec.probe,rec.stim_table,rec.condition,kind].join('|');
            if(_csdDecodedCache.has(key)) return _csdDecodedCache.get(key);
            const b64 = kind==='lfp' ? rec.lfp_b64 : rec.csd_b64;
            const scale = Number(kind==='lfp' ? rec.lfp_scale : rec.csd_scale);
            const binary=atob(b64);
            const bytes=new Uint8Array(binary.length);
            for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
            const view=new DataView(bytes.buffer);
            const nrow=rec.shape[0], ncol=rec.shape[1];
            const out=[];
            let k=0;
            for(let r=0;r<nrow;r++) {
                const row=new Array(ncol);
                for(let c=0;c<ncol;c++,k++) {
                    const q=view.getInt16(k*2,true);
                    row[c]=(q===-32768)?null:q*scale;
                }
                out.push(row);
            }
            _csdDecodedCache.set(key,out);
            return out;
        }

        function _csdFiniteAbs(z) {
            const a=[];
            z.forEach(row=>row.forEach(v=>{
                if(v!==null && Number.isFinite(Number(v))) a.push(Math.abs(Number(v)));
            }));
            return a;
        }

        function _csdRobustLimit(z) {
            const a=_csdFiniteAbs(z).sort((x,y)=>x-y);
            if(!a.length) return 1;
            return a[Math.min(a.length-1,Math.floor(0.99*(a.length-1)))] || 1;
        }

        function _csdMedian(a) {
            const x=a.filter(Number.isFinite).slice().sort((u,v)=>u-v);
            if(!x.length) return NaN;
            const m=Math.floor(x.length/2);
            return x.length%2 ? x[m] : 0.5*(x[m-1]+x[m]);
        }

        function _csdMad(a) {
            const med=_csdMedian(a);
            if(!Number.isFinite(med)) return NaN;
            return _csdMedian(a.filter(Number.isFinite).map(v=>Math.abs(v-med)));
        }

        function _csdSessionRecords() {
            const s=document.getElementById('csd-session').value;
            return csdData.filter(d=>String(d.session_key)===String(s));
        }
        function _csdProbeRecords() {
            const p=document.getElementById('csd-probe').value;
            return _csdSessionRecords().filter(d=>String(d.probe)===String(p));
        }
        function _csdStimRecords() {
            const st=document.getElementById('csd-stim').value;
            return _csdProbeRecords().filter(d=>String(d.stim_table)===String(st));
        }
        function _csdCurrentRecord() {
            const cond=document.getElementById('csd-condition').value;
            return _csdStimRecords().find(d=>String(d.condition)===String(cond));
        }
        function _csdRefreshProbe() {
            const r=_csdSessionRecords();
            _csdSetOptions('csd-probe',_csdUnique(r.map(d=>d.probe)));
            _csdRefreshStim();
        }
        function _csdRefreshStim() {
            const r=_csdProbeRecords();
            _csdSetOptions('csd-stim',_csdUnique(r.map(d=>d.stim_table)));
            _csdRefreshCondition();
        }
        function _csdRefreshCondition() {
            const r=_csdStimRecords();
            const vals=_csdUnique(r.map(d=>d.condition));
            vals.sort((a,b)=>a==='All'?-1:b==='All'?1:String(a).localeCompare(String(b)));
            _csdSetOptions('csd-condition',vals);
            renderStaticCsd();
        }

        function _csdAlignedDepth(rec) {
            let offset=0, note='';
            const mode=document.getElementById('csd-align').value;
            if(mode==='cortex') {
                if(rec.cortex_entry_probe_depth_um===null || !Number.isFinite(Number(rec.cortex_entry_probe_depth_um))) {
                    note='Cortical-entry landmark unavailable for this probe; showing probe-top coordinates.';
                } else {
                    offset=Number(rec.cortex_entry_probe_depth_um);
                }
            } else if(mode==='l1') {
                if(rec.l1_l23_probe_depth_um===null || !Number.isFinite(Number(rec.l1_l23_probe_depth_um))) {
                    note='L1↔L2/3 landmark unavailable for this probe; showing probe-top coordinates.';
                } else {
                    offset=Number(rec.l1_l23_probe_depth_um);
                }
            }
            return {y:rec.depth_probe_um.map(v=>Number(v)-offset),offset,note};
        }

        function _csdNormLabel(label) {
            return String(label || '').trim().toLowerCase().replace(/\s+/g,'');
        }

        function _csdCorticalLayer(label) {
            const s=_csdNormLabel(label)
                .replace(/layer/g,'')
                .replace(/2-3/g,'2/3')
                .replace(/2_3/g,'2/3');
            if(/2\/3$/.test(s)) return '2/3';
            if(/6a$/.test(s)) return '6a';
            if(/6b$/.test(s)) return '6b';
            if(/4$/.test(s)) return '4';
            if(/5$/.test(s)) return '5';
            if(/6$/.test(s)) return '6';
            if(/1$/.test(s)) return '1';
            return null;
        }

        function _csdCorticalArea(label) {
            return _csdNormLabel(label)
                .replace(/layer/g,'')
                .replace(/2-3/g,'2/3')
                .replace(/2_3/g,'2/3')
                .replace(/(2\/3|6a|6b|1|4|5|6)$/,'');
        }

        function _csdHippInfo(label) {
            const s=_csdNormLabel(label).replace(/_/g,'-');
            let m=s.match(/^(ca[123])[-]?(so|sp|sr|slm)$/);
            if(m) {
                const names={so:'oriens',sp:'pyramidal',sr:'radiatum',slm:'lacunosum-moleculare'};
                return {region:m[1].toUpperCase(),stratum:m[2],display:m[1].toUpperCase()+' '+names[m[2]]};
            }
            if(s.includes('stratumoriens')) return {region:'CA',stratum:'so',display:'CA stratum oriens'};
            if(s.includes('stratumpyramidale') || s.includes('pyramidallayer')) return {region:'CA',stratum:'sp',display:'CA pyramidal layer'};
            if(s.includes('stratumradiatum')) return {region:'CA',stratum:'sr',display:'CA stratum radiatum'};
            if(s.includes('lacunosum') || s.includes('slm')) return {region:'CA',stratum:'slm',display:'CA lacunosum-moleculare'};
            if(/^dg[-]?(mo|mol)$/.test(s)) return {region:'DG',stratum:'mo',display:'DG molecular'};
            if(/^dg[-]?(sg|sgz)$/.test(s)) return {region:'DG',stratum:'sg',display:'DG granule'};
            if(/^dg[-]?(po|pol)$/.test(s)) return {region:'DG',stratum:'po',display:'DG polymorphic'};
            return null;
        }

        function _csdIsVisualSpan(sp) {
            const rg=String(sp.region_group || '').trim().toLowerCase();
            const lab=String(sp.label || '').trim().toLowerCase();
            return rg==='visual cortex' || /^vis/.test(lab) || lab.includes('visual');
        }

        function _csdIsHippSpan(sp) {
            const md=String(sp.major_division || '').trim().toLowerCase();
            const lab=String(sp.label || '').trim().toLowerCase();
            return md.includes('hippocamp') || /^ca[123]/.test(lab) || /^dg/.test(lab) || _csdHippInfo(lab)!==null;
        }

        function _csdBoundaryClass(prev,curr) {
            const prevCtx=String(prev.anatomy_class || '').toLowerCase()==='cortical';
            const currCtx=String(curr.anatomy_class || '').toLowerCase()==='cortical';
            if(prevCtx && currCtx) {
                const l0=_csdCorticalLayer(prev.label);
                const l1=_csdCorticalLayer(curr.label);
                const a0=_csdCorticalArea(prev.label);
                const a1=_csdCorticalArea(curr.label);
                if(l0 && l1 && a0 && a0===a1) {
                    if((l0==='2/3' && l1==='4') || (l0==='4' && l1==='2/3')) return 'l23-l4';
                    if((l0==='4' && l1==='5') || (l0==='5' && l1==='4')) return 'l4-l5';
                }
            }
            const h0=_csdHippInfo(prev.label);
            const h1=_csdHippInfo(curr.label);
            if(h0 && h1 && h0.stratum!==h1.stratum) {
                if(h0.region===h1.region || h0.region==='CA' || h1.region==='CA') return 'hipp-stratum';
            }
            return 'generic';
        }

        function _csdDepthInSpans(depth,spans) {
            return spans.some(sp=>{
                const lo=Math.min(Number(sp.z0),Number(sp.z1));
                const hi=Math.max(Number(sp.z0),Number(sp.z1));
                return depth>=lo && depth<=hi;
            });
        }

        function _csdEarlyVisualSink(rec,csd,x,y,offset,ymin,ymax) {
            const vis=(rec.anatomy_spans||[])
                .filter(_csdIsVisualSpan)
                .map(sp=>({z0:Number(sp.z0)-offset,z1:Number(sp.z1)-offset,label:sp.label}));
            if(!vis.length) return null;

            const rows=[];
            for(let r=0;r<y.length;r++) {
                const yy=Number(y[r]);
                if(yy>=ymin && yy<=ymax && _csdDepthInSpans(yy,vis)) rows.push(r);
            }
            if(rows.length<2) return null;
            const rowSet=new Set(rows);

            const early=[],base=[];
            for(let c=0;c<x.length;c++) {
                const t=Number(x[c]);
                if(t>=20 && t<=120) early.push(c);
                if(t>=-100 && t<0) base.push(c);
            }
            if(!early.length) return null;

            const baseline=[];
            rows.forEach(r=>base.forEach(c=>{
                const v=csd[r][c];
                if(v!==null && Number.isFinite(Number(v))) baseline.push(Number(v));
            }));
            const bm=_csdMedian(baseline);
            const bmed=Number.isFinite(bm)?bm:0;
            const mad=_csdMad(baseline);
            const sigma=Number.isFinite(mad) && mad>0 ? 1.4826*mad : NaN;

            const minima=early.map(c=>{
                let best=Infinity,bestRow=-1;
                rows.forEach(r=>{
                    const v=csd[r][c];
                    if(v!==null && Number.isFinite(Number(v)) && Number(v)<best) {best=Number(v);bestRow=r;}
                });
                return {c,best,bestRow};
            }).filter(d=>d.bestRow>=0 && Number.isFinite(d.best));
            if(!minima.length) return null;

            let onset=null;
            if(Number.isFinite(sigma) && sigma>0) {
                const threshold=bmed-4.0*sigma;
                onset=minima.find(d=>d.best<=threshold)||null;
            }

            let candidates=minima;
            let confidence='exploratory';
            if(onset) {
                const t0=Number(x[onset.c]);
                candidates=minima.filter(d=>Number(x[d.c])>=t0 && Number(x[d.c])<=t0+35);
                confidence='thresholded';
            }

            const peak=candidates.reduce((a,b)=>b.best<a.best?b:a,candidates[0]);
            if(!peak || peak.best>=0) return null;
            const snr=(Number.isFinite(sigma) && sigma>0)?Math.abs(peak.best-bmed)/sigma:NaN;
            if(Number.isFinite(snr) && snr<3.0) return null;

            const peakRow=peak.bestRow,peakCol=peak.c;
            const peakY=Number(y[peakRow]),peakT=Number(x[peakCol]),peakVal=Number(peak.best);
            const halfLevel=0.5*peakVal;
            let lo=peakRow,hi=peakRow;
            while(lo-1>=0 && rowSet.has(lo-1)) {
                const v=csd[lo-1][peakCol];
                if(v===null || !Number.isFinite(Number(v)) || Number(v)>halfLevel) break;
                lo--;
            }
            while(hi+1<y.length && rowSet.has(hi+1)) {
                const v=csd[hi+1][peakCol];
                if(v===null || !Number.isFinite(Number(v)) || Number(v)>halfLevel) break;
                hi++;
            }

            let bandTop=Number(y[lo]),bandBottom=Number(y[hi]);
            if(lo>0) bandTop=0.5*(Number(y[lo-1])+Number(y[lo]));
            if(hi<y.length-1) bandBottom=0.5*(Number(y[hi])+Number(y[hi+1]));
            if(bandTop>bandBottom) [bandTop,bandBottom]=[bandBottom,bandTop];
            const width=bandBottom-bandTop;
            const bandValid=Number.isFinite(width) && width>=20 && width<=400;

            const l4Spans=(rec.anatomy_spans||[])
                .filter(sp=>_csdIsVisualSpan(sp) && _csdCorticalLayer(sp.label)==='4')
                .map(sp=>({z0:Number(sp.z0)-offset,z1:Number(sp.z1)-offset,label:String(sp.label||'L4')}));
            let ccfL4=null;
            if(l4Spans.length) {
                const containing=l4Spans.find(sp=>peakY>=Math.min(sp.z0,sp.z1) && peakY<=Math.max(sp.z0,sp.z1));
                ccfL4=containing || l4Spans.slice().sort((a,b)=>Math.abs(peakY-0.5*(a.z0+a.z1))-Math.abs(peakY-0.5*(b.z0+b.z1)))[0];
            }
            let deltaToCcfL4Center=NaN;
            if(ccfL4) deltaToCcfL4Center=peakY-0.5*(ccfL4.z0+ccfL4.z1);

            return {
                peakY,peakT,peakVal,snr,confidence,
                bandTop:bandValid?bandTop:null,
                bandBottom:bandValid?bandBottom:null,
                ccfL4,deltaToCcfL4Center
            };
        }

        function _csdBuildOverlay(rec,offset,ymin,ymax,physiology=null) {
            const shapes=[];
            const annotations=[];
            const spans=(rec.anatomy_spans||[])
                .slice()
                .sort((a,b)=>Number(a.z0)-Number(b.z0));

            const visibleSpan=Math.max(1,Math.abs(ymax-ymin));
            const minLabelHeightUm=
                visibleSpan * (_CSD_MIN_LABEL_HEIGHT_PX / _CSD_PLOT_HEIGHT_PX);

            let labelsShown=0;
            let labelsTotal=0;

            spans.forEach((sp,i)=>{
                const z0=Number(sp.z0)-offset;
                const z1=Number(sp.z1)-offset;
                const lo=Math.min(z0,z1);
                const hi=Math.max(z0,z1);

                if(hi<ymin || lo>ymax) return;

                const clippedLo=Math.max(lo,ymin);
                const clippedHi=Math.min(hi,ymax);
                const mid=0.5*(clippedLo+clippedHi);
                const spanHeight=Math.max(0,clippedHi-clippedLo);

                const fill=(i%2===0)
                    ? 'rgba(247,247,247,0.98)'
                    : 'rgba(235,235,235,0.98)';

                shapes.push({
                    type:'rect',
                    xref:'paper',
                    x0:_CSD_ANATOMY_X0,
                    x1:_CSD_ANATOMY_X1,
                    yref:'y',
                    y0:clippedLo,
                    y1:clippedHi,
                    fillcolor:fill,
                    line:{ color:'rgba(120,120,120,0.34)', width:0.7 },
                    layer:'above'
                });

                if(i>0) {
                    const boundary=lo;
                    if(boundary>=ymin && boundary<=ymax) {
                        const cls=_csdBoundaryClass(spans[i-1],sp);

                        let line={ color:'rgba(90,90,90,0.30)', width:0.9, dash:'solid' };

                        if(cls==='l23-l4' || cls==='l4-l5') {
                            line={ color:'rgba(15,15,15,0.95)', width:2.6, dash:'solid' };
                        } else if(cls==='hipp-stratum') {
                            line={ color:'rgba(25,25,25,0.86)', width:2.0, dash:'dash' };
                        }

                        shapes.push({
                            type:'line',
                            xref:'paper',
                            x0:0,
                            x1:_CSD_ANATOMY_X1,
                            yref:'y',
                            y0:boundary,
                            y1:boundary,
                            line:line,
                            layer:'above'
                        });
                    }
                }

                labelsTotal += 1;

                if(spanHeight>=minLabelHeightUm) {
                    annotations.push({
                        xref:'paper',
                        x:_CSD_ANATOMY_LABEL_X,
                        yref:'y',
                        y:mid,
                        text:String(sp.label||''),
                        showarrow:false,
                        xanchor:'center',
                        yanchor:'middle',
                        align:'center',
                        bgcolor:'rgba(255,255,255,0)',
                        borderwidth:0,
                        font:{ size:9, color:'#3f3f3f' }
                    });
                    labelsShown += 1;
                }
            });

            annotations.push({
                xref:'paper',
                x:0.5*(_CSD_ANATOMY_X0+_CSD_ANATOMY_X1),
                yref:'paper',
                y:1.035,
                text:'<b>CCF anatomy</b>',
                showarrow:false,
                xanchor:'center',
                yanchor:'bottom',
                font:{size:10,color:'#444'}
            });

            if(physiology) {
                if(physiology.bandTop!==null && physiology.bandBottom!==null) {
                    shapes.push({
                        type:'rect', xref:'paper', x0:0, x1:_CSD_HEATMAP_DOMAIN_END,
                        yref:'y', y0:physiology.bandTop, y1:physiology.bandBottom,
                        fillcolor:'rgba(33,102,172,0.10)', line:{width:0}, layer:'below'
                    });

                    [physiology.bandTop,physiology.bandBottom].forEach(z=>{
                        shapes.push({
                            type:'line', xref:'paper', x0:0, x1:_CSD_HEATMAP_DOMAIN_END,
                            yref:'y', y0:z, y1:z,
                            line:{ color:'rgba(33,102,172,0.78)', width:1.4, dash:'dot' }, layer:'above'
                        });
                    });

                    shapes.push({
                        type:'rect', xref:'paper', x0:_CSD_ANATOMY_X0, x1:_CSD_ANATOMY_X1,
                        yref:'y', y0:physiology.bandTop, y1:physiology.bandBottom,
                        fillcolor:'rgba(33,102,172,0.06)',
                        line:{ color:'rgba(33,102,172,0.82)', width:1.4, dash:'dot' }, layer:'above'
                    });
                }

                shapes.push({
                    type:'line', xref:'paper', x0:0, x1:_CSD_HEATMAP_DOMAIN_END,
                    yref:'y', y0:physiology.peakY, y1:physiology.peakY,
                    line:{ color:'rgba(33,102,172,0.98)', width:2.7, dash:'dashdot' }, layer:'above'
                });

                shapes.push({
                    type:'line', xref:'paper', x0:_CSD_ANATOMY_X0, x1:_CSD_ANATOMY_X1,
                    yref:'y', y0:physiology.peakY, y1:physiology.peakY,
                    line:{ color:'rgba(33,102,172,1.0)', width:2.7, dash:'dashdot' }, layer:'above'
                });
            }

            return { shapes, annotations, labelsShown, labelsTotal };
        }

        function _csdVisibleRange(rec,offset,y) {
            let ymin=Math.min(...y),ymax=Math.max(...y),note='';
            const view=document.getElementById('csd-view').value;
            if(view==='visual') {
                const vis=(rec.anatomy_spans||[]).filter(_csdIsVisualSpan);
                if(vis.length) {
                    ymin=Math.min(...vis.map(sp=>Math.min(Number(sp.z0)-offset,Number(sp.z1)-offset)));
                    ymax=Math.max(...vis.map(sp=>Math.max(Number(sp.z0)-offset,Number(sp.z1)-offset)));
                } else note=' No visual-cortex CCF span on this probe; full depth shown.';
            }
            if(view==='hipp') {
                const hip=(rec.anatomy_spans||[]).filter(_csdIsHippSpan);
                if(hip.length) {
                    ymin=Math.min(...hip.map(sp=>Math.min(Number(sp.z0)-offset,Number(sp.z1)-offset)));
                    ymax=Math.max(...hip.map(sp=>Math.max(Number(sp.z0)-offset,Number(sp.z1)-offset)));
                } else note=' No recognised hippocampal CCF span on this probe; full depth shown.';
            }
            return {ymin,ymax,note};
        }

        function _spectralCurrentRecord() {
            const s=document.getElementById('csd-session').value;
            const p=document.getElementById('csd-probe').value;

            return spectralData.find(
                d => String(d.session_key)===String(s) && String(d.probe)===String(p)
            );
        }

        function _spectralBandNames(rec) {
            if(!rec || !rec.band_profiles) return [];
            return Object.keys(rec.band_profiles);
        }

        function _spectralEnsureBandOptions() {
            const rec = _spectralCurrentRecord();
            const el = document.getElementById('csd-band');
            const cbList = document.getElementById('band-checkbox-list');
            if (!el || !cbList) return;

            if (!rec) {
                el.innerHTML = '';
                const o = document.createElement('option');
                o.value = ''; o.textContent = 'No spectral record';
                el.appendChild(o);
                cbList.innerHTML = '<div>No spectral record</div>';
                return;
            }

            const names = _spectralBandNames(rec);
            const old = el.value;
            const current = [...el.options].map(o => o.value);
            const same = current.length === names.length && current.every((v, i) => v === names[i]);

            if (!same) {
                el.innerHTML = '';
                names.forEach(name => {
                    const o = document.createElement('option');
                    o.value = name; o.textContent = name;
                    el.appendChild(o);
                });
                
                if (names.includes(old)) el.value = old;
                else if (names.includes('Hipp theta 6–10 Hz')) el.value = 'Hipp theta 6–10 Hz';
                else if (names.length) el.value = names[0];

                const oldChecked = Array.from(cbList.querySelectorAll('.band-cb:checked')).map(cb => cb.value);
                cbList.innerHTML = '';
                names.forEach((name, i) => {
                    const color = PLOTLY_COLORS[i % PLOTLY_COLORS.length];
                    const div = document.createElement('div');
                    div.style.display = 'flex';
                    div.style.alignItems = 'center';
                    div.style.gap = '8px';
                    
                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.className = 'band-cb';
                    cb.value = name;
                    cb.dataset.color = color;
                    cb.id = 'cb-band-' + i;
                    
                    if (oldChecked.length > 0) {
                        cb.checked = oldChecked.includes(name);
                    } else {
                        cb.checked = (name === el.value);
                    }
                    
                    cb.addEventListener('change', renderStaticSpectral);
                    
                    const colorBox = document.createElement('div');
                    colorBox.style.width = '12px';
                    colorBox.style.height = '12px';
                    colorBox.style.flexShrink = '0';
                    colorBox.style.backgroundColor = color;
                    colorBox.style.borderRadius = '2px';
                    
                    const lbl = document.createElement('label');
                    lbl.htmlFor = cb.id;
                    lbl.textContent = name;
                    lbl.style.fontSize = '0.8rem';
                    lbl.style.cursor = 'pointer';
                    lbl.style.userSelect = 'none';
                    lbl.style.lineHeight = '1.2';
                    lbl.style.whiteSpace = 'normal'; 
                    
                    div.appendChild(cb);
                    div.appendChild(colorBox);
                    div.appendChild(lbl);
                    cbList.appendChild(div);
                });
            }
        }

        function _spectralFiniteValues(z) {
            const out=[];
            (z||[]).forEach(row=>{
                (row||[]).forEach(v=>{
                    const x=Number(v);
                    if(Number.isFinite(x)) out.push(x);
                });
            });
            return out;
        }

        function _spectralQuantile(a,q) {
            const x=(a||[]).filter(Number.isFinite).slice().sort((u,v)=>u-v);
            if(!x.length) return NaN;
            const pos=(x.length-1)*q;
            const lo=Math.floor(pos),hi=Math.ceil(pos);
            if(lo===hi) return x[lo];
            return x[lo]*(hi-pos)+x[hi]*(pos-lo);
        }

        function _spectralBandRangeShape(profile,ymin,ymax) {
            if(!profile) return null;
            if(!Number.isFinite(Number(profile.f0)) || !Number.isFinite(Number(profile.f1))) return null;

            return {
                type:'rect', xref:'x', x0:Number(profile.f0), x1:Number(profile.f1),
                yref:'y', y0:ymin, y1:ymax, fillcolor:'rgba(60,60,60,0.07)',
                line:{ color:'rgba(60,60,60,0.35)', width:1, dash:'dot' }, layer:'above'
            };
        }

        function renderStaticSpectral() {
            if(currentTab!=='tab-csd') return;

            const rec=_spectralCurrentRecord();
            const status=document.getElementById('spectral-status');

            if(!rec) {
                if(status) {
                    status.textContent='No cached long-window spectral record for this session/probe.';
                }
                Plotly.purge('plot-static-psd');
                Plotly.purge('plot-static-band');
                return;
            }

            _spectralEnsureBandOptions();

            const al=_csdAlignedDepth(rec);
            const y=al.y;
            const vr=_csdVisibleRange(rec,al.offset,y);
            const ymin=vr.ymin;
            const ymax=vr.ymax;

            const anatomy=_csdBuildOverlay(rec,al.offset,ymin,ymax,null);

            const freq=(rec.freq_hz||[]).map(Number);
            const psd=rec.psd_db||[];

            const allVals=_spectralFiniteValues(psd);
            let zmin=_spectralQuantile(allVals,0.03);
            let zmax=_spectralQuantile(allVals,0.97);

            if(!Number.isFinite(zmin)) zmin=-100;
            if(!Number.isFinite(zmax)) zmax=-20;
            if(zmax<=zmin) zmax=zmin+1;

            const bandName=document.getElementById('csd-band').value;
            const profile=rec.band_profiles ? rec.band_profiles[bandName] : null;
            const bandShape=_spectralBandRangeShape(profile,ymin,ymax);

            if(status) {
                const bandTxt=bandName ? ` · selected ${bandName}` : '';
                status.innerHTML=
                    `<b>${rec.session_key} · ${rec.probe}</b>` +
                    ` · ${rec.n_windows} × ${Number(rec.window_seconds).toFixed(1)} s windows` +
                    ` · ${rec.source_mode}` +
                    ` · source LFP ${Number(rec.sampling_rate_hz).toFixed(1)} Hz` +
                    bandTxt +
                    `${al.note?' · <b>'+al.note+'</b>':''}` +
                    `${vr.note||''}`;
            }

            const common={
                yaxis:{ title:document.getElementById('csd-align').selectedOptions[0].text, range:[ymax,ymin], automargin:true },
                margin:{l:82,r:52,t:48,b:60}, hovermode:'closest'
            };

            const psdShapes=[...anatomy.shapes];
            if(bandShape) psdShapes.push(bandShape);

            Plotly.react(
                'plot-static-psd',
                [{
                    type:'heatmap', x:freq, y:y, z:psd, colorscale:'Viridis', zmin:zmin, zmax:zmax,
                    colorbar:{ title:{text:'dB/Hz',side:'right'}, x:_CSD_COLORBAR_X, xanchor:'center', len:0.88, thickness:15 },
                    hovertemplate:'f=%{x:.1f} Hz<br>depth=%{y:.0f} µm<br>PSD=%{z:.2f} dB/Hz<extra></extra>'
                }],
                {
                    ...common,
                    xaxis:{ title:'Frequency (Hz)', domain:[0,_CSD_HEATMAP_DOMAIN_END], range:[STATIC_PSD_FMIN,STATIC_PSD_FMAX] },
                    shapes:psdShapes, annotations:anatomy.annotations, title:{ text:'LFP power spectrum by depth', font:{size:16} }
                },
                BASE_PLOTLY_CONFIG
            ).then(() => enhancePlotlyTooltips());

            const checkedCbs = Array.from(document.querySelectorAll('.band-cb:checked'));
            if(checkedCbs.length === 0) { Plotly.purge('plot-static-band'); return; }

            const metricEl = document.getElementById('csd-band-metric');
            const metric = (metricEl && metricEl.value === 'absolute') ? 'absolute' : 'relative';
            
            const traces = [];
            let allRatios = true;
            let anyRatios = false;

            checkedCbs.forEach(cb => {
                const bName = cb.value;
                const color = cb.dataset.color;
                const p = rec.band_profiles[bName];
                if(!p) return;
                
                const isRatio = String(p.unit) === 'ratio';
                if(isRatio) anyRatios = true; else allRatios = false;
                
                const valueArray = isRatio
                    ? p.values
                    : (metric === 'absolute'
                        ? (p.values_absolute_db || p.values)
                        : (p.values_relative || p.values));

                if(!Array.isArray(valueArray)) return;

                const vals = valueArray.map(v => { const x = Number(v); return Number.isFinite(x) ? x : null; });
                
                traces.push({
                    type: 'scatter', mode: 'lines+markers', x: vals, y: y,
                    name: bName,
                    line: { color: color, width: 2 }, marker: { color: color, size: 5 },
                    hovertemplate: `${bName}<br>depth=%{y:.0f} µm<br>` +
                        (isRatio ? 'ratio=%{x:.3f}' : (metric === 'absolute' ? 'absolute power=%{x:.2f} dB mV²' : 'relative power=%{x:.2f}%')) +
                        '<extra></extra>'
                });
            });

            if(traces.length === 0) { Plotly.purge('plot-static-band'); return; }

            let xTitle = 'Power';
            if(allRatios) xTitle = 'Power ratio';
            else if(anyRatios) xTitle = 'Power / Ratio';
            else xTitle = (metric === 'absolute' ? 'Absolute band power (dB mV²)' : 'Relative band power (% of 1–200 Hz)');

            Plotly.react(
                'plot-static-band',
                traces,
                {
                    ...common,
                    xaxis: { title: xTitle, domain: [0, _CSD_HEATMAP_DOMAIN_END], rangemode: 'tozero', zeroline: true, zerolinecolor: '#bbb' },
                    shapes: anatomy.shapes, annotations: anatomy.annotations, 
                    title: { text: traces.length === 1 ? traces[0].name : 'Selected spectral bands', font: { size: 16 } },
                    showlegend: false
                },
                BASE_PLOTLY_CONFIG
            ).then(() => enhancePlotlyTooltips());
        }

        function renderStaticCsd() {
            if(currentTab!=='tab-csd') return;
            const rec=_csdCurrentRecord();
            const status=document.getElementById('csd-status');
            if(!rec) {
                status.textContent='No embedded LFP/CSD record matches these selectors.';
                Plotly.purge('plot-static-lfp'); Plotly.purge('plot-static-csd');
                Plotly.purge('plot-static-psd'); Plotly.purge('plot-static-band');
                return;
            }

            const lfp=_decodeInt16Matrix(rec,'lfp');
            const csd=_decodeInt16Matrix(rec,'csd');
            const al=_csdAlignedDepth(rec);
            const y=al.y;
            const x=rec.time_ms.map(Number);
            const vr=_csdVisibleRange(rec,al.offset,y);
            const ymin=vr.ymin,ymax=vr.ymax;

            const physiology=_csdEarlyVisualSink(rec,csd,x,y,al.offset,ymin,ymax);
            const anatomyOverlay=_csdBuildOverlay(rec,al.offset,ymin,ymax,null);
            const csdOverlay=_csdBuildOverlay(rec,al.offset,ymin,ymax,physiology);
            const finitePct=(100*Number(rec.csd_finite_fraction||0)).toFixed(1);

            let physText='';
            if(physiology) {
                const snrTxt=Number.isFinite(physiology.snr)?` · baseline SNR≈${physiology.snr.toFixed(1)}`:'';
                const bandTxt=(physiology.bandTop!==null && physiology.bandBottom!==null)?` · half-peak band ${physiology.bandTop.toFixed(0)}–${physiology.bandBottom.toFixed(0)} µm`:'';
                const deltaTxt=Number.isFinite(physiology.deltaToCcfL4Center)?` · Δ vs CCF L4 centre ${physiology.deltaToCcfL4Center>=0?'+':''}${physiology.deltaToCcfL4Center.toFixed(0)} µm`:'';
                physText=`<br><b>Physiology:</b> early visual sink at ${physiology.peakT.toFixed(1)} ms, depth ${physiology.peakY.toFixed(0)} µm${snrTxt}${bandTxt}${deltaTxt}. Sink centre is an L4 landmark; band edges are heuristic.`;
            }

            const labelTxt=csdOverlay.labelsShown<csdOverlay.labelsTotal
                ? ` · CCF anatomy track ${csdOverlay.labelsShown}/${csdOverlay.labelsTotal} labels shown (thin spans remain visible but text is suppressed)`
                : ` · CCF anatomy track ${csdOverlay.labelsShown} labels`;

            status.innerHTML=`<b>${rec.session_key} · ${rec.probe}</b> &nbsp; ${rec.stim_table} &nbsp; ${rec.group_by?rec.group_by+': ':''}${rec.condition}<br>${rec.n_trials} trials · source LFP ${Number(rec.sampling_rate_hz).toFixed(1)} Hz · displayed ${rec.shape[0]} depth × ${rec.shape[1]} time bins · CSD: ${rec.csd_mode} · finite ${finitePct}%${labelTxt}${al.note?' · <b>'+al.note+'</b>':''}${vr.note}${physText}`;

            const common={
                xaxis:{title:'Time from stimulus onset (ms)',zeroline:true,zerolinewidth:1.5,zerolinecolor:'#111',domain:[0,_CSD_HEATMAP_DOMAIN_END]},
                yaxis:{title:document.getElementById('csd-align').selectedOptions[0].text,range:[ymax,ymin],automargin:true},
                margin:{l:82,r:52,t:48,b:58}, hovermode:'closest'
            };
            const onsetLine={type:'line',xref:'x',x0:0,x1:0,yref:'y',y0:ymin,y1:ymax,line:{color:'#111',width:1.3}};

            const llim=_csdRobustLimit(lfp),clim=_csdRobustLimit(csd);
            Plotly.react('plot-static-lfp',[{type:'heatmap',x:x,y:y,z:lfp,colorscale:_LFP_BLUE_GREY_RED,reversescale:false,zmid:0,zmin:-llim,zmax:llim,colorbar:{title:{text:'mV',side:'right'},x:_CSD_COLORBAR_X,xanchor:'center',len:0.88,thickness:15},hovertemplate:'t=%{x:.1f} ms<br>depth=%{y:.0f} µm<br>LFP=%{z:.4g} mV<extra></extra>'}],{...common,shapes:[...anatomyOverlay.shapes,onsetLine],annotations:anatomyOverlay.annotations,title:{text:'Stimulus-triggered LFP',font:{size:16}}},BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());
            Plotly.react('plot-static-csd',[{type:'heatmap',x:x,y:y,z:csd,colorscale:_CSD_BLUE_GREY_RED,reversescale:false,zmid:0,zmin:-clim,zmax:clim,colorbar:{title:{text:'CSD',side:'right'},x:_CSD_COLORBAR_X,xanchor:'center',len:0.88,thickness:15},hovertemplate:'t=%{x:.1f} ms<br>depth=%{y:.0f} µm<br>CSD=%{z:.4g}<extra></extra>'}],{...common,shapes:[...csdOverlay.shapes,onsetLine],annotations:csdOverlay.annotations,title:{text:'CSD · blue = sink · red = source',font:{size:16}}},BASE_PLOTLY_CONFIG).then(() => enhancePlotlyTooltips());

            renderStaticSpectral();
        }

        function initStaticCsd() {
            if(!csdData.length) return;
            const sessions=_csdUnique(csdData.map(d=>d.session_key));
            const labels=sessions.map(s=>{const r=csdData.find(d=>String(d.session_key)===String(s));return r?r.session_label:s;});
            _csdSetOptions('csd-session',sessions,labels);
            document.getElementById('csd-session').addEventListener('change',_csdRefreshProbe);
            document.getElementById('csd-probe').addEventListener('change',_csdRefreshStim);
            document.getElementById('csd-stim').addEventListener('change',_csdRefreshCondition);
            document.getElementById('csd-condition').addEventListener('change',renderStaticCsd);
            document.getElementById('csd-align').addEventListener('change',renderStaticCsd);
            document.getElementById('csd-view').addEventListener('change',renderStaticCsd);
            document.getElementById('csd-band').addEventListener('change',renderStaticSpectral);
            document.getElementById('csd-band-metric').addEventListener('change',renderStaticSpectral);
            _spectralEnsureBandOptions();
            _csdRefreshProbe();
        }

        // ROUTING
        document.getElementById('global-mouse').addEventListener('change', renderAll);
        document.getElementById('global-probe').addEventListener('change', renderAll);
        document.getElementById('global-division').addEventListener('change', updateCascadingFilters);
        document.getElementById('global-group').addEventListener('change', updateCascadingFilters);
        document.getElementById('global-family').addEventListener('change', updateCascadingFilters);
        document.getElementById('global-subregion').addEventListener('change', updateCascadingFilters);
        document.getElementById('global-layer').addEventListener('change', updateCascadingFilters);
        document.getElementById('global-area').addEventListener('change', renderAll);
        document.getElementById('boundary-overlay').addEventListener('change', renderAll);
        document.getElementById('depth-alignment').addEventListener('change', renderAll);
        document.getElementById('color-scale-mode').addEventListener('change', renderAll);
        document.getElementById('target-metric').addEventListener('change', renderAll);
        document.getElementById('custom-x').addEventListener('change', renderAll);

        function renderAll() {
            const targetMetGroup = document.getElementById('y-axis-group');
            const xAxisGroup = document.getElementById('x-axis-group');
            const sidebar = document.querySelector('.sidebar');
            const grid = document.querySelector('.layout-grid');
            const countBox = document.getElementById('unit-count');

            if (currentTab === 'tab-timeseries' || currentTab === 'tab-yield' || currentTab === 'tab-csd') {
                targetMetGroup.style.display = 'none';
                document.getElementById('log-toggles-container').style.display = 'none';
            } else {
                targetMetGroup.style.display = 'flex';
                if (currentTab === 'tab-ecdf' || currentTab === 'tab-custom') {
                    document.getElementById('log-toggles-container').style.display = 'block';
                    if (currentTab === 'tab-ecdf') {
                        document.getElementById('log-y-axis-container').style.display = 'none';
                    } else {
                        document.getElementById('log-y-axis-container').style.display = 'inline-block';
                    }
                } else {
                    document.getElementById('log-toggles-container').style.display = 'none';
                }
            }

            if (xAxisGroup) xAxisGroup.style.display = (currentTab === 'tab-custom') ? 'flex' : 'none';
            if (sidebar) sidebar.style.display = (currentTab === 'tab-csd') ? 'none' : 'block';
            if (grid) grid.classList.toggle('csd-mode', currentTab === 'tab-csd');
            if (countBox) countBox.style.display = (currentTab === 'tab-csd') ? 'none' : 'block';

            if(currentTab === 'tab-yield') renderYield();
            else if(currentTab === 'tab-ecdf') renderECDF();
            else if(currentTab === 'tab-depth') renderDepth();
            else if(currentTab === 'tab-layers') renderLayerProfile();
            else if(currentTab === 'tab-timeseries') renderTimeSeries();
            else if(currentTab === 'tab-custom') renderCustom();
            else if(currentTab === 'tab-csd') renderStaticCsd();
        }

        window.switchTab = function(evt, tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');
            currentTab = tabName;
            renderAll();
        }

        initStaticCsd();
        if (rawData && rawData.length > 0) {
            updateCascadingFilters();
        }
    </script>
</body>
</html>"""

    final_html = (
        html_template
        .replace("{{DATABASE_PAYLOAD}}", units_json)
        .replace("{{BOUNDARY_PAYLOAD}}", boundaries_json)
        .replace("{{CSD_PAYLOAD}}", csd_json)
        .replace("{{SPECTRAL_PAYLOAD}}", psd_json)
    )
    
    with open(OUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Interactive HTML saved to: {OUT_HTML_PATH}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_JSON_PATH):
        raise FileNotFoundError(f"Missing JSON file: {DATA_JSON_PATH}\nPlease run extract_ecephys_qc_dashboard.py first.")

    print(f"Loading data from {DATA_JSON_PATH}...")
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    # 1. Generate Static SVG
    generate_static_svg(master_data.get("units", []))

    # 2. Generate Interactive HTML
    generate_html(master_data)

    print("\nSUCCESS! Both SVG and HTML were generated.")


