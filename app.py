import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import os
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

# ==========================
# DATA
# ==========================
df = pd.read_csv("TFG_panel_balanced.csv")
df = df.sort_values(["country_iso2", "year"]).copy()

iso2_to_name = (
    df.dropna(subset=["country_iso2", "country"])
      .drop_duplicates("country_iso2")
      .set_index("country_iso2")["country"]
      .to_dict()
)

ALL_COUNTRIES = sorted(df["country_iso2"].dropna().unique().tolist())
YEAR_MIN = int(df["year"].min())
YEAR_MAX = int(df["year"].max())

# ==========================
# MÉTRICAS
# ==========================
df["d_debt"] = df.groupby("country_iso2")["debt"].diff()

# déficit (negativo) -> presión de déficit (positivo)
df["deficit_pressure"] = -df["deficit"]
df["d_deficit_pressure"] = df.groupby("country_iso2")["deficit_pressure"].diff()

# ==========================
# COORDS (centroides aprox)
# ==========================
coords = {
    'AT': (47.5, 14.5), 'BE': (50.5, 4.5), 'BG': (42.7, 25.5), 'CY': (35.1, 33.4),
    'CZ': (49.8, 15.5), 'DE': (51.0, 10.0), 'DK': (56.0, 10.0), 'EE': (58.7, 25.0),
    'ES': (40.0, -4.0), 'FI': (64.0, 26.0), 'FR': (46.5, 2.5), 'GR': (39.0, 22.0),
    'HR': (45.1, 15.2), 'HU': (47.1, 19.5), 'IE': (53.4, -8.0), 'IT': (42.5, 12.5),
    'LT': (55.2, 23.9), 'LU': (49.8, 6.1), 'LV': (56.8, 24.6), 'MT': (35.9, 14.4),
    'NL': (52.3, 5.3), 'PL': (52.0, 19.0), 'PT': (39.5, -8.0), 'RO': (45.9, 24.9),
    'SE': (62.0, 15.0), 'SI': (46.1, 14.8), 'SK': (48.7, 19.7)
}

# ==========================
# BUILD GRAPH (ventana móvil)
# ==========================
THR = 0.5
DEFAULT_WINDOW = 8

def build_graph_for_year(metric_col: str, year_end: int, window: int):
    year_start = max(YEAR_MIN, year_end - window + 1)
    sub = df[(df["year"] >= year_start) & (df["year"] <= year_end)].copy()
    piv = sub.pivot(index="year", columns="country_iso2", values=metric_col)
    corr = piv.corr()

    G = nx.Graph()
    for c in corr.columns:
        G.add_node(c)

    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            w = corr.loc[c1, c2]
            if pd.notna(w) and abs(w) >= THR:
                G.add_edge(c1, c2, weight=float(abs(w)), signed_weight=float(w))

    if G.number_of_edges() > 0:
        part = community_louvain.best_partition(G, weight="weight", random_state=42)
    else:
        part = {n: i for i, n in enumerate(G.nodes())}

    return G, part, year_start, year_end

# ==========================
# SUBGRAPHS
# ==========================
def ego_subgraph(G, country):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if u == country or v == country:
            H.add_edge(u, v, **d)
    return H

def community_subgraph(G, part, country):
    cid = part.get(country, None)
    if cid is None:
        return nx.Graph(), None, []
    members = [n for n, c in part.items() if c == cid]
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if u in members and v in members:
            H.add_edge(u, v, **d)
    return H, cid, members

def country_within_community(G, part, country):
    Hc, cid, members = community_subgraph(G, part, country)
    if cid is None:
        return nx.Graph(), None, []
    He = ego_subgraph(Hc, country)
    return He, cid, members

# ==========================
# FIGURE
# - contexto gris (edges + nodos)
# - overlay por comunidad (colores)
# - click funciona porque el trace "clickable" lleva text=ISO2
# ==========================
def make_figure(G, part, selected_country, mode, title):
    fig = go.Figure()

    # ---------- CONTEXTO GRIS: edges ----------
    grey_edge_lats, grey_edge_lons = [], []
    for u, v, d in G.edges(data=True):
        if u in coords and v in coords:
            lat1, lon1 = coords[u]
            lat2, lon2 = coords[v]
            grey_edge_lats += [lat1, lat2, None]
            grey_edge_lons += [lon1, lon2, None]

    fig.add_trace(go.Scattergeo(
        lat=grey_edge_lats, lon=grey_edge_lons,
        mode="lines",
        line=dict(width=1, color="rgba(140,140,140,0.18)"),
        hoverinfo="none",
        showlegend=False,
        name="context_edges"
    ))

    # ---------- CONTEXTO GRIS: nodos (clickable + hover) ----------
    base_lats, base_lons, base_text, base_cdata = [], [], [], []
    for n in G.nodes():
        if n in coords:
            lat, lon = coords[n]
            base_lats.append(lat); base_lons.append(lon)
            base_text.append(n)  # <- ISO2 (click)
            cname = iso2_to_name.get(n, n)
            cid = part.get(n, None)
            base_cdata.append([cname, n, cid])

    fig.add_trace(go.Scattergeo(
        lat=base_lats, lon=base_lons,
        text=base_text,
        customdata=base_cdata,
        mode="markers",
        marker=dict(size=10, color="rgba(160,160,160,0.35)"),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Comunidad: %{customdata[2]}<extra></extra>"
        ),
        showlegend=False,
        name="click_layer"
    ))

    # ---------- SUBGRAFO OVERLAY según modo ----------
    drawG = G
    current_cid = part.get(selected_country) if selected_country else None

    if mode == "ego" and selected_country:
        drawG = ego_subgraph(G, selected_country)
    elif mode == "community" and selected_country:
        drawG, current_cid, _ = community_subgraph(G, part, selected_country)
    elif mode == "ego_in_community" and selected_country:
        drawG, current_cid, _ = country_within_community(G, part, selected_country)
    else:
        drawG = G  # all

    # ---------- OVERLAY: edges (azul) ----------
    edge_lats, edge_lons = [], []
    for u, v, d in drawG.edges(data=True):
        if u in coords and v in coords:
            lat1, lon1 = coords[u]
            lat2, lon2 = coords[v]
            edge_lats += [lat1, lat2, None]
            edge_lons += [lon1, lon2, None]

    fig.add_trace(go.Scattergeo(
        lat=edge_lats, lon=edge_lons,
        mode="lines",
        line=dict(width=2, color="rgba(30,100,220,0.55)"),
        hoverinfo="none",
        showlegend=False,
        name="overlay_edges"
    ))

    # ---------- OVERLAY: nodos por comunidad (COLORES) ----------
    # Solo dibujamos nodos que estén en drawG (para que ego/communidad filtren de verdad)
    nodes_in_overlay = set([n for n in drawG.nodes() if n in coords])

    comms = {}
    for n, cid in part.items():
        if n in nodes_in_overlay:
            comms.setdefault(cid, []).append(n)

    for cid, nodes in sorted(comms.items(), key=lambda x: len(x[1]), reverse=True):
        lats, lons, texts, sizes, cdata = [], [], [], [], []
        for n in nodes:
            lat, lon = coords[n]
            lats.append(lat); lons.append(lon)

            cname = iso2_to_name.get(n, n)
            cdata.append([cname, n, cid])

            # solo mostramos etiqueta para el seleccionado
            if n == selected_country:
                texts.append(n)
                sizes.append(26)
            else:
                texts.append("")
                sizes.append(14)

        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            text=texts,
            customdata=cdata,
            mode="markers+text",
            textposition="top center",
            marker=dict(size=sizes),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Comunidad: %{customdata[2]}<extra></extra>"
            ),
            name=f"Community {cid}",
            showlegend=True
        ))

    fig.update_layout(
        title=title,
        geo=dict(scope="europe", projection_type="mercator"),
        height=850,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
        legend_title="Comunidades (Louvain)"
    )

    return fig, current_cid

# ==========================
# DASH APP
# ==========================
app = dash.Dash(__name__)
server = app.server
app.title = "EU Fiscal Network"

app.layout = html.Div([
    html.H2("Red fiscal europea (ML: comunidades Louvain)"),

    html.Div([
        html.Div([
            html.Label("Año (fin de ventana)"),
            dcc.Slider(
                id="year",
                min=YEAR_MIN,
                max=YEAR_MAX,
                step=1,
                value=min(2013, YEAR_MAX),
                marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX+1, 3)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"width": "72%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            html.Label("Ventana (años)"),
            dcc.Dropdown(
                id="window",
                options=[{"label": str(w), "value": w} for w in [5, 6, 8, 10, 12]],
                value=DEFAULT_WINDOW,
                clearable=False
            ),
        ], style={"width": "25%", "display": "inline-block", "paddingLeft": "20px", "verticalAlign": "top"}),
    ]),

    html.Br(),

    html.Div([
        html.Div([
            html.Label("Métrica"),
            dcc.RadioItems(
                id="metric",
                options=[
                    {"label": "ΔDeuda (d_debt)", "value": "d_debt"},
                    {"label": "ΔPresión de déficit (d_deficit_pressure)", "value": "d_deficit_pressure"},
                ],
                value="d_debt",
                inline=True
            )
        ], style={"width":"45%", "display":"inline-block"}),

        html.Div([
            html.Label("Modo"),
            dcc.RadioItems(
                id="mode",
                options=[
                    {"label":"Red completa", "value":"all"},
                    {"label":"Enlaces del país (ego)", "value":"ego"},
                    {"label":"Comunidad del país", "value":"community"},
                    {"label":"País dentro de su comunidad", "value":"ego_in_community"},
                ],
                value="ego",
                inline=False
            )
        ], style={"width":"35%", "display":"inline-block", "paddingLeft":"20px"}),

        html.Div([
            html.Button("▶ Play", id="play", n_clicks=0, style={"marginRight":"10px"}),
            html.Button("⏸ Pause", id="pause", n_clicks=0),
            dcc.Interval(id="ticker", interval=900, n_intervals=0, disabled=True),
        ], style={"width":"18%", "display":"inline-block", "verticalAlign":"top", "paddingLeft":"10px"}),
    ]),

    html.Br(),
    html.Div(id="status", style={"fontSize":"16px", "marginBottom":"8px"}),

    dcc.Store(id="selected_country_store", data=None),

    dcc.Graph(id="network", clear_on_unhover=True),
    html.Div("Tip: haz click en un país (nodo). Hover: nombre completo + comunidad. El país seleccionado se ve más grande.",
             style={"opacity":0.7})
])

# ==========================
# CLICK → país seleccionado
# ==========================
@app.callback(
    Output("selected_country_store", "data"),
    Input("network", "clickData"),
    State("selected_country_store", "data")
)
def update_selected_country(clickData, current):
    if clickData and "points" in clickData and len(clickData["points"]) > 0:
        iso2 = clickData["points"][0].get("text")
        if iso2 in ALL_COUNTRIES:
            return iso2
    return current

# ==========================
# PLAY/PAUSE
# ==========================
@app.callback(
    Output("ticker", "disabled"),
    Input("play", "n_clicks"),
    Input("pause", "n_clicks"),
    prevent_initial_call=True
)
def control_play(play_clicks, pause_clicks):
    trig = ctx.triggered_id
    return False if trig == "play" else True

@app.callback(
    Output("year", "value"),
    Input("ticker", "n_intervals"),
    State("year", "value")
)
def tick_year(n, year):
    if year is None:
        year = YEAR_MIN
    nxt = year + 1
    if nxt > YEAR_MAX:
        nxt = YEAR_MIN
    return nxt

# ==========================
# REDRAW
# ==========================
@app.callback(
    Output("network", "figure"),
    Output("status", "children"),
    Input("year", "value"),
    Input("window", "value"),
    Input("metric", "value"),
    Input("mode", "value"),
    Input("selected_country_store", "data")
)
def redraw(year_end, window, metric, mode, selected_country):
    G, part, y0, y1 = build_graph_for_year(metric, int(year_end), int(window))

    metric_name = "Δdebt" if metric == "d_debt" else "Δpresión déficit"
    mode_name = {
        "all": "Red completa",
        "ego": "Enlaces del país (ego)",
        "community": "Comunidad del país",
        "ego_in_community": "País dentro de su comunidad"
    }[mode]

    # si no hay país seleccionado, ego/community no tiene sentido → forzamos all
    if selected_country is None and mode != "all":
        mode = "all"
        mode_name = "Red completa"

    title = f"{metric_name} | Ventana {y0}–{y1} | {mode_name}" + (f" | País: {selected_country}" if selected_country else "")

    fig, cid = make_figure(G, part, selected_country, mode, title)

    if selected_country:
        full = iso2_to_name.get(selected_country, selected_country)
        status = f"País seleccionado: {full} ({selected_country}) | Comunidad en esta ventana: {cid}"
    else:
        status = f"Sin país seleccionado | Ventana {y0}–{y1} | (Haz click en un país para fijarlo)"

    return fig, status

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False
    )

