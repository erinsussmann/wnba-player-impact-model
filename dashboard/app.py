"""WNBA player-impact dashboard: ratings table, lineup projector, validation chart."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

from src.lineup_tool import project_lineup
from src.rapm_model import fit_rapm, compare_to_published

SEASON = 2025
PER_100 = 100
# From src.possessions.build_possessions's printed output for this committed
# model version -- not recomputed here since the raw/intermediate possession
# data isn't shipped with the deployed app (only the small fitted outputs are).
N_GAMES = 279
N_POSSESSIONS = 43143

# --- data -------------------------------------------------------------
ratings = fit_rapm(SEASON)
ratings = ratings.dropna(subset=["player_name", "team_id"]).copy()
ratings["rapm_per_100"] = ratings["rapm"] * PER_100
validation = compare_to_published(SEASON, ratings.drop(columns=["rapm_per_100"]))
PEARSON_R = validation[["rapm_mine", "rapm_published"]].corr().iloc[0, 1]
TOP_PLAYER = ratings.sort_values("rapm_per_100", ascending=False).iloc[0]

TEAM_NAMES = {
    3: "Dallas Wings", 5: "Indiana Fever", 6: "Los Angeles Sparks", 8: "Minnesota Lynx",
    9: "New York Liberty", 11: "Phoenix Mercury", 14: "Seattle Storm", 16: "Washington Mystics",
    17: "Las Vegas Aces", 18: "Connecticut Sun", 19: "Chicago Sky", 20: "Atlanta Dream",
    129689: "Golden State Valkyries",
}
ratings["team_name"] = ratings["team_id"].map(TEAM_NAMES).fillna(ratings["team_id"].astype(str))

# --- palette ------------------------------------------------------------
# Diverging pair for positive/negative net rating (identity signal, used only
# for that one meaning throughout). Accent is UI chrome only, never used to
# encode data.
POSITIVE = "#2563EB"
NEGATIVE = "#DC4C3D"
ACCENT = "#4F46E5"
INK = "#0F172A"
MUTED = "#64748B"
BORDER = "rgba(15, 23, 42, 0.08)"
CARD_BG = "#FFFFFF"
PAGE_BG = "#F6F7FB"
CHART_FONT = dict(family="'Inter', -apple-system, Helvetica, Arial, sans-serif", color=INK, size=13)

team_options = [{"label": "All teams", "value": "all"}] + [
    {"label": name, "value": tid} for tid, name in sorted(TEAM_NAMES.items(), key=lambda kv: kv[1])
]
player_options = [
    {"label": f"{row.player_name} ({row.team_name})", "value": row.player_name}
    for row in ratings.sort_values("player_name").itertuples()
]

app = dash.Dash(__name__)
server = app.server  # exposed for gunicorn (`gunicorn dashboard.app:server`)
app.title = "WNBA Player Impact"

# Built separately from Dash's own index_string template below: that template
# uses single-brace {%...%} placeholders Dash substitutes itself, which would
# collide with Python string formatting if this CSS were inlined into it directly.
_CUSTOM_CSS = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{
    background: {PAGE_BG}; margin: 0; color: {INK};
    font-family: 'Inter', -apple-system, Helvetica, Arial, sans-serif;
}}
.card {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 24px; margin-bottom: 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}
.eyebrow {{
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 11.5px;
    font-weight: 700; color: {ACCENT}; margin: 0 0 6px 0;
}}
.section-title {{ font-size: 19px; font-weight: 700; margin: 0 0 4px 0; }}
.section-sub {{ color: {MUTED}; font-size: 14px; margin: 0 0 16px 0; line-height: 1.5; }}
.stat-tile {{
    flex: 1; min-width: 140px; background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 16px 18px;
}}
.stat-value {{ font-size: 26px; font-weight: 800; color: {INK}; line-height: 1.15; }}
.stat-label {{ font-size: 12.5px; color: {MUTED}; margin-top: 4px; }}
.Select-control, .dash-dropdown .Select-control {{
    border-radius: 10px !important; border-color: {BORDER} !important;
}}
table.dash-spreadsheet-container {{ border-radius: 10px; overflow: hidden; }}
a {{ color: {ACCENT}; }}
"""

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>""" + _CUSTOM_CSS + """</style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""


def stat_tile(value, label):
    return html.Div(
        className="stat-tile",
        children=[html.Div(value, className="stat-value"), html.Div(label, className="stat-label")],
    )


def section_header(eyebrow, title, subtitle=None):
    children = [html.P(eyebrow, className="eyebrow"), html.H2(title, className="section-title")]
    if subtitle:
        children.append(html.P(subtitle, className="section-sub"))
    return html.Div(children, style={"marginBottom": "16px" if subtitle else "16px"})


app.layout = html.Div(
    style={"maxWidth": "1080px", "margin": "0 auto", "padding": "40px 20px 60px"},
    children=[
        # --- header --------------------------------------------------
        html.Div(
            style={"marginBottom": "28px"},
            children=[
                html.P("WNBA ANALYTICS", className="eyebrow", style={"color": ACCENT}),
                html.H1(
                    f"Player Impact — {SEASON} Season",
                    style={"fontSize": "34px", "fontWeight": 800, "margin": "0 0 8px 0", "letterSpacing": "-0.02em"},
                ),
                html.P(
                    "A RAPM-style (Regularized Adjusted Plus-Minus) model fit from real play-by-play "
                    "possession data, validated against published RAPM ratings.",
                    style={"color": MUTED, "fontSize": "15px", "maxWidth": "640px", "lineHeight": "1.6"},
                ),
            ],
        ),

        # --- stat tiles ------------------------------------------------
        html.Div(
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "28px"},
            children=[
                stat_tile(f"{len(ratings)}", "players rated"),
                stat_tile(f"{N_GAMES}", "games analyzed"),
                stat_tile(f"{N_POSSESSIONS:,}", "possessions"),
                stat_tile(f"{PEARSON_R:.2f}", "corr. vs. published RAPM"),
                stat_tile(TOP_PLAYER["player_name"], f"top rating: {TOP_PLAYER['rapm_per_100']:+.1f} / 100 poss"),
            ],
        ),

        # --- ratings table ----------------------------------------------
        html.Div(className="card", children=[
            section_header("Ratings", "Player net ratings",
                            "Sum of the fitted RAPM coefficient across on-court possessions. "
                            "Sortable; filter by team below."),
            dcc.Dropdown(id="team-filter", options=team_options, value="all", clearable=False,
                         style={"maxWidth": "300px", "marginBottom": "16px"}),
            dash_table.DataTable(
                id="ratings-table",
                columns=[
                    {"name": "Player", "id": "player_name"},
                    {"name": "Team", "id": "team_name"},
                    {"name": "Net rating /100 poss", "id": "rapm_per_100", "type": "numeric",
                     "format": {"specifier": "+.1f"}},
                    {"name": "Off. poss.", "id": "off_possessions"},
                    {"name": "Def. poss.", "id": "def_possessions"},
                ],
                sort_action="native",
                page_size=12,
                style_as_list_view=True,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left", "padding": "10px 14px", "fontSize": "13.5px",
                    "fontFamily": "'Inter', sans-serif", "border": "none",
                },
                style_header={
                    "fontWeight": "700", "backgroundColor": "#F1F3F9", "color": MUTED,
                    "fontSize": "11.5px", "textTransform": "uppercase", "letterSpacing": "0.04em",
                    "border": "none",
                },
                style_data={"borderBottom": f"1px solid {BORDER}"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#FAFBFD"},
                    {"if": {"filter_query": "{rapm_per_100} > 0", "column_id": "rapm_per_100"},
                     "color": POSITIVE, "fontWeight": "600"},
                    {"if": {"filter_query": "{rapm_per_100} < 0", "column_id": "rapm_per_100"},
                     "color": NEGATIVE, "fontWeight": "600"},
                ],
            ),
        ]),

        # --- rankings chart ----------------------------------------------
        html.Div(className="card", children=[
            section_header("Rankings", "Best and worst net ratings",
                            "Top/bottom 10 league-wide, or a full team when one is selected above."),
            dcc.Loading(dcc.Graph(id="rankings-chart", config={"displayModeBar": False})),
        ]),

        # --- lineup projector ----------------------------------------------
        html.Div(className="card", children=[
            section_header("Lineup tool", "Lineup projector",
                            "Pick up to 5 players from any team to project a combined net rating."),
            dcc.Dropdown(id="lineup-picker", options=player_options, multi=True,
                         placeholder="Search players by name..."),
            html.Div(id="lineup-result", style={"marginTop": "16px"}),
        ]),

        # --- validation ----------------------------------------------
        html.Div(className="card", children=[
            section_header(
                "Validation", "vs. published RAPM ratings",
                f"n={len(validation)} players matched via an ESPN↔WNBA-Stats ID crosswalk. "
                f"Pearson r={PEARSON_R:.2f}. The two models use different possession definitions "
                "and units (per-possession here vs. per-100 in the published data), so this "
                "compares rank agreement, not absolute values.",
            ),
            dcc.Loading(dcc.Graph(id="validation-chart", config={"displayModeBar": False})),
        ]),

        html.Div(
            "Source & methodology on GitHub.",
            style={"textAlign": "center", "color": MUTED, "fontSize": "13px", "marginTop": "8px"},
        ),
    ],
)


def _style_fig(fig, height):
    fig.update_layout(
        font=CHART_FONT, height=height, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=30, t=10, b=40),
        xaxis=dict(gridcolor="rgba(15,23,42,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(15,23,42,0.06)", zeroline=False),
    )
    return fig


@app.callback(Output("ratings-table", "data"), Output("rankings-chart", "figure"),
              Input("team-filter", "value"))
def update_team_view(team_id):
    df = ratings if team_id == "all" else ratings[ratings["team_id"] == team_id]
    df = df.sort_values("rapm_per_100", ascending=False)

    if team_id == "all":
        top = df.head(10)
        bottom = df.tail(10)
        chart_df = (top if len(bottom) == 0 else
                    pd.concat([top, bottom]).drop_duplicates("athlete_id"))
        chart_df = chart_df.sort_values("rapm_per_100")
    else:
        chart_df = df.sort_values("rapm_per_100")

    colors = [POSITIVE if v >= 0 else NEGATIVE for v in chart_df["rapm_per_100"]]
    labels = [f"{v:+.1f}" for v in chart_df["rapm_per_100"]]
    fig = go.Figure(
        go.Bar(
            x=chart_df["rapm_per_100"], y=chart_df["player_name"], orientation="h",
            marker=dict(color=colors, cornerradius=3),
            text=labels, textposition="outside", textfont=dict(size=12, color=INK),
            cliponaxis=False,
            hovertemplate="%{y}<br>Net rating: %{x:+.1f} / 100 poss<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Net rating per 100 possessions", yaxis_title=None, showlegend=False)
    _style_fig(fig, height=max(340, 26 * len(chart_df)))
    fig.add_vline(x=0, line_width=1, line_color="rgba(15,23,42,0.25)")
    return df.to_dict("records"), fig


@app.callback(Output("lineup-result", "children"), Input("lineup-picker", "value"))
def update_lineup(selected):
    if not selected:
        return html.Div(
            "Select up to 5 players to see a projected net rating.",
            style={"color": MUTED, "fontSize": "14px"},
        )
    if len(selected) > 5:
        return html.Div(
            f"{len(selected)} selected — pick at most 5.",
            style={"color": NEGATIVE, "fontWeight": 600},
        )
    try:
        score = project_lineup(ratings, selected)
    except ValueError as e:
        return html.Div(str(e), style={"color": NEGATIVE})

    color = POSITIVE if score >= 0 else NEGATIVE
    bg = "rgba(37,99,235,0.07)" if score >= 0 else "rgba(220,76,61,0.07)"
    return html.Div(
        style={
            "background": bg, "border": f"1px solid {color}22", "borderRadius": "10px",
            "padding": "16px 18px", "display": "flex", "alignItems": "baseline", "gap": "10px",
        },
        children=[
            html.Span(f"{score:+.1f}", style={"fontSize": "26px", "fontWeight": 800, "color": color}),
            html.Span("net rating per 100 possessions", style={"color": MUTED, "fontSize": "13.5px"}),
        ],
    )


@app.callback(Output("validation-chart", "figure"), Input("team-filter", "value"))
def update_validation_chart(_):
    lo = min(validation["rapm_published"].min(), (validation["rapm_mine"] * PER_100).min())
    hi = max(validation["rapm_published"].max(), (validation["rapm_mine"] * PER_100).max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(15,23,42,0.18)", dash="dash", width=1.5),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=validation["rapm_published"], y=validation["rapm_mine"] * PER_100,
        mode="markers", text=validation["player_name"],
        marker=dict(size=9, color=ACCENT, opacity=0.65, line=dict(width=0)),
        hovertemplate="%{text}<br>Published: %{x:.1f}<br>Mine: %{y:.1f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="Published RAPM (per 100 possessions)",
        yaxis_title="My RAPM (per 100 possessions)",
    )
    _style_fig(fig, height=440)
    return fig


if __name__ == "__main__":
    import os

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
