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

# --- data -------------------------------------------------------------
ratings = fit_rapm(SEASON)
ratings = ratings.dropna(subset=["player_name", "team_id"]).copy()
ratings["rapm_per_100"] = ratings["rapm"] * PER_100
validation = compare_to_published(SEASON, ratings.drop(columns=["rapm_per_100"]))

TEAM_NAMES = {
    3: "Dallas Wings", 5: "Indiana Fever", 6: "Los Angeles Sparks", 8: "Minnesota Lynx",
    9: "New York Liberty", 11: "Phoenix Mercury", 14: "Seattle Storm", 16: "Washington Mystics",
    17: "Las Vegas Aces", 18: "Connecticut Sun", 19: "Chicago Sky", 20: "Atlanta Dream",
    129689: "Golden State Valkyries",
}
ratings["team_name"] = ratings["team_id"].map(TEAM_NAMES).fillna(ratings["team_id"].astype(str))

# diverging pair for positive/negative net rating; neutral gray midpoint implied at 0
POSITIVE = "#3B7DD8"
NEGATIVE = "#D8583B"
NEUTRAL_TEXT = "rgba(20,20,20,0.75)"

team_options = [{"label": "All teams", "value": "all"}] + [
    {"label": name, "value": tid} for tid, name in sorted(TEAM_NAMES.items(), key=lambda kv: kv[1])
]
player_options = [
    {"label": f"{row.player_name} ({row.team_name})", "value": row.player_name}
    for row in ratings.sort_values("player_name").itertuples()
]

app = dash.Dash(__name__)
app.title = "WNBA Player Impact"
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>body { background: #ffffff; margin: 0; }</style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""

app.layout = html.Div(
    style={"fontFamily": "-apple-system, Helvetica, Arial, sans-serif", "maxWidth": "1100px",
           "margin": "0 auto", "padding": "24px", "background": "#ffffff", "color": "#141414"},
    children=[
        html.H1(f"WNBA Player Impact -- {SEASON} Season", style={"marginBottom": "4px"}),
        html.P(
            "RAPM-style (Regularized Adjusted Plus-Minus) player ratings fit from real "
            "play-by-play, validated against published RAPM ratings.",
            style={"color": NEUTRAL_TEXT, "marginTop": 0},
        ),

        html.H2("Player ratings"),
        dcc.Dropdown(id="team-filter", options=team_options, value="all", clearable=False,
                     style={"maxWidth": "320px", "marginBottom": "12px"}),
        dash_table.DataTable(
            id="ratings-table",
            columns=[
                {"name": "Player", "id": "player_name"},
                {"name": "Team", "id": "team_name"},
                {"name": "Net rating /100 poss", "id": "rapm_per_100", "type": "numeric",
                 "format": {"specifier": "+.1f"}},
                {"name": "Off. possessions", "id": "off_possessions"},
                {"name": "Def. possessions", "id": "def_possessions"},
            ],
            sort_action="native",
            page_size=15,
            style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "14px"},
            style_header={"fontWeight": "600", "backgroundColor": "rgba(0,0,0,0.04)"},
            style_data_conditional=[
                {"if": {"filter_query": "{rapm_per_100} > 0", "column_id": "rapm_per_100"},
                 "color": POSITIVE},
                {"if": {"filter_query": "{rapm_per_100} < 0", "column_id": "rapm_per_100"},
                 "color": NEGATIVE},
            ],
        ),

        html.H2("Top / bottom net ratings", style={"marginTop": "36px"}),
        dcc.Graph(id="rankings-chart"),

        html.H2("Lineup projector", style={"marginTop": "36px"}),
        html.P("Pick up to 5 players (any team) to project a combined net rating.",
               style={"color": NEUTRAL_TEXT}),
        dcc.Dropdown(id="lineup-picker", options=player_options, multi=True,
                     placeholder="Select up to 5 players..."),
        html.Div(id="lineup-result", style={"fontSize": "20px", "fontWeight": "600",
                                             "marginTop": "12px"}),

        html.H2("Validation vs. published RAPM ratings", style={"marginTop": "36px"}),
        html.P(
            f"n={len(validation)} players matched via an ESPN<->WNBA-Stats ID crosswalk. "
            f"Pearson r={validation[['rapm_mine', 'rapm_published']].corr().iloc[0, 1]:.2f}. "
            "The two models use different possession definitions and units (per-possession "
            "here vs. per-100 in the published data), so this compares rank agreement, not "
            "absolute values.",
            style={"color": NEUTRAL_TEXT},
        ),
        dcc.Graph(id="validation-chart"),
    ],
)


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
    fig = go.Figure(
        go.Bar(
            x=chart_df["rapm_per_100"], y=chart_df["player_name"], orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Net rating: %{x:+.1f} / 100 poss<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Net rating per 100 possessions", yaxis_title=None,
        margin=dict(l=10, r=10, t=10, b=40), height=max(320, 24 * len(chart_df)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.3)")
    return df.to_dict("records"), fig


@app.callback(Output("lineup-result", "children"), Input("lineup-picker", "value"))
def update_lineup(selected):
    if not selected:
        return ""
    if len(selected) > 5:
        return "Pick at most 5 players."
    try:
        score = project_lineup(ratings, selected)
    except ValueError as e:
        return str(e)
    color = POSITIVE if score >= 0 else NEGATIVE
    return html.Span(f"Projected net rating: {score:+.1f} / 100 possessions", style={"color": color})


@app.callback(Output("validation-chart", "figure"), Input("team-filter", "value"))
def update_validation_chart(_):
    fig = go.Figure(
        go.Scatter(
            x=validation["rapm_published"], y=validation["rapm_mine"] * PER_100,
            mode="markers", text=validation["player_name"],
            marker=dict(size=8, color=POSITIVE, opacity=0.7),
            hovertemplate="%{text}<br>Published: %{x:.1f}<br>Mine: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Published RAPM (per 100 possessions)",
        yaxis_title="My RAPM (per 100 possessions)",
        margin=dict(l=10, r=10, t=10, b=40), height=460,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
