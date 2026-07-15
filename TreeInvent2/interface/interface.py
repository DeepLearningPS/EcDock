import dash
from dash import  dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Files", href="#")),
            dbc.NavItem(dbc.NavLink("Setting", id="open-setting-modal")),
        ],
        brand="TreeInvent2",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    dbc.Modal(
        [
            dbc.ModalHeader("Settings"),
            dbc.ModalBody(
                [
                    html.Div("Set your variables here:"),
                    dcc.Input(id="input-variable", type="text", placeholder="Enter variable value"),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Save", id="save-settings", className="ml-auto")
            ),
        ],
        id="setting-modal",
        is_open=False,
    ),
])

@app.callback(
    Output("setting-modal", "is_open"),
    [Input("open-setting-modal", "n_clicks"), Input("save-settings", "n_clicks")],
    [State("setting-modal", "is_open")]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)