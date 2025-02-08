from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import copy
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for Heroku deployment


def encode_image_contents(contents):
    content_type, content_string = contents.split(',')
    return content_string

def analyze_with_gpt(image_base64):
    """Send the image to GPT-4 Vision and return the response."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = """Je suis ophtalmologue et j'ai besoin que tu analyses cette image OCT maculaire. Fournis-moi une réponse JSON structurée uniquement, sans texte explicatif avant ou après.

Pour chaque œil (OD et OG), analyse les biomarqueurs suivants et retourne le résultat au format JSON :

1. DRIL (Désorganisation des couches rétiniennes internes)
   - Présence/absence
   - Si présent, décrire l'étendue

2. Œdème maculaire Cystoïde
   - Nombre de logettes
   - Taille de la plus grande logette:
     * Petite: <100μm
     * Grande: 100-200μm
     * Volumineuse: >200μm
   - Localisation: fovéolaire ou parafovéolaire

3. Intégrité des membranes
   Membrane limitante externe (MLE):
   - Continue
   - Partiellement interrompue
   - Complètement interrompue

   Zone ellipsoïde (ZE):
   - Continue
   - Partiellement interrompue
   - Complètement interrompue

4. Points hyperréflectifs
   - Présence/absence
   - Si présent:
     * Nombre approximatif
     * Localisation (intrarétinien/choroïdien)

5. Épaisseur rétinienne
   - Analyser la carte ETDRS
   - Donner les chiffres importants dans chaque secteur

POINTS IMPORTANTS:
- Ne mentionner que ce qui est clairement visible
- Éviter les surinterprétations
- Pour les kystes, ne les mentionner que s'ils sont clairement identifiables
- Rester objectif et précis dans les mesures

RETOURNER UNIQUEMENT UN OBJET JSON VALIDE AVEC LA STRUCTURE SUIVANTE:

{
    "left_eye": {
        "dril": {
            "status": "Présente/Absente",
            "extent": "description si présent"
        },
        "oedeme": {
            "status": "Présent/Absent",
            "nb_logette": "nombre",
            "taille": "petite/grande/volumineuse",
            "localisation": "fovéolaire/parafovéolaire"
        },
        "mle": "Continue/Partiellement interrompue/Complètement interrompue",
        "ze": "Continue/Partiellement interrompue/Complètement interrompue",
        "points_hyperreflectifs": {
            "status": "Présents/Absents",
            "nombre": "nombre approximatif",
            "localisation": "intrarétinien/choroïdien"
        },
        "epaisseur_retinienne": {
            "central": "valeur",
            "superieur": "valeur",
            "inferieur": "valeur",
            "nasal": "valeur",
            "temporal": "valeur"
        }
    },
    "right_eye": {
        "dril": {
            "status": "Présente/Absente",
            "extent": "description si présent"
        },
        "oedeme": {
            "status": "Présent/Absent",
            "nb_logette": "nombre",
            "taille": "petite/grande/volumineuse",
            "localisation": "fovéolaire/parafovéolaire"
        },
        "mle": "Continue/Partiellement interrompue/Complètement interrompue",
        "ze": "Continue/Partiellement interrompue/Complètement interrompue",
        "points_hyperreflectifs": {
            "status": "Présents/Absents",
            "nombre": "nombre approximatif",
            "localisation": "intrarétinien/choroïdien"
        },
        "epaisseur_retinienne": {
            "central": "valeur",
            "superieur": "valeur",
            "inferieur": "valeur",
            "nasal": "valeur",
            "temporal": "valeur"
        }
    }
}"""

    try:
        print("\n=== Sending request to GPT-4 ===")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un ophtalmologue expert en analyse d'OCT maculaire. Tu retournes uniquement des réponses au format JSON valide, sans texte explicatif."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        print("\n=== Raw GPT Response ===")
        print(json.dumps(response.model_dump(), indent=2))
        
        response_text = response.choices[0].message.content.strip()
        print("\n=== Response Text Content ===")
        print(response_text)
        
        # Extract JSON from response if wrapped in code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
            print("\n=== Extracted JSON from code block ===")
            print(response_text)
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            print("\n=== Extracted content from code block ===")
            print(response_text)
            
        try:
            parsed_response = json.loads(response_text)
            print("\n=== Successfully parsed JSON ===")
            print(json.dumps(parsed_response, indent=2))
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"\n=== JSON Parsing Error ===")
            print(f"Failed to parse: {response_text}")
            print(f"Error: {str(e)}")
            return {"error": str(e)}
            
    except Exception as e:
        print(f"\n=== API Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return {"error": str(e)}
        
    except Exception as e:
        print(f"\n=== ERROR in analyze_with_gpt ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return {"error": str(e)}

def process_gpt_response(response):
    """Process and standardize the GPT output into a format compatible with the UI."""
    print("\n=== Processing GPT Response ===")
    print("Input response:", json.dumps(response, indent=2))
    
    default_structure = {
        "left_eye": {
            "dril": {"status": "Absente", "extent": ""},
            "oedeme": {
                "status": "Absent",
                "nb_logette": "",
                "taille": "",
                "localisation": ""
            },
            "mle": "Continue",
            "ze": "Continue",
            "points_hyperreflectifs": {
                "status": "Absents",
                "nombre": "",
                "localisation": ""
            },
            "epaisseur_retinienne": {
                "central": "",
                "superieur": "",
                "inferieur": "",
                "nasal": "",
                "temporal": ""
            },
            "briding": "Absent",
            "decollement": "Absent"
        },
        "right_eye": {
            # Same structure as left_eye
        }
    }
    
    # Copy left_eye structure to right_eye
    default_structure["right_eye"] = copy.deepcopy(default_structure["left_eye"])

    if "error" in response:
        print(f"\n=== Error in response ===")
        print(f"Error: {response['error']}")
        return default_structure

    # Size standardization mapping
    taille_mapping = {
        "small": "petite",
        "medium": "grande",
        "large": "volumineuse",
        "petit": "petite",
        "grand": "grande",
        "<100": "petite",
        "100-200": "grande",
        ">200": "volumineuse"
    }

    # Process each eye
    for eye in ["left_eye", "right_eye"]:
        if eye not in response:
            response[eye] = default_structure[eye]
            continue

        eye_data = response[eye]
        
        # Standardize DRIL status
        if "dril" in eye_data:
            if isinstance(eye_data["dril"], dict):
                eye_data["dril"]["status"] = eye_data["dril"]["status"].capitalize()
                if eye_data["dril"]["status"] not in ["Présente", "Absente"]:
                    eye_data["dril"]["status"] = "Présente" if "present" in eye_data["dril"]["status"].lower() else "Absente"

        # Standardize Oedeme status and details
        if "oedeme" in eye_data:
            if isinstance(eye_data["oedeme"], dict):
                eye_data["oedeme"]["status"] = eye_data["oedeme"]["status"].capitalize()
                if eye_data["oedeme"]["status"] not in ["Présent", "Absent"]:
                    eye_data["oedeme"]["status"] = "Présent" if "present" in eye_data["oedeme"]["status"].lower() else "Absent"
                
                # Standardize taille
                if "taille" in eye_data["oedeme"]:
                    taille = eye_data["oedeme"]["taille"].lower()
                    eye_data["oedeme"]["taille"] = taille_mapping.get(taille, taille)

        # Standardize MLE/ZE values
        for field in ["mle", "ze"]:
            if field in eye_data:
                value = eye_data[field]
                if "continu" in value.lower():
                    eye_data[field] = "Continue"
                elif "partiel" in value.lower():
                    eye_data[field] = "Partiellement interrompue"
                elif "complet" in value.lower():
                    eye_data[field] = "Complètement interrompue"

        # Standardize points hyperreflectifs
        if "points_hyperreflectifs" in eye_data:
            points = eye_data["points_hyperreflectifs"]
            if isinstance(points, dict):
                points["status"] = points["status"].capitalize()
                if points["status"] not in ["Présents", "Absents"]:
                    points["status"] = "Présents" if "present" in points["status"].lower() else "Absents"

        # Format retinal thickness values
        if "epaisseur_retinienne" in eye_data:
            if isinstance(eye_data["epaisseur_retinienne"], str):
                # If it's a string, convert to structured format
                eye_data["epaisseur_retinienne"] = {
                    "central": eye_data["epaisseur_retinienne"],
                    "superieur": "",
                    "inferieur": "",
                    "nasal": "",
                    "temporal": ""
                }

        # Add missing fields with default values
        for key, default_value in default_structure[eye].items():
            if key not in eye_data:
                eye_data[key] = default_value
                print(f"Added missing key '{key}' with default value for {eye}")

    print("\n=== Processed Response ===")
    print(json.dumps(response, indent=2))
    return response

def create_eye_section(side):
    side_text = "Œil Gauche" if side.lower() == "left" else "Œil Droit"
    return dbc.Card([
        dbc.CardHeader(html.H3(side_text, className="text-primary")),
        dbc.CardBody([
            html.Div([
                html.H5("DRIL", className="text-secondary"),
                dcc.RadioItems(
                    options=['Présente', 'Absente'],
                    id=f'dril-{side.lower()}',
                    inline=True,
                    className="radio-group"
                )
            ], className="parameter-group"),

            html.Div([
                html.H5("Œdème maculaire cystoïde", className="text-secondary"),
                dcc.RadioItems(
                    options=['Présent', 'Absent'],
                    id=f'oedeme-{side.lower()}',
                    inline=True,
                    className="radio-group"
                ),
                html.Div([
                    dbc.Label("Nb de logette"),
                    dbc.Input(
                        type="text",
                        id=f'nb-logette-input-{side.lower()}',
                        placeholder="Entrer le nombre de logettes",
                        className="mb-2"
                    ),
                    dbc.Label("Taille"),
                    dbc.Input(
                        type="text",
                        id=f'taille-logette-input-{side.lower()}',
                        placeholder="Entrer la taille",
                        className="mb-2"
                    ),
                    dbc.Label("Localisation"),
                    dbc.Input(
                        type="text",
                        id=f'localisation-input-{side.lower()}',
                        placeholder="Entrer la localisation",
                        className="mb-2"
                    ),
                ], id=f'oedeme-details-{side.lower()}', style={"display": "none"})
            ], className="parameter-group"),

            html.Div([
                html.H5("Processus Rétiniens de Pontage", className="text-secondary"),
                dcc.RadioItems(
                    options=['Présent', 'Absent'],
                    id=f'briding-{side.lower()}',
                    inline=True,
                    className="radio-group"
                )
            ], className="parameter-group"),

            html.Div([
                html.H5("Intégrité de la MLE/ZE", className="text-secondary"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("MLE"),
                        dbc.Select(
                            id=f'mle-{side.lower()}',
                            options=[
                                {"label": "Continue", "value": "Continue"},
                                {"label": "Partiellement interrompue", "value": "Partiellement interrompue"},
                                {"label": "Complètement interrompue", "value": "Complètement interrompue"}
                            ]
                        )
                    ]),
                    dbc.Col([
                        dbc.Label("ZE"),
                        dbc.Select(
                            id=f'ze-{side.lower()}',
                            options=[
                                {"label": "Continue", "value": "Continue"},
                                {"label": "Partiellement interrompue", "value": "Partiellement interrompue"},
                                {"label": "Complètement interrompue", "value": "Complètement interrompue"}
                            ]
                        )
                    ])
                ])
            ], className="parameter-group"),

            html.Div([
                html.H5("Points Hyperréflectifs", className="text-secondary"),
                dcc.RadioItems(
                    options=['Présents', 'Absents'],
                    id=f'points-{side.lower()}',
                    inline=True,
                    className="radio-group"
                )
            ], className="parameter-group"),

            html.Div([
                html.H5("Décollement Séreux Rétinien", className="text-secondary"),
                dcc.RadioItems(
                    options=['Présent', 'Absent'],
                    id=f'decollement-{side.lower()}',
                    inline=True,
                    className="radio-group"
                )
            ], className="parameter-group"),

            html.Div([
                html.H5("Épaisseur Rétinienne (EDTRS)", className="text-secondary"),
                dbc.Input(
                    type="text",
                    id=f'edtrs-input-{side.lower()}',
                    placeholder="Entrer la valeur EDTRS",
                    className="mt-2"
                )
            ], className="parameter-group")
        ])
    ], className="h-100")

# Main layout modification
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("OCT Master", className="text-center text-primary my-4"))]),
    dbc.Card([
        dbc.CardBody([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    html.I(className="fas fa-upload me-2"),
                    'Glisser-déposer ou ',
                    html.A('Sélectionner une image OCT', className="text-primary")
                ]),
                className="upload-box"
            ),
            html.Div(id='output-image-upload', className="mt-3")
        ])
    ], className="mb-4"),
    dbc.Card([
        dbc.CardHeader(html.H2("Biomarqueurs", className="text-primary mb-0")),
        dbc.CardBody([
            dbc.Button(
                "Analyser avec OCT Master",
                id="analyze-button",
                color="success",
                size="lg",
                className="w-100 mb-3"
            )
        ])
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(create_eye_section("Left"), md=6, className="mb-4"),
        dbc.Col(create_eye_section("Right"), md=6, className="mb-4")
    ])
], fluid=True, className="py-4")

@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def display_uploaded_image(contents):
    if contents:
        return html.Div([
            html.Img(src=contents, style={"max-width": "100%", "height": "auto"}),
            html.Hr()
        ])
    return None

# Add callbacks for showing/hiding oedeme details
for side in ['left', 'right']:
    @app.callback(
        Output(f'oedeme-details-{side}', 'style'),
        Input(f'oedeme-{side}', 'value'),
        prevent_initial_call=True
    )
    def toggle_oedeme_details(value, side=side):
        print(f"\n=== Toggle Oedeme Details for {side} eye ===")
        print(f"Value: {value}")
        if value == 'Présent':
            return {'display': 'block'}
        return {'display': 'none'}

# Add callback to capture all form values
@app.callback(
    Output('dummy-output', 'children'),  # Dummy output for capturing values
    [Input(f'{component}-{side.lower()}', 'value') 
     for side in ['left', 'right']
     for component in ['dril', 'oedeme', 'briding', 'mle', 'ze', 'points', 'decollement']] +
    [Input(f'nb-logette-input-{side.lower()}', 'value') for side in ['left', 'right']] +
    [Input(f'taille-logette-input-{side.lower()}', 'value') for side in ['left', 'right']] +
    [Input(f'localisation-input-{side.lower()}', 'value') for side in ['left', 'right']] +
    [Input(f'edtrs-input-{side.lower()}', 'value') for side in ['left', 'right']],
    prevent_initial_call=True
)
def log_form_values(*values):
    print("\n=== Current Form Values ===")
    components = ['dril', 'oedeme', 'briding', 'mle', 'ze', 'points', 'decollement']
    
    for side in ['left', 'right']:
        print(f"\n{side.upper()} EYE:")
        # Print main components
        for i, component in enumerate(components):
            idx = (0 if side == 'left' else 7) + i
            print(f"{component}: {values[idx]}")
        
        # Print oedeme details
        base_idx = 14 + (0 if side == 'left' else 3)
        print(f"nb_logette: {values[base_idx]}")
        print(f"taille: {values[base_idx + 1]}")
        print(f"localisation: {values[base_idx + 2]}")
        print(f"edtrs: {values[base_idx + 3]}")
    
    return ""

@app.callback(
    [Output(f'{component}-{side.lower()}', 'value') 
     for side in ['left', 'right']
     for component in ['dril', 'oedeme', 'briding', 'mle', 'ze', 'points', 'decollement']] +
    [Output(f'nb-logette-input-{side.lower()}', 'value') for side in ['left', 'right']] +
    [Output(f'taille-logette-input-{side.lower()}', 'value') for side in ['left', 'right']] +
    [Output(f'localisation-input-{side.lower()}', 'value') for side in ['left', 'right']] +
    [Output(f'edtrs-input-{side.lower()}', 'value') for side in ['left', 'right']],
    Input('analyze-button', 'n_clicks'),
    [State('upload-image', 'contents')] + 
    [State(f'{component}-{side.lower()}', 'value') 
     for side in ['left', 'right']
     for component in ['briding', 'decollement']],
    prevent_initial_call=True
)
def analyze_image(n_clicks, contents, left_briding, left_decollement, right_briding, right_decollement):
    if not contents:
        return [None] * 22

    image_base64 = encode_image_contents(contents)
    raw_response = analyze_with_gpt(image_base64)
    analysis = process_gpt_response(raw_response)
    
    analysis['left_eye']['briding'] = left_briding if left_briding is not None else "Absent"
    analysis['left_eye']['decollement'] = left_decollement if left_decollement is not None else "Absent"
    analysis['right_eye']['briding'] = right_briding if right_briding is not None else "Absent"
    analysis['right_eye']['decollement'] = right_decollement if right_decollement is not None else "Absent"
    results = []
    
    # Process main components first (14 values total)
    for side in ['left', 'right']:
        eye_data = analysis[f'{side}_eye']
        main_values = [
            eye_data['dril']['status'],
            eye_data['oedeme']['status'],
            left_briding if side == 'left' else right_briding,  # Use the State values directly
            eye_data['mle'],
            eye_data['ze'],
            eye_data['points_hyperreflectifs']['status'],
            left_decollement if side == 'left' else right_decollement  # Use the State values directly
        ]
        results.extend(main_values)

    # Handle nb_logette values (positions 14-15)
    for side in ['left', 'right']:
        eye_data = analysis[f'{side}_eye']
        oedeme_data = eye_data['oedeme']
        results.append(oedeme_data['nb_logette'] if oedeme_data['status'] == 'Présent' else '')

    # Handle taille values (positions 16-17)
    for side in ['left', 'right']:
        eye_data = analysis[f'{side}_eye']
        oedeme_data = eye_data['oedeme']
        results.append(oedeme_data['taille'].lower() if oedeme_data['status'] == 'Présent' else '')

    # Handle localisation values (positions 18-19)
    for side in ['left', 'right']:
        eye_data = analysis[f'{side}_eye']
        oedeme_data = eye_data['oedeme']
        results.append(oedeme_data['localisation'] if oedeme_data['status'] == 'Présent' else '')

    # Handle EDTRS values (positions 20-21)
    for side in ['left', 'right']:
        eye_data = analysis[f'{side}_eye']
        retinal_thickness = eye_data['epaisseur_retinienne']
        
        if isinstance(retinal_thickness, dict):
            sections = {
                'central': 'Central',
                'superieur': 'Supérieur',
                'inferieur': 'Inférieur',
                'nasal': 'Nasal',
                'temporal': 'Temporal'
            }
            edtrs_parts = []
            for key, label in sections.items():
                value = retinal_thickness.get(key, '')
                if value:
                    edtrs_parts.append(f"{label}: {value}μm")
            results.append(", ".join(edtrs_parts) if edtrs_parts else '')
        else:
            results.append('')

    # Print verification
    print("\n=== Value Mapping Verification ===")
    
    # Main components verification (0-13)
    components = ['dril', 'oedeme', 'briding', 'mle', 'ze', 'points', 'decollement']
    for side in ['left', 'right']:
        base_idx = 0 if side == 'left' else 7
        print(f"\n{side.upper()} EYE MAIN VALUES:")
        for i, comp in enumerate(components):
            print(f"{comp}: {results[base_idx + i]}")

    # Details verification (14-21)
    print("\nDETAILS VALUES:")
    details = ['nb_logette', 'taille', 'localisation', 'edtrs']
    for side in ['left', 'right']:
        print(f"\n{side.upper()} EYE DETAILS:")
        for i, field in enumerate(details):
            idx = 14 + (i * 2) + (0 if side == 'left' else 1)
            print(f"{field} [{idx}]: {results[idx]}")

    return results

if not any(isinstance(child, html.Div) and child.id == 'dummy-output' for child in app.layout.children):
    app.layout.children.append(html.Div(id='dummy-output', style={'display': 'none'}))

if __name__ == '__main__':
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get("PORT", 8005))
    
    # Run server with Heroku configuration
    app.run_server(
        host='0.0.0.0',
        port=port,
        debug='PRODUCTION' not in os.environ
    )
