import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
df = sns.load_dataset("titanic")

# ----------------------------------------------------
# 2. BASIC EDA
# ----------------------------------------------------
print(df.head())
print(df.isna().sum())
print(df.describe())

sns.countplot(data=df, x="survived")
plt.show()

sns.histplot(df["age"].dropna(), kde=True)
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# ----------------------------------------------------
# 3. CLEANING (NO LEAKAGE)
# ----------------------------------------------------
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Remove columns that leak survival or are redundant
df = df.drop(columns=[
    "deck", "embark_town", "alive", "who", "adult_male", "alone"
])

# ----------------------------------------------------
# 4. ONE-HOT ENCODING
# ----------------------------------------------------
df_model = pd.get_dummies(df, drop_first=True)

X = df_model.drop("survived", axis=1)
y = df_model["survived"]

# ----------------------------------------------------
# 5. TRAIN/TEST SPLIT
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 6. MODEL TRAINING
# ----------------------------------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ----------------------------------------------------
# 7. MODEL EVALUATION
# ----------------------------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# LOAD & CLEAN DATA
# -----------------------------
df = sns.load_dataset("titanic").dropna(subset=["survived"])

# Fix missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Remove columns that leak survival or are redundant
df = df.drop(columns=[
    "deck", "embark_town", "alive", "who", "adult_male", "alone"
])

# One-hot encode
df_model = pd.get_dummies(df, drop_first=True)

# Split features/target
X = df_model.drop("survived", axis=1)
y = df_model["survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Confusion matrix figure
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted: Died", "Predicted: Survived"],
    y=["Actual: Died", "Actual: Survived"],
    colorscale="Blues",
    showscale=True
)
cm_fig.update_layout(title="Confusion Matrix")

# -----------------------------
# EDA FIGURES
# -----------------------------
fig_survival = px.histogram(df, x="survived", color="survived", title="Survival Distribution")
fig_survival.update_xaxes(tickvals=[0, 1], ticktext=["Died", "Survived"])

fig_age = px.histogram(df, x="age", nbins=30, color="survived", title="Age Distribution")

fig_class = px.histogram(df, x="class", color="survived", title="Passenger Class Distribution")

# -----------------------------
# DASH APP
# -----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Titanic Survival Analysis & Prediction"),
    html.H4("Research question: Can we predict whether a passenger survived the Titanic disaster based on demographic and ticket information?"),

    # KPI cards
    html.Div([
        html.Div([
            html.H3("Model Accuracy"),
            html.H2(f"{accuracy:.3f}")
        ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "8px", "marginRight": "20px"}),

        html.Div([
            html.H3("Dataset Size"),
            html.H2(f"{len(df)} passengers")
        ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "8px"}),
    ], style={"display": "flex", "marginBottom": "30px"}),

    # EDA
    html.Div([
        html.Div([dcc.Graph(figure=fig_survival)], style={"flex": "1", "marginRight": "10px"}),
        html.Div([dcc.Graph(figure=fig_class)], style={"flex": "1", "marginLeft": "10px"}),
    ], style={"display": "flex", "marginBottom": "30px"}),

    dcc.Graph(figure=fig_age),

    # Model performance + prediction
    html.Div([
        html.Div([
            html.H3("Model Performance"),
            dcc.Graph(figure=cm_fig)
        ], style={"flex": "1", "marginRight": "20px"}),

        html.Div([
            html.H3("Passenger Survival Prediction"),
            html.P("Adjust the inputs to simulate a passenger and estimate survival probability."),

            html.Label("Sex"),
            dcc.Dropdown(
                id="input-sex",
                options=[{"label": "Male", "value": "male"}, {"label": "Female", "value": "female"}],
                value="male",
                clearable=False
            ),

            html.Label("Age", style={"marginTop": "10px"}),
            dcc.Slider(id="input-age", min=0, max=80, step=1, value=30),

            html.Label("Passenger Class", style={"marginTop": "10px"}),
            dcc.Dropdown(
                id="input-class",
                options=[
                    {"label": "First", "value": "First"},
                    {"label": "Second", "value": "Second"},
                    {"label": "Third", "value": "Third"},
                ],
                value="Third",
                clearable=False
            ),

            html.Label("Fare", style={"marginTop": "10px"}),
            dcc.Slider(id="input-fare", min=0, max=550, step=5, value=50),

            html.Label("Siblings/Spouses Aboard (sibsp)", style={"marginTop": "10px"}),
            dcc.Slider(id="input-sibsp", min=0, max=8, step=1, value=0),

            html.Label("Parents/Children Aboard (parch)", style={"marginTop": "10px"}),
            dcc.Slider(id="input-parch", min=0, max=6, step=1, value=0),

            html.Label("Embarked", style={"marginTop": "10px"}),
            dcc.Dropdown(
                id="input-embarked",
                options=[
                    {"label": "Cherbourg (C)", "value": "C"},
                    {"label": "Queenstown (Q)", "value": "Q"},
                    {"label": "Southampton (S)", "value": "S"},
                ],
                value="S",
                clearable=False
            ),

            html.H3(id="prediction-output", style={"marginTop": "20px", "color": "#0074D9"}),
        ], style={"flex": "1", "border": "1px solid #ddd", "padding": "15px", "borderRadius": "8px"}),
    ], style={"display": "flex", "marginBottom": "30px"}),
])

# -----------------------------
# CALLBACK FOR PREDICTION
# -----------------------------
@app.callback(
    Output("prediction-output", "children"),
    Input("input-sex", "value"),
    Input("input-age", "value"),
    Input("input-class", "value"),
    Input("input-fare", "value"),
    Input("input-sibsp", "value"),
    Input("input-parch", "value"),
    Input("input-embarked", "value"),
)
def predict_survival(sex, age, pclass, fare, sibsp, parch, embarked):

    # Build row with correct dummy names
    row = {
        "age": age,
        "fare": fare,
        "sibsp": sibsp,
        "parch": parch,
        "class_Second": 1 if pclass == "Second" else 0,
        "class_Third": 1 if pclass == "Third" else 0,
        "sex_male": 1 if sex == "male" else 0,
        "embarked_Q": 1 if embarked == "Q" else 0,
        "embarked_S": 1 if embarked == "S" else 0,
    }

    # Ensure all model columns exist
    model_cols = X.columns
    row_full = {col: 0 for col in model_cols}
    for k, v in row.items():
        if k in row_full:
            row_full[k] = v

    X_new = pd.DataFrame([row_full])
    prob_survived = model.predict_proba(X_new)[0][1]

    return f"Estimated survival probability: {prob_survived:.1%}"

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
