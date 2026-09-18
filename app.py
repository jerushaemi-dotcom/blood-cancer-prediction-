import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
df = pd.read_csv("blood_cancer_symptoms_dataset.csv")

X = df.drop("Blood_Cancer", axis=1)
y = df["Blood_Cancer"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy
train_accuracy = accuracy_score(
    y_train,
    model.predict(x_train)
)

test_accuracy = accuracy_score(
    y_test,
    model.predict(x_test)
)


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        # Get values from HTML
        fatigue = int(request.form["fatigue"])
        fever = int(request.form["fever"])
        night_sweats = int(request.form["night_sweats"])
        weight_loss = int(request.form["weight_loss"])
        bruising = int(request.form["bruising"])
        bone_pain = int(request.form["bone_pain"])
        swollen_nodes = int(request.form["swollen_nodes"])
        infections = int(request.form["infections"])
        pale_skin = int(request.form["pale_skin"])
        short_breath = int(request.form["short_breath"])

        # Create input data
        input_data = pd.DataFrame([[
            fatigue,
            fever,
            night_sweats,
            weight_loss,
            bruising,
            bone_pain,
            swollen_nodes,
            infections,
            pale_skin,
            short_breath
        ]], columns=X.columns)

        # Make prediction
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "HIGH PREDICTED RISK OF BLOOD CANCER"
        else:
            result = "LOW PREDICTED RISK OF BLOOD CANCER"

    return render_template(
        "index.html",
        result=result,
        train_accuracy=round(train_accuracy * 100, 2),
        test_accuracy=round(test_accuracy * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)