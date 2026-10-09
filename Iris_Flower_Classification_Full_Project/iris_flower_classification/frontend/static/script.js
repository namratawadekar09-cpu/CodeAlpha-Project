const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const statusBox = document.getElementById("status");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const probabilitiesBox = document.getElementById("probabilities");
const predictButton = document.getElementById("predictButton");

const sampleButton = document.getElementById("sampleButton");
const resetButton = document.getElementById("resetButton");

function getInputValues() {
    return {
        sepal_length: Number(document.getElementById("sepal_length").value),
        sepal_width: Number(document.getElementById("sepal_width").value),
        petal_length: Number(document.getElementById("petal_length").value),
        petal_width: Number(document.getElementById("petal_width").value)
    };
}

function displayProbabilities(probabilities) {
    probabilitiesBox.innerHTML = "";

    Object.entries(probabilities).forEach(([species, probability]) => {
        const row = document.createElement("div");
        row.className = "probability-row";

        row.innerHTML = `
            <div class="probability-heading">
                <span>${species}</span>
                <strong>${probability}%</strong>
            </div>
            <div class="progress-track">
                <div class="progress-bar" style="width: ${probability}%"></div>
            </div>
        `;

        probabilitiesBox.appendChild(row);
    });
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    statusBox.textContent = "";
    result.classList.add("hidden");
    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    try {
        const values = getInputValues();

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(values)
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Prediction failed.");
        }

        predictionText.textContent = `Iris ${data.prediction}`;
        confidenceText.textContent = `${data.confidence}%`;
        displayProbabilities(data.probabilities);
        result.classList.remove("hidden");

    } catch (error) {
        statusBox.textContent = error.message;
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Species";
    }
});

sampleButton.addEventListener("click", () => {
    document.getElementById("sepal_length").value = 5.1;
    document.getElementById("sepal_width").value = 3.5;
    document.getElementById("petal_length").value = 1.4;
    document.getElementById("petal_width").value = 0.2;
    statusBox.textContent = "Setosa sample loaded.";
});

resetButton.addEventListener("click", () => {
    form.reset();
    result.classList.add("hidden");
    statusBox.textContent = "";
    probabilitiesBox.innerHTML = "";
});
