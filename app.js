(function () {
  const model = window.HCV_XGB_MODEL;

  const refs = {
    form: document.getElementById("riskForm"),
    statusMessage: document.getElementById("statusMessage"),
    emptyState: document.getElementById("emptyState"),
    resultContent: document.getElementById("resultContent"),
    birthYear: document.getElementById("birthYear"),
    sex: document.getElementById("sex"),
    birthplace: document.getElementById("birthplace"),
    ethnicity: document.getElementById("ethnicity"),
    incomePir: document.getElementById("incomePir"),
    resetButton: document.getElementById("resetButton"),
    exampleButton: document.getElementById("exampleButton"),
    modelName: document.getElementById("modelName"),
    cohortSize: document.getElementById("cohortSize"),
    validationAuroc: document.getElementById("validationAuroc"),
    riskValue: document.getElementById("riskValue"),
    decileValue: document.getElementById("decileValue"),
    resultNarrative: document.getElementById("resultNarrative"),
    recommendationTitle: document.getElementById("recommendationTitle"),
    recommendationBadge: document.getElementById("recommendationBadge"),
    recommendationText: document.getElementById("recommendationText"),
    decileMeanRisk: document.getElementById("decileMeanRisk"),
    decileObservedRisk: document.getElementById("decileObservedRisk"),
    decileCount: document.getElementById("decileCount"),
    decileBar: document.getElementById("decileBar"),
    decileTableBody: document.getElementById("decileTableBody")
  };

  const PRIORITY_TEST_THRESHOLD = 0.02;
  const inputFields = [
    "birthYear",
    "sex",
    "birthplace",
    "ethnicity",
    "incomePir"
  ];

  function formatNumber(value) {
    return new Intl.NumberFormat("en-US").format(value);
  }

  function formatPercent(value, digits) {
    return `${(value * 100).toFixed(digits)}%`;
  }

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function clampProbability(value) {
    return Math.min(1 - 1e-6, Math.max(1e-6, value));
  }

  function logit(value) {
    const p = clampProbability(value);
    return Math.log(p / (1 - p));
  }

  function setStatus(message) {
    refs.statusMessage.textContent = message || "";
    refs.statusMessage.classList.toggle("is-visible", Boolean(message));
  }

  function clearFieldErrors() {
    inputFields.forEach((fieldName) => {
      refs[fieldName].classList.remove("is-invalid");
    });
  }

  function markFieldError(fieldName) {
    if (refs[fieldName]) {
      refs[fieldName].classList.add("is-invalid");
    }
  }

  function clearResult() {
    refs.resultContent.classList.remove("is-entering");
    refs.resultContent.classList.add("hidden");
    refs.emptyState.classList.remove("hidden");
  }

  function animateResultReveal() {
    refs.resultContent.classList.remove("is-entering");
    void refs.resultContent.offsetWidth;
    refs.resultContent.classList.add("is-entering");
  }

  function compileTree(node) {
    if (Object.prototype.hasOwnProperty.call(node, "leaf")) {
      return {
        leaf: Number(node.leaf)
      };
    }

    const childMap = new Map();
    (node.children || []).forEach((child) => {
      childMap.set(Number(child.nodeid), compileTree(child));
    });

    return {
      split: node.split,
      split_condition: Number(node.split_condition),
      yes: Number(node.yes),
      no: Number(node.no),
      missing: Number(node.missing),
      childMap
    };
  }

  function scoreTree(node, features) {
    if (Object.prototype.hasOwnProperty.call(node, "leaf")) {
      return node.leaf;
    }

    const value = features[node.split];
    const featureValue = Math.fround(value);
    const splitValue = Math.fround(node.split_condition);
    let nextId = node.missing;
    if (!Number.isNaN(featureValue) && value !== null && value !== undefined) {
      nextId = featureValue < splitValue ? node.yes : node.no;
    }
    return scoreTree(node.childMap.get(nextId), features);
  }

  function clampPir(value) {
    return Math.max(0, Math.min(5, value));
  }

  function encodeFeatures(input) {
    return {
      birth_year: input.birthYear,
      male: input.sex === "Male" ? 1 : 0,
      race_ethnicityWhite: input.ethnicity === "White" ? 1 : 0,
      race_ethnicityBlack: input.ethnicity === "Black" ? 1 : 0,
      race_ethnicityHispanic: input.ethnicity === "Hispanic" ? 1 : 0,
      "race_ethnicityAsian/Other": input.ethnicity === "Asian/Other" ? 1 : 0,
      born_usa: input.birthplace === "USA" ? 1 : 0,
      incomepir: clampPir(input.incomePir)
    };
  }

  function getDecile(probability) {
    const thresholds = model.deciles.thresholds || [];
    let decile = 1;
    thresholds.forEach((threshold) => {
      if (probability > Number(threshold)) {
        decile += 1;
      }
    });
    return Math.min(10, decile);
  }

  function getDecileStats(decile) {
    return (model.deciles.stats || []).find((entry) => Number(entry.decile) === Number(decile));
  }

  function buildNarrative(probability, decile, stats) {
    const positionText =
      decile >= 9
        ? "This patient falls into the upper end of the modeled cohort."
        : decile >= 7
          ? "This patient falls above the middle of the modeled cohort."
          : decile >= 4
            ? "This patient falls near the middle of the modeled cohort."
            : "This patient falls below most of the modeled cohort.";

    return `${positionText} Estimated risk of current or past HCV infection is ${formatPercent(probability, 2)}, corresponding to decile ${decile} of 10. In the source cohort, observed prevalence for this infection-history outcome in the same decile was ${formatPercent(Number(stats.observed_prevalence), 2)}.`;
  }

  function buildTestingRecommendation(probability, birthYear) {
    const currentYear = new Date().getFullYear();
    const age = currentYear - birthYear;
    const isAdult = Number.isFinite(age) && age >= 18;
    const isPriority = probability >= PRIORITY_TEST_THRESHOLD;

    if (isAdult && isPriority) {
      return {
        title: "Encourage HCV status testing now",
        badge: "Higher-priority",
        isPriority,
        text: `This patient meets CDC's age range for universal one-time adult screening and also exceeds the calculator's higher-priority flag of ${formatPercent(PRIORITY_TEST_THRESHOLD, 1)} estimated risk.`
      };
    }

    if (isAdult) {
      return {
        title: "Encourage at least one HCV status test",
        badge: "Routine adult screening",
        isPriority,
        text: `CDC recommends at least one lifetime HCV screening test for adults 18 and older. This patient's score is below the calculator's higher-priority flag of ${formatPercent(PRIORITY_TEST_THRESHOLD, 1)}, but testing is still encouraged.`
      };
    }

    if (isPriority) {
      return {
        title: "Consider HCV status testing",
        badge: "Risk-based",
        isPriority,
        text: `This patient is outside the standard adult universal-screening age band but exceeds the calculator's higher-priority flag of ${formatPercent(PRIORITY_TEST_THRESHOLD, 1)} estimated risk. Use clinician judgment and exposure history.`
      };
    }

    return {
      title: "Use age and exposure context",
      badge: "Risk-based",
      isPriority,
      text: "This patient is outside the standard adult universal-screening age band and does not exceed the calculator's higher-priority flag. Consider testing based on exposures, pregnancy, or other clinical risk factors."
    };
  }

  function renderDecileBar(activeDecile) {
    refs.decileBar.innerHTML = "";
    (model.deciles.stats || []).forEach((entry) => {
      const segment = document.createElement("div");
      segment.className = "decile-segment" + (Number(entry.decile) === activeDecile ? " is-active" : "");
      segment.innerHTML = `
        <span>D${entry.decile}</span>
        <strong>${formatPercent(Number(entry.predicted_risk_mean), 1)}</strong>
      `;
      refs.decileBar.appendChild(segment);
    });
  }

  function renderDecileTable(activeDecile) {
    refs.decileTableBody.innerHTML = "";
    (model.deciles.stats || []).forEach((entry) => {
      const row = document.createElement("tr");
      if (Number(entry.decile) === activeDecile) {
        row.className = "is-selected";
      }
      row.innerHTML = `
        <td>D${entry.decile}</td>
        <td>${formatPercent(Number(entry.predicted_risk_mean), 2)}</td>
        <td>${formatPercent(Number(entry.observed_prevalence), 2)}</td>
        <td>${formatNumber(Number(entry.n))}</td>
      `;
      refs.decileTableBody.appendChild(row);
    });
  }

  function showResult(probability, decile, stats, input) {
    const recommendation = buildTestingRecommendation(probability, input.birthYear);
    refs.emptyState.classList.add("hidden");
    refs.resultContent.classList.remove("hidden");
    refs.riskValue.textContent = formatPercent(probability, 2);
    refs.decileValue.textContent = `${decile}/10`;
    refs.resultNarrative.textContent = buildNarrative(probability, decile, stats);
    refs.recommendationTitle.textContent = recommendation.title;
    refs.recommendationBadge.textContent = recommendation.badge;
    refs.recommendationBadge.classList.toggle("is-priority", recommendation.isPriority);
    refs.recommendationText.textContent = recommendation.text;
    refs.decileMeanRisk.textContent = formatPercent(Number(stats.predicted_risk_mean), 2);
    refs.decileObservedRisk.textContent = formatPercent(Number(stats.observed_prevalence), 2);
    refs.decileCount.textContent = formatNumber(Number(stats.n));
    renderDecileBar(decile);
    renderDecileTable(decile);
    animateResultReveal();
  }

  function readInputs() {
    return {
      birthYear: Number(refs.birthYear.value),
      sex: refs.sex.value,
      birthplace: refs.birthplace.value,
      ethnicity: refs.ethnicity.value,
      incomePir: Number(refs.incomePir.value)
    };
  }

  function validateInputs(input) {
    if (!Number.isFinite(input.birthYear) || input.birthYear < 1900 || input.birthYear > 2100) {
      return { message: "Enter a valid birth year.", field: "birthYear" };
    }
    if (!input.sex) {
      return { message: "Enter a valid sex.", field: "sex" };
    }
    if (!input.birthplace) {
      return { message: "Enter a valid birthplace.", field: "birthplace" };
    }
    if (!input.ethnicity) {
      return { message: "Enter a valid race/ethnicity.", field: "ethnicity" };
    }
    if (!Number.isFinite(input.incomePir)) {
      return { message: "Enter a valid income-to-poverty ratio.", field: "incomePir" };
    }
    if (input.incomePir < 0 || input.incomePir > 5) {
      return { message: "Enter a valid income-to-poverty ratio between 0 and 5.", field: "incomePir" };
    }
    return null;
  }

  function scorePatient(input) {
    const features = encodeFeatures(input);
    const rawTreeScore = compiledTrees.reduce((sum, tree) => sum + scoreTree(tree, features), 0);
    const margin = Number(model.scoring.base_margin) + rawTreeScore;
    const rawProbability = sigmoid(margin);
    const calibration = model.scoring.calibration;
    if (!calibration) {
      return rawProbability;
    }
    return sigmoid(
      Number(calibration.intercept) + Number(calibration.slope) * logit(rawProbability)
    );
  }

  function loadExample() {
    refs.birthYear.value = "1962";
    refs.sex.value = "Male";
    refs.birthplace.value = "USA";
    refs.ethnicity.value = "Black";
    refs.incomePir.value = "1.20";
    refs.form.requestSubmit();
  }

  function resetForm() {
    refs.form.reset();
    clearFieldErrors();
    setStatus("");
    clearResult();
  }

  function handleFieldInput(event) {
    event.target.classList.remove("is-invalid");
    if (refs.statusMessage.textContent) {
      setStatus("");
    }
  }

  if (!model || !model.scoring || !Array.isArray(model.scoring.trees)) {
    setStatus("Model bundle is missing. Run scripts/export_xgb_web_artifact.R to rebuild model-data.js.");
    return;
  }

  const compiledTrees = model.scoring.trees.map(compileTree);

  refs.modelName.textContent = "DEMOGRAPHIC XGB";
  refs.cohortSize.textContent = formatNumber(Number(model.cohort.size));
  refs.validationAuroc.textContent = model.reference_validation && model.reference_validation.auroc
    ? Number(model.reference_validation.auroc).toFixed(3)
    : "N/A";
  refs.birthYear.max = String(new Date().getFullYear());

  refs.form.addEventListener("submit", (event) => {
    event.preventDefault();
    clearFieldErrors();
    setStatus("");

    const input = readInputs();
    const error = validateInputs(input);
    if (error) {
      clearResult();
      setStatus(error.message);
      markFieldError(error.field);
      refs[error.field].focus();
      return;
    }

    const probability = scorePatient(input);
    const decile = getDecile(probability);
    const stats = getDecileStats(decile);

    if (!stats) {
      clearResult();
      setStatus("Unable to map this score to a cohort decile.");
      return;
    }

    showResult(probability, decile, stats, input);
  });

  refs.resetButton.addEventListener("click", resetForm);
  refs.exampleButton.addEventListener("click", loadExample);
  inputFields.forEach((fieldName) => {
    refs[fieldName].addEventListener("input", handleFieldInput);
    refs[fieldName].addEventListener("change", handleFieldInput);
  });
  requestAnimationFrame(() => {
    document.body.classList.add("is-ready");
  });
})();
