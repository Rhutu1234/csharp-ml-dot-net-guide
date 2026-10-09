# ML.NET

*A deep-dive walkthrough of ML.NET — Microsoft's open-source, cross-platform machine learning framework for .NET — covering `MLContext`, data loading and `IDataView`, preprocessing and feature engineering, how training pipelines (estimators and transformers) actually work, model training and evaluation, saving/loading models, making predictions safely in production, AutoML, and complete worked examples for classification, regression, clustering, recommendation systems, and anomaly detection.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [ML.NET Fundamentals](#1-mlnet-fundamentals)
3. [MLContext](#2-mlcontext)
4. [Data Loading](#3-data-loading)
5. [IDataView](#4-idataview)
6. [Data Preprocessing](#5-data-preprocessing)
7. [Feature Engineering](#6-feature-engineering)
8. [Training Pipelines: Estimators and Transformers](#7-training-pipelines-estimators-and-transformers)
9. [Model Training](#8-model-training)
10. [Model Evaluation](#9-model-evaluation)
11. [Model Saving and Loading](#10-model-saving-and-loading)
12. [Model Prediction](#11-model-prediction)
13. [Classification](#12-classification)
14. [Regression](#13-regression)
15. [Clustering](#14-clustering)
16. [Recommendation Systems](#15-recommendation-systems)
17. [Anomaly Detection](#16-anomaly-detection)
18. [AutoML](#17-automl)
19. [Common Pitfalls](#18-common-pitfalls)
20. [Quick Reference Table](#quick-reference-table)
21. [Conclusion](#conclusion)

---

## Introduction

ML.NET lets .NET developers build, train, and run machine learning models **entirely in C# or F#**, without switching to Python or leaving the .NET ecosystem. A model trained with ML.NET is just a file you load into an ordinary .NET application — a web API, a worker service, a desktop app — and call like any other dependency.

```csharp
var mlContext = new MLContext(seed: 0);

IDataView data = mlContext.Data.LoadFromTextFile<HouseData>("houses.csv", hasHeader: true, separatorChar: ',');

var pipeline = mlContext.Transforms.Concatenate("Features", "Size", "Bedrooms")
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price"));

ITransformer model = pipeline.Fit(data);             // TRAINING
var engine = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
var result = engine.Predict(new HouseData { Size = 1800, Bedrooms = 3 });   // INFERENCE
```

Those six lines contain the whole ML.NET workflow: **load data → build a pipeline → fit it → predict.** Everything else in this guide is detail on one of those steps. If the vocabulary here (features, labels, training vs. inference, overfitting, precision/recall) is unfamiliar, this series' *AI/ML Fundamentals* guide covers the concepts; this guide focuses on how to apply them in .NET.

---

## 1. ML.NET Fundamentals

### What ML.NET is — and where it fits

ML.NET is a machine learning library for .NET. It provides data loading, transformation, a catalog of training algorithms, evaluation tools, and model persistence, all behind a consistent API. It runs on Windows, Linux, and macOS, and is designed for **integrating ML into .NET applications** rather than for research.

```plaintext
Strong fit:
  - Classic ML on tabular and text data (classification, regression, clustering,
    recommendation, anomaly detection, time series)
  - Teams that are .NET-first and want ML inside their existing services
  - Training and serving in the same language, with no separate Python service
  - Using models trained elsewhere via ONNX import, then serving them from .NET

Weaker fit:
  - Cutting-edge deep learning research and large-scale neural network training
    (the Python ecosystem — PyTorch, TensorFlow — is far richer here)
  - Needing the very latest model architectures as soon as they're published
```

An honest framing: ML.NET is the pragmatic choice for classic ML inside .NET applications. For deep learning, a common pattern is to train in Python, export to **ONNX**, and consume the model from .NET (ML.NET and ONNX Runtime both support this).

### The tasks ML.NET supports

```plaintext
Task                      Example question                          Catalog
------------------------  ----------------------------------------  ---------------------------
Binary classification     Is this review positive?                  mlContext.BinaryClassification
Multiclass classification Which category is this ticket?            mlContext.MulticlassClassification
Regression                What will this house sell for?            mlContext.Regression
Clustering                Which customers behave alike?             mlContext.Clustering
Recommendation            Which movies will this user like?         mlContext.Recommendation()
Anomaly detection         Is this data point unusual?               mlContext.AnomalyDetection / Transforms.Detect*
Ranking / Forecasting     Order results; predict future values      mlContext.Ranking / Forecasting
```

### Installing

```plaintext
dotnet add package Microsoft.ML                  # core: data, transforms, common trainers
dotnet add package Microsoft.ML.AutoML           # AutoML (Section 17)
dotnet add package Microsoft.ML.Recommender      # matrix factorization (Section 15)
dotnet add package Microsoft.ML.TimeSeries       # time-series anomaly detection (Section 16)
dotnet add package Microsoft.ML.FastTree         # gradient-boosted tree trainers
dotnet add package Microsoft.Extensions.ML       # PredictionEnginePool for ASP.NET Core (Section 11)
```

### The universal workflow

```plaintext
1. Create an MLContext
2. Load data into an IDataView
3. Split into train / test sets
4. Build a pipeline: preprocessing transforms + a trainer
5. Fit the pipeline on the training set  -> a trained model (ITransformer)
6. Evaluate on the test set
7. Save the model to a file
8. Load it in your application and make predictions
```

---

## 2. MLContext

### The single entry point to everything in ML.NET

```csharp
var mlContext = new MLContext(seed: 0);
```

`MLContext` is the factory and the catalog root. Every operation hangs off it:

```plaintext
mlContext.Data           -> loading, saving, splitting, filtering data
mlContext.Transforms     -> preprocessing and feature engineering
mlContext.Regression     -> regression trainers and evaluation
mlContext.BinaryClassification / MulticlassClassification
mlContext.Clustering / AnomalyDetection / Ranking / Forecasting
mlContext.Recommendation()    -> recommendation trainers
mlContext.Model          -> save / load / create prediction engines
mlContext.Auto()         -> AutoML (needs Microsoft.ML.AutoML)
```

### The `seed` parameter matters for reproducibility

```plaintext
Passing a seed makes operations that involve randomness (data shuffling in
splits, some trainers' initialization) produce repeatable results — which
is essential when comparing two pipeline variants, or when a teammate needs
to reproduce your numbers. Without a seed, results can differ slightly run to run.
```

`MLContext` also exposes logging (`mlContext.Log += ...`) for observing what trainers are doing, and it should generally be **created once and reused** rather than constructed repeatedly.

---

## 3. Data Loading

### Describe your data as a C# class, then load it

```csharp
public class HouseData
{
    [LoadColumn(0)] public float Size { get; set; }
    [LoadColumn(1)] public float Bedrooms { get; set; }
    [LoadColumn(2)] public string Neighborhood { get; set; }
    [LoadColumn(3)] public float Price { get; set; }       // the label we want to predict
}

public class HousePrediction
{
    [ColumnName("Score")] public float Price { get; set; } // regression trainers write predictions to "Score"
}
```

```csharp
IDataView data = mlContext.Data.LoadFromTextFile<HouseData>(
    path: "houses.csv",
    hasHeader: true,
    separatorChar: ',');
```

Notes worth knowing precisely:

```plaintext
- [LoadColumn(n)] maps a class property to the n-th column in the file
  (zero-based). [LoadColumn(1, 5)] maps a RANGE of columns into a float[] .
- Use float, not double, for numeric columns — it's ML.NET's native numeric type.
- [ColumnName("Label")] renames a column; ML.NET trainers look for a column
  named "Label" by default (and "Features" for the feature vector).
```

### Other sources

```csharp
// From an in-memory collection (great for tests and small data)
IDataView fromList = mlContext.Data.LoadFromEnumerable(houses);

// From a database (requires the Microsoft.ML package's database loader support)
var loader = mlContext.Data.CreateDatabaseLoader<HouseData>();
var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString,
                                  "SELECT Size, Bedrooms, Neighborhood, Price FROM Houses");
IDataView fromDb = loader.Load(dbSource);
```

Loading from SQL Server is a natural fit for .NET teams — your training data is often already in a database (see this series' SQL guides). Whichever source you use, the result is the same type: an `IDataView`.

### Splitting into train and test sets

```csharp
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
IDataView trainData = split.TrainSet;
IDataView testData  = split.TestSet;
```

```plaintext
Split BEFORE any fitting, so the test set never influences training.
For time-ordered data, do NOT use a random split — split by time instead
(train on the past, test on the future), or you'll leak future information.
```

---

## 4. IDataView

### ML.NET's lazy, schema-aware, columnar table abstraction

`IDataView` is the type that flows through every part of ML.NET — what loaders produce, what transforms consume and emit, and what trainers learn from. Think of it as a **read-only, lazily-evaluated table with a schema.**

```plaintext
Key properties:
  - LAZY: nothing is read or computed until something iterates over it.
    Building a pipeline doesn't touch the data; only Fit/Transform-and-iterate does.
  - FORWARD-ONLY cursor access: designed to stream data, so datasets larger
    than memory can be processed.
  - SCHEMA-ful: every column has a name and a type (float, string, vector, key...).
  - IMMUTABLE: transforms produce a NEW IDataView layered on the previous one;
    the original is never modified.
```

### Inspecting an IDataView

```csharp
// Peek at the first rows and the schema — the standard debugging move
var preview = data.Preview(maxRows: 5);
foreach (var row in preview.RowView)
    Console.WriteLine(string.Join(", ", row.Values.Select(kv => $"{kv.Key}={kv.Value}")));

// Pull a single column out
IEnumerable<float> prices = data.GetColumn<float>("Price");

// Convert back to a strongly typed C# collection
var houses = mlContext.Data.CreateEnumerable<HouseData>(data, reuseRowObject: false).ToList();
```

### Why laziness matters (and bites)

```plaintext
Because an IDataView is lazy, a transform that is expensive (text featurization,
say) is RE-COMPUTED every time something iterates the data — and iterative
trainers iterate many times. Cache after expensive steps:

  pipeline.AppendCacheCheckpoint(mlContext).Append(trainer)

(See Section 7. Skip caching for very large data that won't fit in memory.)
```

Also: `reuseRowObject: false` in `CreateEnumerable` is essential if you store the results — with `true`, the same object is overwritten on each iteration, so a collected list would contain many references to one final row's values.

---

## 5. Data Preprocessing

### Cleaning and converting raw columns into model-ready form

Preprocessing transforms live in `mlContext.Transforms`. Each one is an **estimator** that you chain into a pipeline (Section 7).

```csharp
// Missing values: replace with the column's mean (default) before training
mlContext.Transforms.ReplaceMissingValues("Size", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)

// Categorical text -> numbers: one-hot encode
mlContext.Transforms.Categorical.OneHotEncoding("NeighborhoodEncoded", "Neighborhood")

// Scale numeric features to a comparable range
mlContext.Transforms.NormalizeMinMax("Features")        // squashes to [0, 1]
mlContext.Transforms.NormalizeMeanVariance("Features")  // zero mean, unit variance

// Map a string label to the "key" type multiclass trainers require
mlContext.Transforms.Conversion.MapValueToKey("Label", "Category")
```

### Filtering rows

```csharp
// Drop obviously bad rows (e.g. impossible prices) BEFORE training
IDataView cleaned = mlContext.Data.FilterRowsByColumn(
    data, columnName: "Price", lowerBound: 10_000, upperBound: 5_000_000);
```

### Why these steps exist

```plaintext
- Models consume NUMBERS. Strings (neighborhood names) must be encoded.
- Missing values can crash or silently distort a trainer; decide a policy explicitly.
- Features on wildly different scales (Size ~ 2000, Bedrooms ~ 3) can make
  gradient-based trainers converge slowly or favor the large-valued feature.
  Normalization puts them on equal footing.
- Tree-based trainers (FastTree, LightGBM) are far less sensitive to scale, so
  normalization matters most for linear models and neural-network-style learners.
```

---

## 6. Feature Engineering

### Turning raw columns into the single `Features` vector trainers consume

Every ML.NET trainer expects **one vector column** (named `Features` by default) containing all the model's inputs. `Concatenate` builds it:

```csharp
mlContext.Transforms.Concatenate("Features", "Size", "Bedrooms", "NeighborhoodEncoded")
```

### Text featurization — raw text to numeric features in one step

```csharp
mlContext.Transforms.Text.FeaturizeText("Features", "ReviewText")
```

`FeaturizeText` tokenizes, normalizes, and converts text into a numeric vector of word and character n-gram features — it's the standard starting point for any text classification problem, such as sentiment analysis (Section 12).

### Other useful feature-engineering transforms

```plaintext
Transforms.NormalizeBinning(...)           -> bucket a numeric column into ranges
Transforms.Categorical.OneHotHashEncoding  -> hash-encode HIGH-cardinality categories
                                              (thousands of distinct values) compactly
Transforms.Text.ProduceWordBags(...)       -> bag-of-words / n-grams
Transforms.Conversion.ConvertType(...)     -> change a column's data type
Transforms.SelectColumns / DropColumns     -> keep only what you need
Transforms.FeatureSelection.*              -> select the most informative features
Transforms.CustomMapping(...)              -> your own C# logic (with a caveat below)
```

### A caveat on `CustomMapping`

```plaintext
CustomMapping lets you run arbitrary C# per row — flexible, but a model that
contains one CANNOT simply be saved and loaded in another process unless the
mapping is registered through a contract assembly. Prefer a built-in transform,
or do the derived-feature computation in your data-loading code BEFORE ML.NET
sees the data, so the saved model stays self-contained.
```

### Feature engineering is usually where the gains are

```plaintext
Good features typically move model quality more than choosing a fancier algorithm:
  - Derive "price per square foot" or "age of house" from raw columns
  - Extract "day of week" or "hour" from a timestamp
  - Combine related fields into ratios or flags
Do this thoughtfully, and avoid features that wouldn't exist at prediction time
(data leakage — see this series' AI/ML Fundamentals guide).
```

---

## 7. Training Pipelines: Estimators and Transformers

### The two core abstractions

```plaintext
IEstimator<T>   — a RECIPE for a step. It hasn't seen data yet.
                  It may need to LEARN something from data (the mean to
                  normalize by, the vocabulary to encode, a model's weights).

ITransformer    — the RESULT of fitting an estimator. It has learned its
                  parameters and can now TRANSFORM data.

estimator.Fit(trainingData)  ->  ITransformer
transformer.Transform(data)  ->  IDataView with new/changed columns
```

### A pipeline is a chain of estimators

```csharp
var pipeline = mlContext.Transforms.ReplaceMissingValues("Size")
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("NeighborhoodEncoded", "Neighborhood"))
    .Append(mlContext.Transforms.Concatenate("Features", "Size", "Bedrooms", "NeighborhoodEncoded"))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .AppendCacheCheckpoint(mlContext)                                    // cache the preprocessed data
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", featureColumnName: "Features"));

ITransformer model = pipeline.Fit(trainData);   // fits EVERY step, in order, on trainData
```

### Why this design is valuable

```plaintext
1. ONE object captures preprocessing AND the model. When you save the trained
   model (Section 10), the normalization parameters, encoders, and trainer all
   travel together — so predictions in production apply EXACTLY the same
   transformations as training. This eliminates a classic bug: preprocessing
   in production subtly differing from preprocessing at training time.

2. Fitting happens on the TRAINING data only. The normalization mean/range is
   learned from the training set and merely APPLIED to the test set —
   preventing test-set information from leaking into training.

3. Pipelines are lazy and composable: define once, Fit, Transform, reuse.
```

`AppendCacheCheckpoint` stores the data computed up to that point in memory so iterative trainers don't recompute upstream transforms on every pass. Place it **after** the expensive preprocessing and **before** the trainer. For datasets too large for memory, omit it.

---

## 8. Model Training

### Fitting the pipeline is the training step

```csharp
ITransformer model = pipeline.Fit(trainData);
```

`Fit` runs the whole pipeline over the training data: transforms learn their parameters, and the trainer runs its optimization to learn the model's weights. The returned `ITransformer` is your **trained model** (preprocessing + predictor together).

### Choosing a trainer

ML.NET's catalog offers many trainers per task. A practical way to choose:

```plaintext
Start simple and fast:
  - Linear trainers (Sdca*, Lbfgs*, Sgd*): fast, a good baseline, work well on
    sparse/text data; benefit from normalization.
Then try stronger:
  - Tree-ensemble trainers (FastTree*, FastForest*, LightGbm*): often the best
    accuracy on tabular data; less sensitive to scaling; handle non-linear patterns.
```

```csharp
// Swapping a trainer is a ONE-LINE change — the rest of the pipeline is untouched
.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price"))        // linear baseline
.Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Price"))    // gradient-boosted trees
```

That one-line interchangeability is a major practical strength: you can compare several algorithms against the same preprocessing and the same test set with minimal code (or let AutoML do it, Section 17).

### Training takes the label by name

```plaintext
Trainers take labelColumnName and featureColumnName. If your label column is
literally "Label" and features are in "Features", you can omit them. Otherwise
pass them explicitly — a mismatched column name is the most common
"Schema mismatch" error beginners hit.
```

---

## 9. Model Evaluation

### Evaluate on the held-out test set, using the task-appropriate metrics

```csharp
IDataView predictions = model.Transform(testData);          // run the trained model on the TEST data
RegressionMetrics metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Price");

Console.WriteLine($"R²:   {metrics.RSquared:F3}");
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F0}");
Console.WriteLine($"MAE:  {metrics.MeanAbsoluteError:F0}");
```

Each task has its own evaluator and metrics:

```plaintext
Task                    Evaluate call                                   Key metrics
----------------------  ----------------------------------------------  -----------------------------------------
Regression              mlContext.Regression.Evaluate                   RSquared, RootMeanSquaredError, MeanAbsoluteError
Binary classification   mlContext.BinaryClassification.Evaluate         Accuracy, AreaUnderRocCurve, F1Score,
                                                                        PositivePrecision, PositiveRecall
Multiclass              mlContext.MulticlassClassification.Evaluate     MicroAccuracy, MacroAccuracy, LogLoss
Clustering              mlContext.Clustering.Evaluate                   AverageDistance, DaviesBouldinIndex
Anomaly detection       mlContext.AnomalyDetection.Evaluate             AreaUnderRocCurve, DetectionRateAtFalsePositiveCount
```

### Cross-validation for a more reliable estimate

```csharp
var cvResults = mlContext.Regression.CrossValidate(
    data, pipeline, numberOfFolds: 5, labelColumnName: "Price");

double avgR2 = cvResults.Average(r => r.Metrics.RSquared);
```

A single split can be lucky or unlucky; cross-validation trains and evaluates across several folds and lets you look at both the average *and* the spread.

### Evaluation principles (the same ones from this series' AI/ML Fundamentals guide)

```plaintext
- NEVER evaluate on the training data — it measures memorization.
- Compare against a baseline; a "good" metric means little without one.
- Look at the train-vs-test gap: great training metrics with poor test metrics
  means overfitting.
- On imbalanced classification, don't trust accuracy alone — check precision,
  recall, F1, and AUC.
```

---

## 10. Model Saving and Loading

### A trained model is a `.zip` file you can ship

```csharp
// Save: the model AND the schema of the data it was trained on
mlContext.Model.Save(model, trainData.Schema, "model.zip");

// Load (in the same app, or — typically — in a different one)
var loadMlContext = new MLContext();
ITransformer loadedModel = loadMlContext.Model.Load("model.zip", out DataViewSchema inputSchema);
```

The saved file contains the **entire pipeline** — preprocessing steps with their learned parameters, plus the trained predictor — so the consuming application doesn't need the training data or the training code, only the file and the matching C# input/output classes.

```plaintext
Practical guidance:
  - Treat model.zip as a build/deployment artifact, versioned like any other.
    Name or folder it by version (model-v3.zip) so a rollback is trivial.
  - Keep the input/output classes (HouseData, HousePrediction) in a shared
    project referenced by both the training and the consuming app, so the
    schema can't drift apart.
  - Models can also be saved/loaded via Stream — useful for storing them in
    blob storage or a database rather than on local disk.
```

---

## 11. Model Prediction

### Single predictions: `PredictionEngine`

```csharp
var engine = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);

HousePrediction result = engine.Predict(new HouseData { Size = 1800, Bedrooms = 3, Neighborhood = "Westside" });
Console.WriteLine($"Predicted price: {result.Price:C0}");
```

### ⚠️ `PredictionEngine` is NOT thread-safe

```plaintext
A PredictionEngine holds internal state and must not be shared across threads.
Creating a new one per request is wasteful (it's relatively expensive to build).
In a web application, using a single shared PredictionEngine instance — a very
common mistake — produces race conditions and wrong results under concurrent load.
```

### The ASP.NET Core answer: `PredictionEnginePool`

```csharp
// Program.cs  (requires the Microsoft.Extensions.ML package)
builder.Services.AddPredictionEnginePool<HouseData, HousePrediction>()
    .FromFile(modelName: "HouseModel", filePath: "model.zip", watchForChanges: true);
```

```csharp
// In a controller / minimal API endpoint — inject the pool
app.MapPost("/predict", (HouseData input, PredictionEnginePool<HouseData, HousePrediction> pool) =>
{
    var prediction = pool.Predict(modelName: "HouseModel", example: input);
    return Results.Ok(prediction.Price);
});
```

The pool manages thread-safe engine reuse for you, integrates with dependency injection, and with `watchForChanges: true` automatically reloads the model when the file changes — so you can deploy a retrained model without restarting the app.

### Batch predictions: `Transform`

```csharp
IDataView batch = mlContext.Data.LoadFromEnumerable(newHouses);
IDataView scored = model.Transform(batch);
var results = mlContext.Data.CreateEnumerable<HousePrediction>(scored, reuseRowObject: false).ToList();
```

For scoring many rows at once (nightly jobs, bulk imports), `Transform` over an `IDataView` is much more efficient than looping a `PredictionEngine`.

---

## 12. Classification

### Binary classification: a yes/no question — sentiment analysis

```csharp
public class SentimentData
{
    [LoadColumn(0)] public string Text { get; set; }
    [LoadColumn(1), ColumnName("Label")] public bool Sentiment { get; set; }   // true = positive
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")] public bool IsPositive { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
```

```csharp
var data = mlContext.Data.LoadFromTextFile<SentimentData>("reviews.tsv", hasHeader: true);
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
        labelColumnName: "Label", featureColumnName: "Features"));

var model = pipeline.Fit(split.TrainSet);

var predictions = model.Transform(split.TestSet);
var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

Console.WriteLine($"Accuracy:  {metrics.Accuracy:P1}");
Console.WriteLine($"AUC:       {metrics.AreaUnderRocCurve:F3}");
Console.WriteLine($"F1:        {metrics.F1Score:F3}");
Console.WriteLine($"Precision: {metrics.PositivePrecision:F3}   Recall: {metrics.PositiveRecall:F3}");

var engine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
var result = engine.Predict(new SentimentData { Text = "Absolutely loved it, would buy again!" });
Console.WriteLine($"{(result.IsPositive ? "Positive" : "Negative")} ({result.Probability:P0})");
```

Logistic-regression trainers output a **probability**, not just a yes/no — you can apply your own threshold. If a false positive and a false negative cost different amounts (see precision vs. recall in the AI/ML Fundamentals guide), tune that threshold rather than accepting the default 0.5.

### Multiclass classification: one of several categories — ticket routing

```csharp
public class TicketData
{
    [LoadColumn(0)] public string Area { get; set; }    // the label: "Billing", "Bug", "Feature", ...
    [LoadColumn(1)] public string Title { get; set; }
}

public class TicketPrediction
{
    [ColumnName("PredictedLabelText")] public string Area { get; set; }
    public float[] Score { get; set; }
}
```

```csharp
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(TicketData.Area))   // string -> key
    .Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(TicketData.Title)))
    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
    // map the predicted key back to readable text, in a NEW column so evaluation can still use the key
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelText", "PredictedLabel"));

var model = pipeline.Fit(split.TrainSet);
var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(split.TestSet), labelColumnName: "Label");

Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy:P1}");   // overall fraction correct
Console.WriteLine($"Macro-accuracy: {metrics.MacroAccuracy:P1}");   // average per-class accuracy
```

```plaintext
Micro- vs macro-accuracy: micro-accuracy is dominated by the most common classes;
macro-accuracy weights every class equally. If rare classes matter, watch MACRO —
a large gap between the two means the model is doing well mainly on the common classes.
```

---

## 13. Regression

### Predicting a number — house prices

This is the pipeline built up through Sections 3–9, in one place:

```csharp
var data  = mlContext.Data.LoadFromTextFile<HouseData>("houses.csv", hasHeader: true, separatorChar: ',');
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

var pipeline = mlContext.Transforms.ReplaceMissingValues(nameof(HouseData.Size))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("NeighborhoodEncoded", nameof(HouseData.Neighborhood)))
    .Append(mlContext.Transforms.Concatenate("Features",
        nameof(HouseData.Size), nameof(HouseData.Bedrooms), "NeighborhoodEncoded"))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .AppendCacheCheckpoint(mlContext)
    .Append(mlContext.Regression.Trainers.Sdca(
        labelColumnName: nameof(HouseData.Price), featureColumnName: "Features"));

var model   = pipeline.Fit(split.TrainSet);
var metrics = mlContext.Regression.Evaluate(model.Transform(split.TestSet), labelColumnName: nameof(HouseData.Price));

Console.WriteLine($"R²: {metrics.RSquared:F3}   RMSE: {metrics.RootMeanSquaredError:F0}   MAE: {metrics.MeanAbsoluteError:F0}");
```

### Reading the regression metrics

```plaintext
MAE   (mean absolute error):  the average size of the miss, in the label's own units.
                              "On average we're off by about $18,000."  Easy to explain.
RMSE  (root mean squared):    like MAE but penalizes LARGE misses much more heavily.
                              RMSE much bigger than MAE => a few very bad predictions.
R²    (coefficient of determination): fraction of the variation in price the model
                              explains. 1.0 = perfect; 0 = no better than always
                              predicting the average; NEGATIVE = worse than that.
```

Always judge these against the scale of the label: an RMSE of 18,000 is excellent for million-dollar homes and terrible for $50,000 ones. And as always, compare against a baseline (predicting the training-set average) to see how much the model genuinely adds.

---

## 14. Clustering

### Grouping similar items without any labels — customer segmentation

```csharp
public class CustomerData
{
    [LoadColumn(0)] public float Age { get; set; }
    [LoadColumn(1)] public float AnnualIncome { get; set; }
    [LoadColumn(2)] public float SpendingScore { get; set; }
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")] public uint ClusterId { get; set; }
    [ColumnName("Score")] public float[] Distances { get; set; }   // distance to each cluster centroid
}
```

```csharp
var data = mlContext.Data.LoadFromTextFile<CustomerData>("customers.csv", hasHeader: true, separatorChar: ',');

var pipeline = mlContext.Transforms.Concatenate("Features",
        nameof(CustomerData.Age), nameof(CustomerData.AnnualIncome), nameof(CustomerData.SpendingScore))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))       // CRITICAL for distance-based clustering
    .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

var model = pipeline.Fit(data);

var predictions = model.Transform(data);
var metrics = mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");
Console.WriteLine($"Average distance: {metrics.AverageDistance:F3}   Davies-Bouldin: {metrics.DaviesBouldinIndex:F3}");

var engine = mlContext.Model.CreatePredictionEngine<CustomerData, ClusterPrediction>(model);
var result = engine.Predict(new CustomerData { Age = 34, AnnualIncome = 72_000, SpendingScore = 61 });
Console.WriteLine($"Cluster {result.ClusterId}");
```

### Things specific to clustering

```plaintext
- NORMALIZE. K-means works on distances; an unscaled feature like income
  (tens of thousands) swamps age (tens) and effectively decides the clusters alone.
- You must CHOOSE numberOfClusters. There's no label to tell you the "right" k.
  Try several values and compare metrics: lower AverageDistance is tighter clusters
  (but always improves as k grows), and a LOWER Davies-Bouldin index means better-
  separated clusters. Pick where adding clusters stops helping much.
- The metrics only say the clusters are TIGHT and SEPARATED — not that they are
  MEANINGFUL. A person still has to look at what each cluster contains
  (average age, income, spending) and decide if the groups make business sense.
- Cluster IDs are arbitrary labels: "cluster 2" has no inherent meaning, and the
  numbering can change between training runs.
```

---

## 15. Recommendation Systems

### Predicting how much a user would like an item — matrix factorization

ML.NET's recommender uses **matrix factorization**: it learns a small vector for each user and each item such that their dot product approximates the rating the user would give. Training data is simply (user, item, rating) triples.

```csharp
public class MovieRating
{
    [LoadColumn(0)] public float userId { get; set; }
    [LoadColumn(1)] public float movieId { get; set; }
    [LoadColumn(2)] public float Label { get; set; }       // the rating
}

public class MovieRatingPrediction
{
    public float Label { get; set; }
    public float Score { get; set; }                       // the predicted rating
}
```

```csharp
using Microsoft.ML.Trainers;     // for MatrixFactorizationTrainer.Options

var data  = mlContext.Data.LoadFromTextFile<MovieRating>("ratings.csv", hasHeader: true, separatorChar: ',');
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "userIdEncoded",
    MatrixRowIndexColumnName    = "movieIdEncoded",
    LabelColumnName             = "Label",
    NumberOfIterations          = 20,
    ApproximationRank           = 100
};

var pipeline = mlContext.Transforms.Conversion.MapValueToKey("userIdEncoded", "userId")
    .Append(mlContext.Transforms.Conversion.MapValueToKey("movieIdEncoded", "movieId"))
    .Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

var model = pipeline.Fit(split.TrainSet);

var metrics = mlContext.Regression.Evaluate(model.Transform(split.TestSet), labelColumnName: "Label", scoreColumnName: "Score");
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F3}");

// Predict: how would user 6 rate movie 10?
var engine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
Console.WriteLine($"Predicted rating: {engine.Predict(new MovieRating { userId = 6, movieId = 10 }).Score:F1}");
```

### Turning predicted ratings into recommendations

```plaintext
The model scores ONE (user, item) pair at a time. To produce a "top 10 for user 6"
list, score that user against EVERY candidate item (usually excluding items they've
already rated) and sort descending by Score.
```

### Honest limitations

```plaintext
- COLD START: a brand-new user or item has no ratings, so the model has nothing to
  learn from — you need a fallback (popular items, content-based rules).
- Matrix factorization learns only from IDs and ratings; it doesn't use item
  attributes or text. (ML.NET also offers a field-aware factorization machine
  trainer for combining extra features.)
- Evaluate with RMSE on held-out ratings, but remember: low RMSE doesn't guarantee
  GOOD recommendations — a ranking quality metric and, ideally, online testing
  matter more for a real product.
```

---

## 16. Anomaly Detection

ML.NET supports two quite different styles of anomaly detection, and choosing the right one matters.

### Style 1: Time-series spike detection — unusual values in a sequence

Best for metrics over time: sales, latency, sensor readings, error counts.

```csharp
using Microsoft.ML.Data;

public class MetricPoint
{
    public float Value { get; set; }
}

public class SpikePrediction
{
    [VectorType(3)] public double[] Prediction { get; set; }   // [alert, score, p-value]
}
```

```csharp
var points = LoadMetricPoints();                                   // List<MetricPoint>, in time order
IDataView dataView = mlContext.Data.LoadFromEnumerable(points);

var pipeline = mlContext.Transforms.DetectIidSpike(
    outputColumnName: nameof(SpikePrediction.Prediction),
    inputColumnName:  nameof(MetricPoint.Value),
    confidence: 95.0,
    pvalueHistoryLength: points.Count / 4);

// These detectors learn as they stream; fit on an EMPTY dataset to create the transformer
ITransformer model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<MetricPoint>()));

IDataView transformed = model.Transform(dataView);
var results = mlContext.Data.CreateEnumerable<SpikePrediction>(transformed, reuseRowObject: false).ToList();

for (int i = 0; i < results.Count; i++)
    if (results[i].Prediction[0] == 1)                             // index 0: 1 = anomaly alert
        Console.WriteLine($"Spike at point {i}: value={points[i].Value}, p-value={results[i].Prediction[2]:F4}");
```

```plaintext
The 3-element output vector is:  [0] alert (1 = anomaly)   [1] score   [2] p-value
- DetectIidSpike     : spikes in data assumed independent and identically distributed
- DetectIidChangePoint / DetectChangePointBySsa : a persistent SHIFT in level, not a one-off spike
- DetectSpikeBySsa   : spikes in data with SEASONALITY (daily/weekly cycles), via
                       singular spectrum analysis — needs window-size settings
Higher confidence => fewer alerts but more are real; lower => more alerts, more noise.
```

### Style 2: Unsupervised PCA anomaly detection — unusual rows in tabular data

Best for records described by several numeric features (transactions, device telemetry) with no time dimension.

```csharp
var pipeline = mlContext.Transforms.Concatenate("Features", "Amount", "Hour", "ItemCount")
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.AnomalyDetection.Trainers.RandomizedPca(featureColumnName: "Features", rank: 2));

var model = pipeline.Fit(normalTrainingData);                      // train primarily on NORMAL data
var predictions = model.Transform(newData);
// output columns: PredictedLabel (bool: true = anomaly) and Score (higher = more anomalous)
```

```plaintext
PCA learns what "normal" looks like (the main directions of variation). Rows that
reconstruct poorly from those directions — that sit far from the normal pattern —
score as anomalies. Keep rank smaller than the number of features.
```

### Reality checks for anomaly detection

```plaintext
- "Anomalous" means STATISTICALLY UNUSUAL, not WRONG. A Black Friday sales spike is
  an anomaly and is perfectly legitimate. Anomaly detectors surface candidates for
  a human or a rule to judge.
- Alert fatigue is the real failure mode: too many false alarms and people ignore
  all of them. Tune confidence/thresholds against how many alerts a team can handle.
- Evaluating is hard because true anomalies are rare; if you have even a small
  labeled set, use mlContext.AnomalyDetection.Evaluate with it.
```

---

## 17. AutoML

### Let ML.NET try multiple trainers and settings for you

AutoML automates the repetitive part of model selection: it runs many combinations of preprocessing and trainers against your data and reports the best one.

```csharp
using Microsoft.ML.AutoML;       // package: Microsoft.ML.AutoML

var experiment = mlContext.Auto().CreateRegressionExperiment(maxExperimentTimeInSeconds: 60);
var result = experiment.Execute(split.TrainSet, labelColumnName: nameof(HouseData.Price));

RunDetail<RegressionMetrics> best = result.BestRun;
Console.WriteLine($"Best trainer: {best.TrainerName}");
Console.WriteLine($"Validation R²: {best.ValidationMetrics.RSquared:F3}");

// Always confirm on the untouched test set
var testMetrics = mlContext.Regression.Evaluate(best.Model.Transform(split.TestSet), labelColumnName: nameof(HouseData.Price));
```

```plaintext
AutoML experiments exist for: regression, binary classification, multiclass
classification, recommendation, and ranking — created via
mlContext.Auto().Create<Task>Experiment(...).
```

### Other ways to use AutoML

```plaintext
- Model Builder (a Visual Studio extension): a GUI wizard — pick a scenario, point to
  your data, train, and it generates the C# consumption code.
- ML.NET CLI (the `mlnet` tool): run AutoML from the command line and generate
  a trained model and project.
```

### Where AutoML helps — and where it doesn't

```plaintext
Great for:
  - Getting a strong baseline quickly, and learning which trainer families suit your data
  - Teams with .NET skills but limited ML experience

Doesn't replace:
  - Understanding your data. AutoML optimizes the metric you give it; it can't tell you
    that you have data leakage, a mislabeled column, or the WRONG metric for your business.
  - Feature engineering and data cleaning, which usually matter more than trainer choice.
  - A final check on a held-out test set — the validation score AutoML reports was used
    to pick the winner, so it's slightly optimistic.
```

AutoML's API surface has evolved across ML.NET versions (including newer, more customizable experiment APIs alongside the one shown here), so check the official documentation for the version you install.

---

## 18. Common Pitfalls

| Pitfall | Why it hurts | Better approach |
|---|---|---|
| Sharing one `PredictionEngine` across threads / requests | It isn't thread-safe; concurrent use gives race conditions and wrong predictions | Use `PredictionEnginePool` in ASP.NET Core (Section 11) |
| Creating a new `PredictionEngine` per request | Building one is relatively expensive and wastes performance | Pool and reuse engines via `PredictionEnginePool` (Section 11) |
| Evaluating on the training data | The metric reflects memorization, not real-world performance | Split first; evaluate on the held-out test set (Section 9) |
| Fitting preprocessing on the whole dataset before splitting | Test-set statistics leak into training and inflate scores | Split first, then `Fit` the pipeline on the training set only (Section 7) |
| Column-name mismatches ("Schema mismatch" errors) | Trainers look for `Label` / `Features` by default; renamed columns aren't found | Pass `labelColumnName` / `featureColumnName` explicitly, or use `[ColumnName]` (Section 8) |
| Using `double` instead of `float` for numeric columns | ML.NET's native numeric type is `float`; mismatches cause schema errors or needless conversions | Use `float` in input/output classes (Section 3) |
| Forgetting to normalize for linear and distance-based models | Large-scale features dominate; clustering results are driven by one feature | Add `NormalizeMinMax` / `NormalizeMeanVariance` before the trainer (Sections 5, 14) |
| Collecting `CreateEnumerable` results with `reuseRowObject: true` | The same object is overwritten each iteration, so the list holds duplicates of the last row | Pass `reuseRowObject: false` when storing results (Section 4) |
| Skipping `AppendCacheCheckpoint` on expensive pipelines | Lazy `IDataView` recomputes transforms on every trainer pass | Cache after costly preprocessing when the data fits in memory (Section 7) |
| Using `CustomMapping` and then failing to load the model elsewhere | The saved model references code that isn't available in the consuming process | Prefer built-in transforms, or precompute derived features before ML.NET (Section 6) |
| Trusting accuracy on imbalanced classification | A majority-class guesser scores high while finding nothing | Check precision, recall, F1, and AUC (Section 9, 12) |
| Treating cluster output as automatically meaningful | Tight, separated clusters may still be business-meaningless; IDs are arbitrary | Inspect each cluster's contents and validate with domain knowledge (Section 14) |
| Acting on every anomaly flag | "Unusual" isn't "wrong"; too many alerts cause alert fatigue | Tune confidence, route flags to review, and add business rules (Section 16) |
| Ignoring cold start in recommenders | New users/items have no ratings to learn from | Provide a popularity or content-based fallback (Section 15) |
| Treating AutoML's validation score as the final answer | It was used to select the winner, so it's optimistic | Confirm on an untouched test set (Section 17) |
| Never retraining the deployed model | Inference doesn't learn; the model goes stale as data drifts | Monitor production quality and retrain on a schedule (Section 10) |

---

## Quick Reference Table

| Concept | API | Purpose |
|---|---|---|
| Entry point | `new MLContext(seed: 0)` | Factory for all ML.NET operations; seed for reproducibility |
| Load a file | `mlContext.Data.LoadFromTextFile<T>(path, hasHeader, separatorChar)` | Read CSV/TSV into an `IDataView` |
| Load in-memory | `mlContext.Data.LoadFromEnumerable(list)` | Wrap a C# collection as an `IDataView` |
| Split data | `mlContext.Data.TrainTestSplit(data, testFraction)` | Separate training and test sets |
| Inspect data | `data.Preview(maxRows)` | Peek at schema and rows |
| Handle missing | `Transforms.ReplaceMissingValues(col)` | Fill in missing numeric values |
| Encode categories | `Transforms.Categorical.OneHotEncoding(out, in)` | Turn text categories into numbers |
| Scale features | `Transforms.NormalizeMinMax("Features")` | Put features on a comparable range |
| Build features | `Transforms.Concatenate("Features", cols...)` | Combine columns into one feature vector |
| Featurize text | `Transforms.Text.FeaturizeText("Features", "Text")` | Text to numeric n-gram features |
| Chain steps | `estimator.Append(next)` | Compose a pipeline |
| Cache | `.AppendCacheCheckpoint(mlContext)` | Avoid recomputing upstream transforms |
| Train | `pipeline.Fit(trainData)` | Produce the trained `ITransformer` |
| Apply model | `model.Transform(data)` | Score a whole `IDataView` |
| Evaluate | `mlContext.<Task>.Evaluate(predictions, ...)` | Compute task-specific metrics |
| Cross-validate | `mlContext.<Task>.CrossValidate(data, pipeline, folds)` | More reliable metric estimate |
| Save / load | `mlContext.Model.Save(...)` / `.Load(...)` | Persist the whole pipeline as a `.zip` |
| Single prediction | `mlContext.Model.CreatePredictionEngine<TIn,TOut>(model)` | One-off prediction (not thread-safe) |
| Web serving | `AddPredictionEnginePool<TIn,TOut>().FromFile(...)` | Thread-safe, DI-friendly prediction in ASP.NET Core |
| Binary classifier | `BinaryClassification.Trainers.SdcaLogisticRegression(...)` | Yes/no predictions with probability |
| Multiclass | `MulticlassClassification.Trainers.SdcaMaximumEntropy(...)` | One-of-many categories |
| Regression | `Regression.Trainers.Sdca(...)` / `FastTree(...)` | Predict a number |
| Clustering | `Clustering.Trainers.KMeans("Features", k)` | Group unlabeled items |
| Recommendation | `Recommendation().Trainers.MatrixFactorization(options)` | Predict user–item ratings |
| Spike detection | `Transforms.DetectIidSpike(...)` | Flag unusual points in a time series |
| PCA anomalies | `AnomalyDetection.Trainers.RandomizedPca(...)` | Flag unusual rows in tabular data |
| AutoML | `mlContext.Auto().Create<Task>Experiment(...)` | Automatically search trainers and settings |

---

## Conclusion

ML.NET's design rests on a small number of ideas that, once clear, make every task feel familiar: an `MLContext` that is the root of everything, an `IDataView` that is the lazy, schema-aware table flowing through the system, and a pipeline of estimators that you `Fit` once to get a single transformer carrying preprocessing and model together. Because the whole pipeline is one saveable artifact, the transformations applied in production are guaranteed to match training — which removes one of the most common and hardest-to-spot sources of ML bugs. And because swapping a trainer is a one-line change, the five task types in this guide — classification, regression, clustering, recommendation, and anomaly detection — share one workflow rather than being five separate skills.

The framework's value is greatest when you're honest about where it fits: classic ML on tabular and text data, running inside .NET services, with models that load like any other dependency. The rest is discipline that no library can supply for you — splitting data before fitting anything, evaluating on data the model hasn't seen, choosing metrics that reflect what mistakes actually cost, treating cluster and anomaly outputs as candidates for human judgment rather than answers, and retraining as the world changes. Combine that discipline with ML.NET's consistent API, `PredictionEnginePool` for safe serving, and AutoML for a fast baseline, and you can take a model from a CSV file to a production endpoint without ever leaving C#.

---

*Found this useful? Feel free to star the repo, open an issue with corrections, or share the "one shared PredictionEngine gave random answers under load" story that made the case for PredictionEnginePool click better than any documentation page.*
