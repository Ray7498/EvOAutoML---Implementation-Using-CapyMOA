import sys
import os

# Specify the actual path to the "project" directory
project_dir = r"C:\Users\ayraa\OneDrive\Desktop\Labs M2\Data Stream Processing Lab\project\CapyMOA\src"  # Update this path as per your system

# Temporarily add the "project" directory to sys.path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    # Import capymoa specifically from the project directory
    import capymoa
    from capymoa.stream.preprocessing import pipeline, transformer
    from capymoa import ensemble, evaluation
    from capymoa.base import classifier
    from capymoa.datasets import Covtype
finally:
    # Restore sys.path to its original state
    if project_dir in sys.path:
        sys.path.remove(project_dir)


from EvOAutoML import classification, pipelinehelper
from moa.streams.filters import NormalisationFilter, StandardisationFilter


dataset = Covtype()
model_pipeline = pipeline.ClassifierPipeline(
    ('Scaler', pipeline.TransformerPipelineElement(pipelinehelper.PipelineHelperTransformer([
        ('StandardScaler', transformer.MOATransformer(filter=StandardisationFilter())),
        ('MinMaxScaler', transformer.MOATransformer(filter=NormalisationFilter())),
    ]))),
    ('Classifier', pipeline.ClassifierPipelineElement(pipelinehelper.PipelineHelperClassifier([
        ('HT', classifier.HoeffdingTree()),
        ('HAT', classifier.HoeffdingAdaptiveTree()),
        ('KNN', classifier.KNN()),
    ]))))
model = classification.EvolutionaryBaggingClassifier(
    model=model_pipeline,
    param_grid={
        'Scaler': model_pipeline.steps['Scaler'].generate({}),
        'Classifier': model_pipeline.steps['Classifier'].generate({
            "HT__nb_threshold": [5, 10, 20, 30],
            "HT__grace_period": [10, 100, 200, 10, 100, 200],
            "HT__tie_threshold": [0.001, 0.01, 0.05, 0.1],
            "HAT__grace_period": [10, 100, 200, 10, 100, 200],
            "KNN__k": [1, 5, 20],
            "KNN__window_size": [100, 500, 1000],
        })
    },
    seed=42
)
ev = evaluation.ClassificationEvaluator()
f1_index = ev.metrics_header().index("f1_score")
for x, y in dataset:
    y_pred = model.predict(x)  # make a prediction
    ev.update(y, y_pred)  # update the metric
    f1_score = ev.metrics()[f1_index]
    model = model.train(x,y)  # make the model learn
