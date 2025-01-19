from capymoa.stream.preprocessing import MOATransformer
from capymoa.stream.preprocessing.pipeline import (ClassifierPipeline, ClassifierPipelineElement, TransformerPipelineElement, 
RegressorPipeline, RegressorPipelineElement)

from capymoa.classifier import HoeffdingTree, HoeffdingAdaptiveTree, NaiveBayes, AdaptiveRandomForestClassifier, KNN
from capymoa.regressor import AdaptiveRandomForestRegressor, KNNRegressor

from EvOAutoML.pipelinehelper import (
    PipelineHelperClassifier,
    PipelineHelperRegressor,
    PipelineHelperTransformer,
)

from moa.streams.filters import NormalisationFilter, StandardisationFilter
from capymoa.stream.preprocessing import RegressorPipeline, ClassifierPipeline

AUTOML_CLASSIFICATION_PIPELINE = ClassifierPipeline(
    (
        "Scaler",
        TransformerPipelineElement(PipelineHelperTransformer(
            [
                ("StandardScaler", MOATransformer(filter=StandardisationFilter())),
                ("MinMaxScaler", MOATransformer(filter=NormalisationFilter())),
                #("MinAbsScaler", preprocessing.MaxAbsScaler()),
            ]
        )),
    ),
    (
        "Classifier",
        ClassifierPipelineElement(PipelineHelperClassifier(
            [
                ("HT", HoeffdingTree()),
                # ('FT', EFDT()),
                #("LR", linear_model.LogisticRegression()),
                ('HAT', HoeffdingAdaptiveTree()),
                ("NB", NaiveBayes()),
                # ('PAC', PassiveAggressiveClassifier()),
                # ('ARF', AdaptiveRandomForestClassifier()),
                ("KNN", KNN()),
            ]
        )),
    ),
)

CLASSIFICATION_PARAM_GRID = {
    "Scaler": AUTOML_CLASSIFICATION_PIPELINE.steps["Scaler"].generate({}),
    "Classifier": AUTOML_CLASSIFICATION_PIPELINE.steps["Classifier"].generate(
        {
            "HT__nb_threshold": [5, 10, 20, 30],
            "HT__grace_period": [10, 100, 200, 10, 100, 200],
            "HT__tie_threshold": [0.001, 0.01, 0.05, 0.1],
            "HAT__grace_period": [10, 100, 200, 10, 100, 200],
            "KNN__k": [1, 5, 20],
            "KNN__window_size": [100, 500, 1000],
        }
    ),
}

AUTOML_REGRESSION_PIPELINE = RegressorPipeline(
    (
        "Scaler",
        TransformerPipelineElement(PipelineHelperTransformer(
            [
                ("StandardScaler", MOATransformer(filter=StandardisationFilter())),
                ("MinMaxScaler", MOATransformer(filter=NormalisationFilter())),
                #("MinAbsScaler", preprocessing.MaxAbsScaler()),
            ]
        )),
    ),
    (
        "Regressor",
        RegressorPipelineElement(PipelineHelperRegressor(
            [
                ("ARF", AdaptiveRandomForestRegressor()),
                ("KNN", KNNRegressor()),
            ]
        )),
    ),
)

REGRESSION_PARAM_GRID = {
    "Regressor": AUTOML_REGRESSION_PIPELINE.steps["Regressor"].generate(
        {
            "KNN__k": [1, 5, 20],
            "KNN__window_size": [100, 500, 1000],
            "KNN__median": [True, False],
            "ARF__ensemble_size": [50, 100, 200],
            "ARF__max_features": [0.4, 0.6, 0.8],
            "ARF__lambda_param": [1.0, 6.0, 10.0],
            "ARF__disable_background_learner": [True, False],
        }
    )
}
