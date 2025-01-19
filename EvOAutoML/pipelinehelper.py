import typing

import pandas as pd
from capymoa import base, estimator
from capymoa.stream import preprocessing, tree
from capymoa.base import Classifier, Regressor
from capymoa.preprocessing import transformer
from moa.streams.filters import NormalisationFilter, StandardisationFilter

from EvOAutoML.base.utils import PipelineHelper


class PipelineHelperClassifier(PipelineHelper, Classifier):
    """
    This class is used to create a pipeline, where multiple classifiers are
    able to used in parallel. The selected classifier (`selected_model`) is
    used to make predictions as well as for training.
    The other classifiers are not trained in parallel.

    Parameters
    ----------
    models: dict
        A dictionary of models that can be used in the pipeline.
    selected_model: Estimator
        the model that is used for training and prediction.
    """
    @classmethod
    def _unit_test_params(cls):
        models = [
            ("HT", classifier.HoeffdingTree()),
            ("EFDT", classifier.EFDT()),
        ]
        yield {
            "models": models,
        }

    def train(
        self, x: dict, y: type_alias.Label, **kwargs
    ) -> Estimator:
        self.selected_model = self.selected_model.train(x=x, y=y, **kwargs)
        return self

    def predict(self, x: dict) -> type_alias.Label:
        return self.selected_model.predict(x)


    def predict_proba(
        self, x: dict
    ) -> typing.Dict[type_alias.LabelProbabilities, float]:
        return self.selected_model.predict_proba(x=x)

    #def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        #return self.selected_model.predict_proba_many(X=X)

    #def predict_many(self, X: pd.DataFrame) -> pd.Series:
        #return self.selected_model.predict_many(X=X)


class PipelineHelperTransformer(PipelineHelper, MOATransformer):
    """
    Add some Text here
    """

    @classmethod
    def _unit_test_params(cls):
        models = [
            ("MinMax", MOATransformer(fiter=NormalisationFilter())),
            ("NORM", MOATransformer(StandardisationFilter())),
        ]
        yield {
            "models": models,
        }

    @property
    def _supervised(self):
        if self.selected_model._supervised:
            return True
        else:
            return False

    def transform(self, x: dict) -> dict:
        """

        Args:
            x:

        Returns:

        """
        return self.selected_model.transform(x=x)

    def train(
        self, x: dict, y: type_alias.TargetValue = None, **kwargs
    ) -> "Transformer":
        """
        Add second text here
        Args:
            x:
            y:
            **kwargs:

        Returns:
            self
        """
        #if self.selected_model._supervised:
        self.selected_model = self.selected_model.train(x, y)
        #else:
            #self.selected_model = self.selected_model.train(x)
        return self


class PipelineHelperRegressor(PipelineHelper, Regressor):
    @classmethod
    def _unit_test_params(cls):
        models = [
            ("SOKNLBT", SOKNLBT()),
            ("KNNRegressor", KNNRegressor()),
        ]
        yield {
            "models": models,
        }

    def train(
        self, x: dict, y: type_alias.TargetValue, **kwargs
    ) -> Estimator:
        self.selected_model = self.selected_model.train(x=x, y=y, **kwargs)
        return self

    def predict(self, x: dict) -> type_alias.TargetValue:
        return self.selected_model.predict(x=x)

