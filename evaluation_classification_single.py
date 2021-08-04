from multiprocessing import Pool
from pathlib import Path

from river import tree, ensemble, linear_model, naive_bayes, neighbors
from tqdm import tqdm

from EvOAutoML.classification import EvolutionaryBestClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    ROLLING_CLASSIFICATION_TRACKS, N_SAMPLES, N_CHECKPOINTS
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    POPULATION_SIZE

from EvOAutoML.utils import plot_track

def evaluate_single(track_dict):
    track = track_dict[1]
    plot_track(
        track=track,
        metric_name="Accuracy",
        models={
            'EvoAutoML': EvolutionaryBestClassifier(population_size=POPULATION_SIZE, estimator=AUTOML_CLASSIFICATION_PIPELINE,
                                                    param_grid=CLASSIFICATION_PARAM_GRID, sampling_rate=250),
            # 'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
            ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
            'Hoeffding Tree': tree.HoeffdingTreeClassifier(),
            # 'FT',:tree.ExtremelyFastDecisionTreeClassifier(),
            'Logistic Regression': linear_model.LogisticRegression(),
            # ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
            'GaussianNB': naive_bayes.GaussianNB(),
            # ('MNB', naive_bayes.MultinomialNB()),
            # ('PAC', linear_model.PAClassifier()),
            'KNN': neighbors.KNNClassifier(),
        },
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        result_path=Path(f'./results/classification/evaluation_single'),
        verbose=2)

if __name__ == '__main__':
    #evaluate_single(CLASSIFICATION_TRACKS[0])
    pool = Pool(60)  # Create a multiprocessing Pool
    output = pool.map(evaluate_single, CLASSIFICATION_TRACKS)
