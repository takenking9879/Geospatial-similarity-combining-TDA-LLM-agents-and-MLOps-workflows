from sklearn.base import BaseEstimator, TransformerMixin
from gtda.base import PlotterMixin
import ripserplusplus as rpp
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from numbers import Real
from typing import Callable
from gtda.utils.intervals import Interval
from gtda.utils.validation import validate_params, check_point_clouds


def _postprocess_diagrams(Xt, format, homology_dimensions, infinity_values, reduced):
    def replace_infinity_values(subdiagram):
        np.nan_to_num(subdiagram, posinf=infinity_values, copy=False)
        return subdiagram[subdiagram[:, 0] < subdiagram[:, 1]]

    if format in ["ripser", "flagser"]:
        slices = {dim: slice(None) if (dim or not reduced) else slice(None, -1)
                  for dim in homology_dimensions}
        Xt = [{dim: replace_infinity_values(diagram[dim][slices[dim]])
               for dim in homology_dimensions} for diagram in Xt]
    elif format == "gudhi":
        slices = {dim: slice(None) if (dim or not reduced) else slice(1, None)
                  for dim in homology_dimensions}
        Xt = [{dim: replace_infinity_values(
            np.array([pers_info[1] for pers_info in diagram
                      if pers_info[0] == dim]).reshape(-1, 2)[slices[dim]]
        ) for dim in homology_dimensions} for diagram in Xt]
    else:
        raise ValueError(f"Unknown input format {format} for collection of diagrams.")

    start_idx_per_dim = np.cumsum(
        [0] + [np.max([len(diagram[dim]) for diagram in Xt] + [1])
               for dim in homology_dimensions]
    )
    min_values = [min([np.min(diagram[dim][:, 0]) if diagram[dim].size else np.inf
                       for diagram in Xt]) for dim in homology_dimensions]
    min_values = [val if val != np.inf else 0 for val in min_values]
    n_features = start_idx_per_dim[-1]
    Xt_padded = np.empty((len(Xt), n_features, 3), dtype=float)

    for i, dim in enumerate(homology_dimensions):
        start_idx, end_idx = start_idx_per_dim[i:i + 2]
        padding_value = min_values[i]
        Xt_padded[:, start_idx:end_idx, 2] = dim
        for j, diagram in enumerate(Xt):
            subdiagram = diagram[dim]
            end_idx_nontrivial = start_idx + len(subdiagram)
            Xt_padded[j, start_idx:end_idx_nontrivial, :2] = subdiagram
            Xt_padded[j, end_idx_nontrivial:end_idx, :2] = [padding_value] * 2

    return Xt_padded


class VietorisRipsPersistencePP(BaseEstimator, TransformerMixin, PlotterMixin):
    """
    Clase que hace Ripser++ compatible con giotto-tda para su uso en pipelines de scikit-learn.
    """
    _hyperparameters = {
        "metric": {"type": (str, Callable)},
        "homology_dimensions": {
            "type": (list, tuple),
            "of": {"type": int, "in": Interval(0, np.inf, closed="left")}
        },
        "max_edge_length": {"type": Real},
        "infinity_values": {"type": (Real, type(None))},
        "reduced_homology": {"type": bool}
    }

    def __init__(self, metric="euclidean",
                 homology_dimensions=(0, 1),
                 max_edge_length=np.inf, infinity_values=None,
                 reduced_homology=True, n_jobs=None):
        self.metric = metric
        self.homology_dimensions = homology_dimensions
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.reduced_homology = reduced_homology
        self.n_jobs = n_jobs

    def _ripser_diagram(self, X):
        ripser_format = []
        pairs = rpp.run("--format point-cloud --dim " + str(self._max_homology_dimension), X)
        for i in range(len(pairs)):
            pairs[i] = np.array(pairs[i].tolist())
            if i == 0:
                inf_point = np.array([[0.0, np.inf]])
                ripser_format.append(np.vstack([pairs[i], inf_point]))
            else:
                ripser_format.append(pairs[i])
        return ripser_format

    def fit(self, X, y=None):
        validate_params(self.get_params(), self._hyperparameters, exclude=["n_jobs"])
        self._is_precomputed = self.metric == "precomputed"
        check_point_clouds(X, accept_sparse=True, distance_matrices=self._is_precomputed)

        self.infinity_values_ = self.max_edge_length if self.infinity_values is None else self.infinity_values
        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True, distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._ripser_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(
            Xt, "ripser", self._homology_dimensions, self.infinity_values_,
            self.reduced_homology
        )
        return Xt
