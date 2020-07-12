""" Implementation of ml_params API """

from os import path
from typing import Tuple

from ml_params_sklearn import get_logger
from ml_params_sklearn.ml_params_impl import SkLearnTrainer

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))

logger.warning('Don\'t use this package, just install xgboost and use ml-params-sklearn directly; '
               'as we use its API')


class XGBoostTrainer(SkLearnTrainer):
    """ Implementation of ml_params BaseTrainer for XGBoostTrainer """


del Tuple, get_logger

__all__ = ['XGBoostTrainer']
