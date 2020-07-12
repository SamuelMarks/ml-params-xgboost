from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from xgboost import train, XGBClassifier

from ml_params_xgboost.ml_params_impl import XGBoostTrainer


class TestMnist(TestCase):
    xgboost_datasets_dir = None  # type: str or None
    model_dir = None  # type: str or None

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.xgboost_datasets_dir = path.join(path.expanduser('~'), 'sklearn_datasets')
        TestMnist.model_dir = mkdtemp('_model_dir')

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.xgboost_datasets_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        num_classes = 10
        epochs = 200

        trainer = XGBoostTrainer()
        trainer.load_data(
            'mnist_784',
            datasets_dir=TestMnist.xgboost_datasets_dir,
            num_classes=num_classes
        )
        trainer.load_model(XGBClassifier, verbosity=3, call=True)
        trainer.train(
            epochs=epochs,
            model_dir=TestMnist.model_dir,
            optimizer=None,
            loss=None,
            metrics=None,
            callbacks=None,
            save_directory=None,
            metric_emit_freq=None
        )


if __name__ == '__main__':
    unittest_main()
