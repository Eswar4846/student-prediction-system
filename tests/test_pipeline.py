import pytest
from sklearn.linear_model import LinearRegression

from src.data_utils import load_data, validate_training_data
from src.predict import predict_dataframe
from src.preprocessing import FEATURE_COLUMNS, prepare_features_and_target
from src.train import build_full_pipeline, train_all_models


@pytest.fixture
def student_features():
    return load_data().loc[:, FEATURE_COLUMNS].iloc[[0]].copy()


def test_dataset_is_valid_and_has_expected_target():
    data = load_data()

    assert data.shape == (395, 33)
    assert validate_training_data(data)["valid"]
    assert data["G3"].between(0, 20).all()


def test_training_data_rejects_out_of_range_target():
    data = load_data()
    data.loc[0, "G3"] = 21

    validation = validate_training_data(data)

    assert not validation["valid"]
    assert "G3" in validation["message"]


def test_prepare_features_requires_complete_schema():
    data = load_data().drop(columns=["G2"])

    with pytest.raises(ValueError, match="G2"):
        prepare_features_and_target(data)


def test_batch_prediction_validates_inputs_and_preserves_extra_columns(
    student_features, monkeypatch
):
    class ConstantModel:
        def predict(self, features):
            assert list(features.columns) == FEATURE_COLUMNS
            assert features.loc[0, "sex"] == "F"
            return [12.5] * len(features)

    monkeypatch.setattr("src.predict.load_best_pipeline", lambda: ConstantModel())
    student_features.loc[student_features.index[0], "sex"] = " F "
    student_features["source_note"] = "test"

    result = predict_dataframe(student_features)

    assert result.loc[0, "predicted_G3"] == 12.5
    assert result.loc[0, "source_note"] == "test"


def test_batch_prediction_rejects_invalid_values(student_features):
    student_features.loc[student_features.index[0], "studytime"] = 5

    with pytest.raises(ValueError, match="studytime"):
        predict_dataframe(student_features)


def test_training_compares_all_requested_models():
    output = train_all_models(load_data())

    assert set(output["results"]) == {
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
    }
    assert output["X_test"].shape[0] == 79
    for metrics in output["results"].values():
        assert set(metrics) == {"MAE", "RMSE", "R2"}


def test_preprocessor_handles_entirely_missing_input_columns():
    data = load_data()
    data["health"] = float("nan")
    data["internet"] = None
    features, target = prepare_features_and_target(data)
    pipeline = build_full_pipeline(LinearRegression())

    pipeline.fit(features, target)

    assert len(pipeline.predict(features[:1])) == 1
