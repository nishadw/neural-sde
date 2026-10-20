import json

import jax
import numpy as np

from ripen.data import make_splits, select
from ripen.eval import predict_all, summarize_by_group
from ripen.eval.evaluate import prediction_instances
from ripen.models import build_model
from ripen.train import TrainConfig, load_model, train


def test_train_two_steps_and_roundtrip(tmp_path, synthetic_items, synthetic_info):
    sp = make_splits(synthetic_items, seed=0)
    tr, va = select(synthetic_items, sp["train"]), select(synthetic_items, sp["val"])
    model = build_model({"name": "gru_d", "horizon": 10}, synthetic_info, jax.random.PRNGKey(0))
    cfg = TrainConfig(epochs=2, batch_size=8, max_steps=3, patience=5)
    best, hist = train(model, tr, va, synthetic_info, cfg, jax.random.PRNGKey(1), out_dir=tmp_path, log=lambda *_: None)
    assert (tmp_path / "model.eqx").exists() and (tmp_path / "history.csv").exists()
    assert len(hist) >= 1 and np.isfinite(hist[0]["train_loss"])
    template = build_model({"name": "gru_d", "horizon": 10}, synthetic_info, jax.random.PRNGKey(0))
    loaded = load_model(template, tmp_path / "model.eqx")
    from ripen.data import collate

    b = collate(va[:3], synthetic_info, n_obs=2)
    assert np.allclose(
        np.asarray(best.predict_cdf(b, jax.random.PRNGKey(0))), np.asarray(loaded.predict_cdf(b, jax.random.PRNGKey(0)))
    )


def test_stage_linear_refits_alpha(synthetic_items, synthetic_info):
    from ripen.data import collate

    model = build_model({"name": "stage_linear", "horizon": 10}, synthetic_info, jax.random.PRNGKey(0))
    fitted = model.fit_closed_form(collate(synthetic_items, synthetic_info, n_obs="all"))
    alpha = np.asarray(fitted.alpha)
    assert (alpha < 0).all()
    # faster group (T20/Tamb) -> fewer days per stage than T10
    g = {name: i for i, name in enumerate(synthetic_info.groups)}
    assert abs(alpha[g["T20"]]) < abs(alpha[g["T10"]])


def test_prediction_instances_and_summary(synthetic_items, synthetic_info, tmp_path):
    inst = prediction_instances(synthetic_items[:5])
    for i, k in inst:
        it = synthetic_items[i]
        assert it.visits[k - 1].stage < 5
    model = build_model({"name": "discrete_surv", "horizon": 10}, synthetic_info, jax.random.PRNGKey(0))
    preds = predict_all(model, synthetic_items[:5], synthetic_info, jax.random.PRNGKey(0), n_obs_list=[1, 3])
    assert preds.F.shape[0] == len(prediction_instances(synthetic_items[:5], [1, 3]))
    s = summarize_by_group(preds.F, preds.lo, preds.hi, preds.group, synthetic_info.groups, preds.item_ids)
    assert "overall" in s and "fruit_weighted" in s
    json.dumps(s)  # serialisable
