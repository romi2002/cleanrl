import optuna

from cleanrl_utils.tuner import Tuner

tuner = Tuner(
    script="cleanrl/rpo_continuous_action.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "usv-asmc-ca-v0": None,
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_loguniform("learning-rate", 0.0001, 0.003),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4, 32, 64]),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        "num-steps": trial.suggest_categorical("num-steps", [16, 32, 64, 128, 256, 512, 1024, 2048]),
        "vf-coef": trial.suggest_uniform("vf-coef", 0, 5),
        "max-grad-norm": trial.suggest_uniform("max-grad-norm", 0, 5),
        "clip-coef": trial.suggest_uniform("clip-coef", 0, 0.5),
        "gamma": trial.suggest_uniform("gamma", 0.95, 0.999),
        "total-timesteps": 300000,
        "num-envs": 8,
    },
    wandb_kwargs={"project": "cleanrl_rpo_hyperparam"},
)
tuner.tune(
    num_trials=200,
    num_seeds=1,
)