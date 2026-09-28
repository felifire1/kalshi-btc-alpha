import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_btc_dataset, load_btc_hourly_prices
from features    import build_features, get_X_y
from models      import (train_test_split_temporal,
                          train_logistic, train_random_forest,
                          train_xgboost, train_gap_regression,
                          plot_model_results)
from backtest    import run_backtest, plot_backtest
import joblib


def main():
    print("Loading data...")
    markets = load_btc_dataset()
    btc     = load_btc_hourly_prices(
                start=str(markets["open_date"].min()),
                end  =str(markets["open_date"].max())
              )

    print("Building features...")
    df, feature_cols = build_features(markets, btc)

    print("Splitting data (temporal)...")
    X, y_clf, _ = get_X_y(df, feature_cols, "y_trade_profitable")
    _, y_reg, _ = get_X_y(df, feature_cols, "y_convergence_pp")

    X_train, X_test, y_train_clf, y_test_clf = train_test_split_temporal(X, y_clf)
    _,       _,      y_train_reg, y_test_reg  = train_test_split_temporal(X, y_reg)

    n = len(X)
    print(f"  Train: {len(X_train):,} ({len(X_train)/n:.0%})  |  Test: {len(X_test):,} ({len(X_test)/n:.0%})")

    print("Training models...")
    lr_model,  lr_scaler, auc_lr,  probs_lr  = train_logistic(
        X_train, y_train_clf, X_test, y_test_clf, feature_cols)

    rf_model,  auc_rf,  probs_rf,  imp_rf    = train_random_forest(
        X_train, y_train_clf, X_test, y_test_clf, feature_cols)

    xgb_model, auc_xgb, probs_xgb, imp_xgb  = train_xgboost(
        X_train, y_train_clf, X_test, y_test_clf, feature_cols)

    reg_model, mae, r2 = train_gap_regression(
        X_train, y_train_reg, X_test, y_test_reg, feature_cols)

    plot_model_results(y_test_clf,
                       probs_lr, probs_rf, probs_xgb,
                       auc_lr,   auc_rf,   auc_xgb,
                       imp_xgb,  feature_cols)

    print("Running backtest...")
    trades = run_backtest(
        df            = df,
        model_probs   = probs_xgb,
        feature_cols  = feature_cols,
        threshold     = 0.55,
        position_size = 100.0,
        transaction_cost = 0.005
    )

    if not trades.empty:
        plot_backtest(trades)
        trades.to_csv("output/trades_log.csv", index=False)

    artifacts = {
        "rf_model":     rf_model,
        "xgb_model":    xgb_model,
        "lr_model":     lr_model,
        "lr_scaler":    lr_scaler,
        "feature_cols": feature_cols,
        "auc_rf":       auc_rf,
        "auc_xgb":      auc_xgb,
        "auc_lr":       auc_lr,
        "imp_rf":       imp_rf,
        "imp_xgb":      imp_xgb,
    }
    joblib.dump(artifacts, "output/model_artifacts.joblib")
    print("Done. Output saved to output/")


if __name__ == "__main__":
    main()
