# House Price Prediction (scikit-learn + TensorFlow)

A small reference project showing **deep learning regression** with:
- **scikit-learn**: train/test split, feature scaling, metrics
- **TensorFlow (Keras)**: neural network model

Dataset included: `data/house_price_regression_dataset.csv`

## Project structure

```
house_price_dl_repo/
  data/
    house_price_regression_dataset.csv
  src/
    train_tf.py
    predict.py
    train_sklearn_mlp.py
  models/            # generated artifacts (ignored by git)
  notebooks/
  requirements.txt
  README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train (TensorFlow / Keras)

```bash
python -m src.train_tf --data data/house_price_regression_dataset.csv
```

Artifacts saved to:
- `models/house_price_model.keras`
- `models/scaler.joblib`

## Predict

Feature order (matches the CSV columns excluding the target):

1. Square_Footage
2. Num_Bedrooms
3. Num_Bathrooms
4. Year_Built
5. Lot_Size
6. Garage_Size
7. Neighborhood_Quality

Example:

```bash
python -m src.predict \
  --model models/house_price_model.keras \
  --scaler models/scaler.joblib \
  --features 2000,3,2,2005,1.2,1,7
```

## Alternative: scikit-learn-only MLPRegressor

```bash
python -m src.train_sklearn_mlp --data data/house_price_regression_dataset.csv
```

## Notes
- Neural networks usually need **feature scaling** to train well.
- For best results, tune layer sizes, learning rate, and early stopping patience.

