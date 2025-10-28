#!/bin/bash
# Run all components of the BTCUSD Trading AI

echo "Setting up Python environment..."
pip install -r requirements.txt

echo "Downloading BTCUSD data..."
python src/data_collector.py

echo "Training the model..."
python src/train_model.py

echo "Running trading strategy..."
python src/trading_strategy.py

echo "Generating analysis..."
python src/analysis.py

echo "Setup complete! Check the results in the data/ and models/ directories."