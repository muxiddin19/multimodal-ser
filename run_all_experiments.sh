#!/bin/bash
# Run all experiments for ACL 2026 Multimodal SER
# This script trains our enhanced model on all available datasets

set -e

echo "============================================================"
echo "ACL 2026 Enhanced Multimodal SER - Full Experiment Suite"
echo "============================================================"

# Create results directory
mkdir -p results
mkdir -p saved_models

# Common settings
NUM_RUNS=5
EPOCHS=100
PATIENCE=15

# ============================================================
# Experiment 1: IEMOCAP 5-class (emotion2vec)
# ============================================================
echo ""
echo "============================================================"
echo "Experiment 1: IEMOCAP 5-class (emotion2vec)"
echo "============================================================"

python main_acl2026_enhanced.py \
    --train features/IEMOCAP_5class_emotion2vec_train.pkl \
    --val features/IEMOCAP_5class_emotion2vec_val.pkl \
    --test features/IEMOCAP_5class_emotion2vec_test.pkl \
    --emotion_config iemocap_5 \
    --num_classes 5 \
    --num_runs $NUM_RUNS \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --output results/iemocap_5class_enhanced.json

# ============================================================
# Experiment 2: IEMOCAP 6-class (emotion2vec)
# ============================================================
echo ""
echo "============================================================"
echo "Experiment 2: IEMOCAP 6-class (emotion2vec)"
echo "============================================================"

python main_acl2026_enhanced.py \
    --train features/IEMOCAP_6class_emotion2vec_train.pkl \
    --val features/IEMOCAP_6class_emotion2vec_val.pkl \
    --test features/IEMOCAP_6class_emotion2vec_test.pkl \
    --emotion_config iemocap_6 \
    --num_classes 6 \
    --num_runs $NUM_RUNS \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --output results/iemocap_6class_enhanced.json

# ============================================================
# Experiment 3: MELD 4-class (emotion2vec)
# ============================================================
echo ""
echo "============================================================"
echo "Experiment 3: MELD 4-class (emotion2vec)"
echo "============================================================"

python main_acl2026_enhanced.py \
    --train features/MELD_emotion2vec_train.pkl \
    --val features/MELD_emotion2vec_val.pkl \
    --test features/MELD_emotion2vec_test.pkl \
    --emotion_config meld_4 \
    --num_classes 4 \
    --num_runs $NUM_RUNS \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --output results/meld_4class_enhanced.json

# ============================================================
# Experiment 4: CREMA-D 6-class (emotion2vec)
# ============================================================
echo ""
echo "============================================================"
echo "Experiment 4: CREMA-D 6-class (emotion2vec)"
echo "============================================================"

python main_acl2026_enhanced.py \
    --train features/CREMAD_emotion2vec_train.pkl \
    --val features/CREMAD_emotion2vec_val.pkl \
    --test features/CREMAD_emotion2vec_test.pkl \
    --emotion_config cremad_6 \
    --num_classes 6 \
    --num_runs $NUM_RUNS \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --output results/cremad_6class_enhanced.json

# ============================================================
# Experiment 5: IEMOCAP 5-class with Curriculum Learning
# ============================================================
echo ""
echo "============================================================"
echo "Experiment 5: IEMOCAP 5-class with Curriculum Learning"
echo "============================================================"

python main_acl2026_enhanced.py \
    --train features/IEMOCAP_5class_emotion2vec_train.pkl \
    --val features/IEMOCAP_5class_emotion2vec_val.pkl \
    --test features/IEMOCAP_5class_emotion2vec_test.pkl \
    --emotion_config iemocap_5 \
    --num_classes 5 \
    --num_runs $NUM_RUNS \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --curriculum \
    --curriculum_phases "30,60,100" \
    --curriculum_classes "4,5,5" \
    --output results/iemocap_5class_curriculum.json

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"
echo "Results saved to:"
echo "  - results/iemocap_5class_enhanced.json"
echo "  - results/iemocap_6class_enhanced.json"
echo "  - results/meld_4class_enhanced.json"
echo "  - results/cremad_6class_enhanced.json"
echo "  - results/iemocap_5class_curriculum.json"
