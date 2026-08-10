#!/usr/bin/env bash

set -e

for SIZE in small medium large xl 10B; do

    case "$SIZE" in
        small)
            D_MODEL=768
            CONTEXT_LENGTH=512
            D_FF=3072
            NUM_LAYERS=12
            NUM_HEADS=12
            ;;
        medium)
            D_MODEL=1024
            CONTEXT_LENGTH=512
            D_FF=4096
            NUM_LAYERS=24
            NUM_HEADS=16
            ;;
        large)
            D_MODEL=1280
            CONTEXT_LENGTH=512
            D_FF=5120
            NUM_LAYERS=36
            NUM_HEADS=20
            ;;
        xl)
            D_MODEL=2560
            CONTEXT_LENGTH=512
            D_FF=10240
            NUM_LAYERS=32
            NUM_HEADS=32
            ;;
        10B)
            D_MODEL=4608
            CONTEXT_LENGTH=512
            D_FF=12288
            NUM_LAYERS=50
            NUM_HEADS=36
            ;;
    esac

    echo "============================================================"
    echo "Running model size: $SIZE"
    echo "============================================================"

    python ../profile/benchmark.py \
        model.size="$SIZE" \
        model.d_model="$D_MODEL" \
        model.context_length="$CONTEXT_LENGTH" \
        model.d_ff="$D_FF" \
        model.num_layers="$NUM_LAYERS" \
        model.num_heads="$NUM_HEADS"

done