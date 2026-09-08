#!/usr/bin/env bash
# Importance-Criterion-Ablation, VALIDATION-Variante.
#
# Identische Konfigurationen wie ../criterion, aber Evaluation auf dem Clotho-
# VALIDATION-Split (eval_subset: val). Diese Ablation entscheidet, welches
# Importance-Kriterium in allen Folgeexperimenten verwendet wird -- eine
# Auswahlentscheidung darf nicht auf dem Reporting-Split getroffen werden.
#
# 8 Punkte (enc {30,50,60,70}, dec {50,70,80,90}) x (random 3 Seeds + wanda
# 3 Seeds + sum_l2 deterministisch) = 56 Laeufe.
# Kalibrierung (nur wanda) unveraendert aus Clotho-dev (Trainingssplit).
# Idempotent: Laeufe mit vorhandener eval.json werden uebersprungen.
# Aufruf aus dem Repo-Root: bash experiments/pruning/ablation/criterion_val/run_criterion_val.sh
set -euo pipefail

BASE="experiments/pruning/ablation/criterion_val"
VAL_AUDIO="data/CLOTHO_v2.1/clotho_audio_files/validation"

# Preflight: der Validation-Split muss lokal vorhanden sein. Ohne diesen Check
# laeuft aac_datasets in einen Download-Versuch oder einen spaeten Fehler.
if [ ! -d "$VAL_AUDIO" ]; then
    echo "FEHLER: Clotho-Validation-Audio nicht gefunden unter $VAL_AUDIO"
    echo "Der Split muss vorhanden sein (er wird auch fuer die KD-Checkpointauswahl benutzt)."
    exit 1
fi
echo "== Clotho-val: $(ls "$VAL_AUDIO" | wc -l) Audiodateien gefunden"

found=0
for run_dir in "$BASE"/*/; do
    name=$(basename "$run_dir")
    [ -f "$run_dir/config.yaml" ] || continue
    found=$((found + 1))
    if [ -e "$run_dir/eval.json" ]; then
        echo "== [skip] $name (eval.json vorhanden)"
    else
        echo "== [run ] $name"
        python src/run_all.py --config "$run_dir/config.yaml"
    fi
done

if [ "$found" -eq 0 ]; then
    echo "FEHLER: keine Config-Ordner unter $BASE gefunden."
    echo "Vom Repo-Root aufrufen und sicherstellen, dass die Configs gepullt sind."
    exit 1
fi

echo "== Fertig: $found Konfigurationen. Ergebnisse: $BASE/*/eval.json"
echo "== Danach lokal: python src/export_latex_tables.py  (schaltet automatisch auf val um)"
