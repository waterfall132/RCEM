#!/usr/bin/env bash
cd "${WORK:-$(dirname "$0")}"; export NDM_FAST_SVD=1
DATA=${DATA:-/root/data}
M=$1; DS=$2; ENC=$3; K=$4; SEEDS=$5; shift 5; EXTRA="$*"
case $DS in CUB)  D=$DATA/CUB/For_FewShot;  DN=CubBird ;;
            Dogs) D=$DATA/Dogs/For_FewShot; DN=StanfordDog ;;
            Cars) D=$DATA/Cars;             DN=StanfordCar ;;
            mini) D=$DATA/miniImageNet;     DN=miniImageNet ;;
            *) echo "unknown dataset $DS"; exit 1 ;; esac
PROT="--epochs 30 --episodes 10000 --val_episodes 1000 --optim sgd --no_nesterov --dampening 0.9 --clip_grad ${CLIP:-0}"
case $ENC in R12) EM=ResNet12_ReLU;      OPT="$PROT --lr 0.01" ;;
             C4)  EM=Conv64F_Local_ReLU; OPT="$PROT --lr 0.05" ;; esac
TR=runs/routeB; [ "$DS-$ENC" = CUB-R12 ] || TR=runs/routeB_${DS}_${ENC}; mkdir -p $TR
P=${DS}_${ENC}_${K}s_dn4${TAG}
done_ () { python - "$1" <<'PY'
import json,sys,os
try:
    h=len(json.load(open(os.path.join(sys.argv[1],'summary.json')))['history']); e=json.load(open(os.path.join(sys.argv[1],'config.json')))['epochs']
    sys.exit(0 if h>=e else 1)
except Exception: sys.exit(1)
PY
}
for s in $SEEDS; do
  if [ "$M" = rcem ]; then o=$TR/${P}_B_eta05_s$s; done_ $o && { echo "skip (done) $o"; continue; }
    python routeB/train_routeB.py --dataset_dir $D --data_name $DN --encoder_model $EM --shot $K $EXTRA \
        --eta_G 0.5 --eta_L 0.5 $OPT --seed $s --outf $o
  else o=$TR/${P}_gap_s$s; done_ $o && { echo "skip (done) $o"; continue; }
    python routeA/train_routeA.py --dataset_dir $D --data_name $DN --encoder_model $EM --shot $K \
        --gamma 0 $OPT --seed $s --workers 8 --outf $o
  fi
done
