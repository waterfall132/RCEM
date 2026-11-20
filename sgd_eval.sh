#!/usr/bin/env bash
cd "${WORK:-$(dirname "$0")}"; R=results_routeB
DATA=${DATA:-/root/data}
fin () { python - "$1" <<'PY'
import json,sys
try:
    d=sys.argv[1]; h=len(json.load(open(d+'/summary.json'))['history']); e=json.load(open(d+'/config.json'))['epochs']
    sys.exit(0 if h>=e else 1)
except Exception: sys.exit(1)
PY
}
for DS in ${DATASETS:-CUB Cars Dogs mini}; do for ENC in R12 C4; do for K in 1 5; do
  case $DS in CUB) D=$DATA/CUB/For_FewShot ;; Dogs) D=$DATA/Dogs/For_FewShot ;;
              Cars) D=$DATA/Cars ;; mini) D=$DATA/miniImageNet ;; esac
  [ -d "$D" ] || continue
  TR=runs/routeB; [ "$DS-$ENC" = CUB-R12 ] || TR=runs/routeB_${DS}_${ENC}
  O=$R/${DS}_${ENC}_dn4_${K}shot; H=B_fixed; [ $K -ge 2 ] && H=B_select
  for r in $TR/${DS}_${ENC}_${K}s_dn4${TAG}_*_s*; do
    [ -d "$r" ] && fin "$r" || continue
    n=$(basename $r); ls -d $O/*__${n: -24}_* >/dev/null 2>&1 && continue
    mkdir -p $O; echo ">>> evaluating $r"
    case $r in *_B_eta05_*) F="--fixed --no_controls" ;; *) F="--refs_only" ;; esac
    python routeB/eval_b.py --dataset_dir $D --data_name ev --shots $K $F --ckpt $r/model_best.pth.tar --outdir $O 2>&1 | tail -n 1
  done
  ls $O/*_dn4${TAG}_B_eta05_s*/test_${K}shot/$H.npz >/dev/null 2>&1 && ls $O/*_dn4${TAG}_gap_s*/test_${K}shot/gap.npz >/dev/null 2>&1 || continue
  echo; echo "################ $DS $ENC ${K}-shot${TAG}"
  python routeB/compare_pair.py --K $K --a "RCEM=$O/*_dn4${TAG}_B_eta05_s*/test_${K}shot/$H.npz" --b "GAP+CE=$O/*_dn4${TAG}_gap_s*/test_${K}shot/gap.npz" | grep -v "^$"
  python routeB/compare_pair.py --K $K --a "RCEM-global=$O/*_dn4${TAG}_B_eta05_s*/test_${K}shot/gap.npz" --b "GAP+CE=$O/*_dn4${TAG}_gap_s*/test_${K}shot/gap.npz" | tail -n 3
done; done; done
