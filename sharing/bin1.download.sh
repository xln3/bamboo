#!/bin/bash
# bin1 asset downloader — generated
set -uo pipefail
DEST="/home/fit1-217/bbdata/bin1"
mkdir -p "$DEST/datasets" "$DEST/models" "$DEST/other" "$DEST/_logs"
: ${HF_ENDPOINT:=https://hf-mirror.com}
: ${HF_HOME:=$DEST/hf-cache}
export HF_ENDPOINT HF_HOME
mkdir -p "$HF_HOME"

log()  { echo "[$(date +%H:%M:%S)] $*"; }

PARALLEL=${PARALLEL:-8}

# ─── Direct HTTP downloads (wget -c) ─────────────────────────
_dl_direct() {
  local kind="$1" name="$2" url="$3"
  local odir="$DEST/$kind/$name"
  mkdir -p "$odir"
  local fname="${url##*/}"; fname="${fname%%\?*}"
  [ -z "$fname" ] && fname="data"
  local out="$odir/$fname"
  if [ -s "$out" ] && [ ! -s "$out.aria2" ]; then
    log "skip-direct $name (have $out)"; return 0
  fi
  log "wget $name → $out"
  wget -c -t 5 --timeout=120 --tries=5 --no-check-certificate -q "$url" -O "$out" \
    >> "$DEST/_logs/$name.wget.log" 2>&1 \
    && log "ok-direct $name" \
    || log "FAIL-direct $name (see _logs/$name.wget.log)"
}

# ─── HF datasets / models (via mirror) ───────────────────────
_dl_hf() {
  local kind_label="$1" name="$2" repo_kind="$3" repo="$4"
  local subdir="$DEST/$kind_label/$name"
  mkdir -p "$subdir"
  if [ -f "$subdir/.complete" ]; then
    log "skip-hf $name (have $subdir/.complete)"; return 0
  fi
  log "hf-cli $repo_kind:$repo → $subdir"
  if [ "$repo_kind" = "dataset" ]; then
    hf download --repo-type dataset --resume-download \
      --local-dir "$subdir" "$repo" \
      >> "$DEST/_logs/$name.hf.log" 2>&1
  else
    hf download --resume-download \
      --local-dir "$subdir" "$repo" \
      >> "$DEST/_logs/$name.hf.log" 2>&1
  fi
  if [ $? -eq 0 ]; then
    touch "$subdir/.complete"
    log "ok-hf $name"
  else
    log "FAIL-hf $name (see _logs/$name.hf.log)"
  fi
}

export -f _dl_direct _dl_hf log
export DEST

run_jobs() {
  local jobs_file="$1"
  if command -v parallel >/dev/null; then
    parallel -j "$PARALLEL" --colsep "\t" --joblog "$DEST/_logs/parallel.log" :::: "$jobs_file"
  else
    xargs -a "$jobs_file" -P "$PARALLEL" -L 1 -I {} bash -c "{}"
  fi
}

# === Direct HTTP jobs ===
cat > "$DEST/_jobs_direct.sh" <<'EOF'
_dl_direct dataset chexpert 'https://stanfordmlgroup.github.io/competitions/chexpert/'
_dl_direct dataset stanford-cars 'https://ai.stanford.edu/~jkrause/cars/car_dataset.html'
_dl_direct dataset svhn 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
_dl_direct model resnet50v2 'https://tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2'
_dl_direct model resnet101v2 'https://tensorflow.org/api_docs/python/tf/keras/applications/ResNet101V2'
_dl_direct dataset celeba 'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'
_dl_direct dataset resisc45 'http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html'
_dl_direct dataset ppi 'https://data.dgl.ai/dataset/ppi.zip'
_dl_direct dataset imagenetbg 'https://www.image-net.org/'
_dl_direct dataset living17 'https://www.image-net.org/'
_dl_direct dataset stanford-dogs 'https://tensorflow.org/datasets/catalog/stanford_dogs'
_dl_direct dataset dtd 'https://www.robots.ox.ac.uk/~vgg/data/dtd/'
_dl_direct dataset proteins 'https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip'
_dl_direct model vgg16 'https://tensorflow.org/api_docs/python/tf/keras/applications/VGG16'
_dl_direct dataset waterbirds 'https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz'
_dl_direct dataset tiny-imagenet 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
_dl_direct dataset stl-10 'http://ai.stanford.edu/~acoates/stl10/'
_dl_direct dataset multinli 'https://cims.nyu.edu/~sbowman/multinli/'
_dl_direct dataset ogbn-arxiv 'https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'
_dl_direct dataset zinc 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_250K.tar.gz'
_dl_direct dataset qm9 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
_dl_direct dataset cifar-10 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_dl_direct dataset cifar-100 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
_dl_direct other gurobi-optimizer 'https://portal.gurobi.com/iam/licenses/request'
_dl_direct other mosek-optimizer 'https://www.mosek.com/downloads/'
_dl_direct model inceptionv3 'https://tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3'
_dl_direct model resnet-50 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
_dl_direct dataset zinc250k 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
_dl_direct dataset fashion-mnist 'https://tensorflow.org/datasets/catalog/fashion_mnist'
_dl_direct model resnet-18 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
_dl_direct dataset pubmed-graph 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz'
_dl_direct model densenet121 'https://tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121'
_dl_direct dataset 20newsgroup 'https://ndownloader.figshare.com/files/5975967'
_dl_direct dataset mnist 'http://yann.lecun.com/exdb/mnist/'
_dl_direct dataset default-of-credit-card-clients 'https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients'
_dl_direct dataset cora 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
_dl_direct dataset citeseer 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz'
_dl_direct dataset adult 'https://archive.ics.uci.edu/ml/datasets/adult'
_dl_direct dataset adult-income 'https://archive.ics.uci.edu/ml/datasets/adult'
_dl_direct dataset imagewoof 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz'
_dl_direct dataset magic-gamma-telescope 'https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope'
_dl_direct dataset swe-bench-lite 'https://www.swebench.com/lite.html'
_dl_direct dataset online-shoppers-purchasing-intention 'https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset'
_dl_direct dataset california-housing 'https://lib.stat.cmu.edu/datasets/houses.zip'
_dl_direct dataset ego-facebook 'https://snap.stanford.edu/data/ego-Facebook.html'
_dl_direct dataset enzymes 'https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip'
_dl_direct dataset uci-letter-recognition 'https://archive.ics.uci.edu/dataset/59/letter+recognition'
_dl_direct dataset mutag 'https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip'
_dl_direct dataset statlog 'https://doi.org/10.24432/C5NC77'
_dl_direct dataset arxiv-gr-qc-collaboration-network 'https://snap.stanford.edu/data/ca-GrQc.html'
_dl_direct dataset cifar-10-c 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
_dl_direct dataset emnist 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
_dl_direct dataset fastmri 'https://fastmri.med.nyu.edu/'
_dl_direct dataset movielens-1m 'https://grouplens.org/datasets/movielens/1m/'
EOF

# === HF jobs ===
cat > "$DEST/_jobs_hf.sh" <<'EOF'
_dl_hf model gpt-2-medium model openai-community/gpt2-medium
_dl_hf model clip-vit-base-patch32 model openai/clip-vit-base-patch32
_dl_hf dataset mmrefine dataset naver-ai/mmrefine
_dl_hf dataset codeif dataset linrany/CodeIF
_dl_hf dataset u-safebench dataset Yeonjun/U-SafeBench
_dl_hf dataset kernelbench dataset ScalingIntelligence/KernelBench
_dl_hf dataset baxbench dataset LogicStar/BaxBench
_dl_hf dataset multi-step-arithmetic dataset maveriq/bigbenchhard
EOF

# Run direct jobs in parallel
log "== direct downloads ($(wc -l < $DEST/_jobs_direct.sh) jobs, P=$PARALLEL) =="
cat "$DEST/_jobs_direct.sh" | xargs -d "\n" -P "$PARALLEL" -L 1 -I {} bash -c "{}"
log "== hf downloads ($(wc -l < $DEST/_jobs_hf.sh) jobs, P=4) =="
cat "$DEST/_jobs_hf.sh" | xargs -d "\n" -P 4 -L 1 -I {} bash -c "{}"
log "== ALL DONE =="
du -sh "$DEST"
