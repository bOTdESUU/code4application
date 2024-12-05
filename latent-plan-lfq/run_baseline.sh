# export PYTHONPATH=.:$PYTHONPATH
# K=512 # Default value for K
# Z=9   # Default value for Z
datasets=(hopper-medium-expert-v2)

# name=lfq_K512_zdim9_wt_tf
# datasets=(hopper-medium-replay-v2 hopper-medium-expert-v2)


while getopts "K:Z:D:" opt; do
  case $opt in
    K) K=$OPTARG ;;
    Z) Z=$OPTARG ;;
    D) datasets=($OPTARG) ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# name="lfq_K${K}_zdim${Z}_wt_tf"
name="test_lfq_K${K}_zdim${Z}_wt_tf"

for round in {1..5}; do
  for data in ${datasets[@]}; do
    python scripts/train.py --dataset $data --exp_name $name-$round --tag baseline --seed $round
    # python scripts/train_lfq.py --dataset $data --exp_name $name-$round --tag lfq_bl --seed $round
    python scripts/trainprior.py --dataset $data --exp_name $name-$round 
    for i in {1..20};
    do
       python scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 64 --n_expand 4 --horizon 15
    done
  done
done

for data in ${datasets[@]}; do
  for round in {1..5}; do
    python plotting/read_results.py --exp_name $name-$round --dataset $data
  done
  python plotting/read_results.py --exp_name $name --dataset $data
done

