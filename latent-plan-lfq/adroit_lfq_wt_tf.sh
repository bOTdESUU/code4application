export PYTHONPATH=.:$PYTHONPATH

name=lfq_K512_zdim9
datasets=(pen-cloned-v0)

for round in {1..5}; do
  for data in ${datasets[@]}; do
    python scripts/train_lfq.py --dataset $data --exp_name $name-$round --tag lfq --seed $round --K 512 --trajectory_embd 9
    python scripts/trainprior.py --dataset $data --exp_name $name-$round
    for i in {1..20};
    do
      python scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --n_expand 4 --beam_width 256 --horizon 24
    done
  done
done

for data in ${datasets[@]}; do
  for round in {1..1}; do
    python plotting/read_results.py --exp_name $name-$round --dataset $data
  done
  python plotting/read_results.py --exp_name $name --dataset $data
done

