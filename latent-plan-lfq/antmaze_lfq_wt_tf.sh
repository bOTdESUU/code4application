export PYTHONPATH=.:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/zhengyao/project/transformer_planning/implicit_q_learning

name=lfq_K2pow16_zdim16_wt_tf
antdatasets=(antmaze-large-diverse-v0)

unset LD_PRELOAD

for round in {1..5}; do
  for data in ${antdatasets[@]}; do
    python scripts/train_lfq_wt_tf.py --dataset $data --exp_name $name-$round --tag lfq_wt_tf --seed $round --K 65536 --trajectory_embd 16
    python scripts/trainprior_wt_tf.py --dataset $data --exp_name $name-$round 
    for i in {1..20};
    do
       python scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 2 --n_expand 4
    done 
  done
done

for data in ${antdatasets[@]}; do
  for round in {1..5}; do
    python plotting/read_results.py --exp_name $name-$round --dataset $data
  done
  python plotting/read_results.py --exp_name $name --dataset $data
done


