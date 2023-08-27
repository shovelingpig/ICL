#!/bin/bash
# export WANDB_PROJECT=icl  # change if needed
# export WANDB_ENTITY=hyperconnect  # change to your wandb account
# export WANDB_API_KEY=feff92576de3c4da75e6b7fc8244a388bf383ba1  # change to your api-key
# export WANDB_START_METHOD=thread
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=8
method=advanced-scorer
num_ice=100
port=7005

#model_name=gpt2-large
#n_tokens=700
#scr_batch_size=128
#inf_batch_size=48

model_name=EleutherAI/gpt-neo-2.7B
n_tokens=1600
scr_batch_size=8
inf_batch_size=8

for task_name in sst5
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  epr_model=output/epr/${task_name}/${model_name}/bert-fix_ctx-shared-bs64

  retrieve_file=${run_dir}/retrieved.json
  python dense_retriever.py \
      hydra.run.dir=${run_dir}/dense_retriever \
      output_file=${retrieve_file} \
      task_name=${task_name} \
      dataset_reader.dataset_split=validation \
      index_reader.dataset_path=${index_data} \
      faiss_index=${run_dir}/index \
      pretrained_model_path=${epr_model} \
      model_config.norm_embed=true \
      advanced_scoring=true \
      num_ice=${num_ice}

  scored_file=${run_dir}/scored.json
  cp ${retrieve_file} ${scored_file}
  for i in {1..100}; do
    accelerate launch --multi_gpu --num_processes ${gpu} --main_process_port ${port}  advanced_scorer.py \
        hydra.run.dir=${run_dir}/scorer \
        task_name=${task_name} \
        output_file=${scored_file} \
        batch_size=${scr_batch_size} \
        model_name=${model_name} \
        dataset_reader.dataset_path=${scored_file} \
        dataset_reader.dataset_split=validation \
        dataset_reader.n_tokens=${n_tokens} \
        index_reader.dataset_path=${index_data} \
        save_score=true
  done

  pred_file=${run_dir}/pred.json
  accelerate launch --num_processes ${gpu} --multi_gpu --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${scored_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size} \
      advanced_scoring=true
done 