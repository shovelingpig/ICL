#!/bin/bash
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=8
method=attention-retriever
num_ice=100
port=10005

#model_name=gpt2-large
#n_tokens=700
#scr_batch_size=128
#inf_batch_size=48

model_name=EleutherAI/gpt-neo-2.7B
n_tokens=1600
scr_batch_size=8
inf_batch_size=8

for task_name in nl2bash
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
      dataset_reader.dataset_split=train \
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
        dataset_reader.dataset_split=train \
        dataset_reader.n_tokens=${n_tokens} \
        index_reader.dataset_path=${index_data} \
        save_score=true
  done

  run_name=attention-retriever
  run_dir=${run_dir}/${run_name}
  accelerate launch --num_processes ${gpu} --multi_gpu --main_process_port ${port}  attention_retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${epr_model} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      training_args.num_train_epochs=1000 \
      training_args.per_device_train_batch_size=256 \
      training_args.per_device_eval_batch_size=256 \
      training_args.learning_rate=0.001 \
      training_args.warmup_steps=125 \
      num_ice=${num_ice} \
      model_config.class_ld=1.0

  retrieve_file=${run_dir}/train_retrieved.json
  accelerate launch --num_processes ${gpu} --multi_gpu --main_process_port ${port}  attention_retriever.py \
      hydra.run.dir=${run_dir}/attention_retriever \
      output_file=${retrieve_file} \
      num_ice=${num_ice} \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${run_dir} \
      faiss_index=${run_dir}/index

  pred_file=${run_dir}/pred.json
  accelerate launch --num_processes ${gpu} --multi_gpu --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size}
done 