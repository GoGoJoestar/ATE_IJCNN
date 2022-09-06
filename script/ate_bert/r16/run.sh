R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../..
# pwd
source ${MYDIR}/config

for few_count in ${few_counts[@]};do
for seed in ${seeds[@]};do
    python ./src/run_ate_bert.py \
        --epoch ${epoch} \
        --lr ${lr} \
        --seed ${seed} \
        --dropout ${dropout} \
        --warm_up_rate ${warm_up_rate} \
        --end_lr ${end_lr} \
        --bert_lr ${bert_lr} \
        --bert_end_lr ${bert_end_lr} \
        --log_path ${log_path} \
        --train_set ${train_set} \
        --dev_set ${dev_set} \
        --test_set ${test_set} \
        --max_seq_len ${max_seq_len} \
        --batch_size ${batch_size} \
        --do_lower_case ${do_lower_case} \
        --label_schema ${label_schema} \
        --output_path ${output_path} \
        --few_count ${few_count} \
        --data_seed ${data_seed} \
        --do_train ${do_train} \
        --do_val ${do_val} \
        --do_test ${do_test} \
        --do_pred ${do_pred} \
        --ckpt ${ckpt} \
        --ckpt_save_count ${ckpt_save_count} \
        --skep_step ${skep_step} \
        --gpu_id ${gpu_id} \
        --pretrain_model_path ${pretrain_model_path}
done
done



if [[ $? -ne 0 ]]; then
    echo "run failed"
    exit 1
fi
exit 0