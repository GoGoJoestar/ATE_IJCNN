R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../..
# pwd
source ${MYDIR}/config

for seed in ${seeds[@]};do
    python ./src/run_ate_bert_ss.py \
        --hidden_size ${hidden_size} \
        --gcn_layers ${gcn_layers} \
        --epoch ${epoch} \
        --lr ${lr} \
        --seed ${seed} \
        --dropout ${dropout} \
        --warm_up_rate ${warm_up_rate} \
        --end_lr ${end_lr} \
        --bert_lr ${bert_lr} \
        --bert_end_lr ${bert_end_lr} \
        --lambda1 ${lambda1} \
        --lambda2 ${lambda2} \
        --temperature ${temperature} \
        --log_path ${log_path} \
        --train_set ${train_set} \
        --dev_set ${dev_set} \
        --test_set ${test_set} \
        --train_dep_label_file ${train_dep_label_file} \
        --test_dep_label_file ${test_dep_label_file} \
        --train_pos_label_file ${train_pos_label_file} \
        --test_pos_label_file ${test_pos_label_file} \
        --train_dep_adj_file ${train_dep_adj_file} \
        --test_dep_adj_file ${test_dep_adj_file} \
        --max_seq_len ${max_seq_len} \
        --batch_size ${batch_size} \
        --do_lower_case ${do_lower_case} \
        --label_schema ${label_schema} \
        --output_path ${output_path} \
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


if [[ $? -ne 0 ]]; then
    echo "run failed"
    exit 1
fi
exit 0