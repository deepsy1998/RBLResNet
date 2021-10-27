python -u main.py \
--gpus 1 \
-e ../../../Results/RBNN_results/ \
--model MC_net_10blocks_binary \
--data_path "../../Dataset" \
--dataset rml \
-bt 128 \