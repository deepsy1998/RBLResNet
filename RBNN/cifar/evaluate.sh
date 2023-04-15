python -u main.py \
--gpus 0 \
-e ../../models/ANN_results/ \
--model_a MC_net_10blocksv2 \
--model_b MC_net_10blocksv2 \
--model_c MC_net_10blocksv2 \
--data_path "../../Dataset" \
--dataset rml \
-bt 128
