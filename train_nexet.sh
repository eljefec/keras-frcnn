[ ! -f resnet50_weights_tf_dim_ordering_tf_kernels.h5 ] && wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5

python train_frcnn.py -o pascal_voc -p ~/data/ --hf --input_weight_path ~/data/nexet/frcnn/1-hf_nosmall/model_frcnn_e347_l0.309_a0.916.hdf5
