[ ! -f resnet50_weights_tf_dim_ordering_tf_kernels.h5 ] && wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5

python train_frcnn.py -o simple -p ~/data/nexet/train_boxes.simple.csv
