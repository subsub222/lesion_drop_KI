model_dragon = dict(
    input_dim=24,
    resume=None,
    # resume='/home/hrlee/PycharmProjects/lesion/runs/train/dragcls_conv3d_focal/weights/best_ckpt_0.6069651741293532.pth',
    rep_dim=700,
    hypo_dim=700,
    sec_dim=200,
    thir_dim=100
)

dataset = dict(
    base_path='/storage/yskim/lesion',
    json_path='/storage/yskim/lesion/splitted_data.json'
)

training_params = dict(
    print_freq=10,
    acc_freq=10,
    eval_freq=10,
)

solver = dict(
    optim='adam',
    scheduler='cyclelr',
    T_up=500,
    T_down=30,
    lr_max=1e-3,
    lr=1e-7,
    gamma=0.9,
    weight_decay=5e-5,
    momentum=0.937,
)
