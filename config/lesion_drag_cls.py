model_dragon = dict(
    input_dim=21,
    resume=None,
    # resume='/home/hrlee/PycharmProjects/lesion/runs/train/dragcls_conv3d_focal/weights/best_ckpt_0.6069651741293532.pth',
    rep_dim=700,
    hypo_dim=700,
    sec_dim=200,
    thir_dim=100

)

model_tiscls = dict(
    model='resnet6',
    num_tis_classes=4,
    resume=None,
)

dataset = dict(
    base_path='/storage/yskim/lesion',
)

training_params = dict(
    print_freq=20,
    acc_freq=800,
    eval_freq=800,
)

solver = dict(
    optim='adam',
    scheduler='cyclelr',  # cyclelr | steplr | cosine
    accumulate_criterion=64,
    T_up=50,
    T_down=50,
    lr_max=0.5e-3,
    lr=1e-8,
    lrf=1e-3,
    gamma=0.9,
    weight_decay=5e-3,
    momentum=0.937,
)
