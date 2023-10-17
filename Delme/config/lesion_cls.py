model_cls = dict(
    model='resnet10',
    num_classes=2,
    # resume='/home/hrlee/PycharmProjects/lesion/runs/train/cls_resnet6v2_focal_rgb_sum_gain/weights/best_ckpt_0.642.pth',
    resume=None,
)

dataset = dict(
    base_path='/storage/yskim/lesion',
)

training_params = dict(
    print_freq=10,
    acc_freq=400,
    eval_freq=400,
)

solver = dict(
    optim='adam',
    scheduler='cyclelr',  # cyclelr | steplr | cosine
    accumulate_criterion=6,
    T_up=5,
    T_down=10,
    lr_max=1e-3,
    lr=1e-6,
    lrf=1e-2,
    gamma=0.9,
    weight_decay=5e-5,
    momentum=0.937,
)
