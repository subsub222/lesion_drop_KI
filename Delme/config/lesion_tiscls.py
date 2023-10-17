model_tiscls = dict(
    model='resnet18',
    num_tis_classes=4,
    # resume='/home/hrlee/PycharmProjects/lesion/runs/train/cls_resnet6v2_focal_rgb_sum_gain/weights/best_ckpt_0.642.pth',
    resume=None,
)

dataset = dict(
    base_path='/storage/yskim/lesion',
)

training_params = dict(
    print_freq=5,
    acc_freq=1,
    eval_freq=1,
)

solver = dict(
    optim='adam',
    scheduler='steplr',  # cyclelr | steplr | cosine
    accumulate_criterion=6,
    T_up=10,
    T_down=25,
    lr_max=1e-3,
    lr=1e-3,
    lrf=1e-2,
    gamma=0.95,
    weight_decay=5e-4,
    momentum=0.937,
)
