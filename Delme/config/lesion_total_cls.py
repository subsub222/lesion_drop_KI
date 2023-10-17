model_tiscls = dict(
    model='resnet24',
    num_tis_classes=4,
    # resume='/home/hrlee/PycharmProjects/lesion/runs/train/total_cls_resnet24_focal_meta_gain3/weights/best_ckpt_0.5323383084577115.pth'
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
    scheduler='cosine',  # cyclelr | steplr | cosine
    accumulate_criterion=6,
    T_up=5,
    T_down=10,
    lr_max=1e-3,
    lr=1e-3,
    lrf=1e-2,
    gamma=0.9,
    weight_decay=5e-4,
    momentum=0.937,
)
