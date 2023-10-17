model_seg = dict(
    type='unet',  # [ deeplab | unet | unet_short]
    model='deeplabv3plus_resnet50',  # For deeplab, [deeplabv3plus_resnet50 |deeplabv3plus_mobilenet]
    output_stride=8,  # [8, 16]
    num_classes=4,
    separable_conv=False,
    resume=None,
)

dataset = dict(
    base_path='/storage/yskim/lesion',
)

training_params = dict(
    print_freq=1,
    acc_freq=7,
    eval_freq=7,
)

solver = dict(
    optim='adam',
    scheduler='cosine',  # cyclelr | steplr
    accumulate_criterion=8,
    T_up=20,
    T_down=20000,
    lr_max=0.5e-3,
    lr=0.5e-3,
    lrf=1e-2,
    gamma=1,
    weight_decay=5e-4,
    momentum=0.937,
)
