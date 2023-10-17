

dataset = dict(
    base_path='/storage/yskim/lesion',
    json_path='/storage/yskim/lesion/splitted_data.json'
)

training_params = dict(
    epoch=100,
    gpu_id='0',
)

solver = dict(
    num_up_down=5,
    optim='adam',
    scheduler='sgdr',
    weight_decay=5e-4,
    momentum=0.937
)
solver['lr'] = 1e-4 if solver['lr_scheduler'] != 'sgdr' else 1e-9
