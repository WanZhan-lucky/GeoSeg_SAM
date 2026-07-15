import copy


datasets = {}
#数据集注册工厂（调度中心）
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

#根据 config 自动实例化数据集
def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset

#统一管理所有数据集并通过名字动态创建实例。