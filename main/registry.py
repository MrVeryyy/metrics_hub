_METRIC_REGISTRY = {}

def register_metric(name, metric_cls):
    _METRIC_REGISTRY[name.lower()] = metric_cls

def get_metric(name, **kwargs):
    name = name.lower()
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' not registered.")
    return _METRIC_REGISTRY[name](**kwargs)

def list_metrics():
    return sorted(_METRIC_REGISTRY.keys())
