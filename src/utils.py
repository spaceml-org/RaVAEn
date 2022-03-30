import importlib
import omegaconf


def load_obj(obj_path):
    """
    Call an object from a string
    """

    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0)

    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object '{obj_name}' cannot be loaded from '{obj_path}'."
        )
    return getattr(module_obj, obj_name)


def deepconvert(omega_conf):
    if isinstance(omega_conf, omegaconf.dictconfig.DictConfig):
        not_omega_conf = {}
        for k, v in omega_conf.items():
            not_omega_conf.update({k: deepconvert(v)})
        return not_omega_conf

    if isinstance(omega_conf, omegaconf.listconfig.ListConfig):
        not_omega_conf = []
        for v in omega_conf:
            not_omega_conf.append(deepconvert(v))
        return not_omega_conf

    return omega_conf
