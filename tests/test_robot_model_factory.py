import importlib.util
import sys
import types
from pathlib import Path


def load_params_module(monkeypatch):
    if "myo_sim" not in sys.modules:
        myo_sim = types.ModuleType("myo_sim")
        myo_sim.get_xml_path = lambda name: Path(f"/fake/{name}.xml")
        myo_sim.load_model = lambda name: {"model": name}
        monkeypatch.setitem(sys.modules, "myo_sim", myo_sim)

    module_path = Path(__file__).resolve().parents[1] / "general_motion_retargeting" / "params.py"
    module_name = "gmr_params_under_test"
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_myofullbody_model_builds_from_myo_sim_mjspec(monkeypatch):
    built_models = []

    myo_sim = types.ModuleType("myo_sim")

    def load_model(name):
        built_models.append(name)
        return {"model": name}

    myo_sim.get_xml_path = lambda name: (_ for _ in ()).throw(ValueError(name))
    myo_sim.load_model = load_model
    monkeypatch.setitem(sys.modules, "myo_sim", myo_sim)

    params = load_params_module(monkeypatch)

    assert params.ROBOT_MODEL_DICT["myofullbody"] == {"model": "myofullbody"}
    assert params.get_robot_model("myofullbody") == {"model": "myofullbody"}
    assert built_models == ["myofullbody"]


def test_xml_robot_model_loads_from_xml_path(monkeypatch):
    loaded_paths = []

    mujoco = types.ModuleType("mujoco")

    class MjModel:
        @staticmethod
        def from_xml_path(path):
            loaded_paths.append(path)
            return {"xml_path": path}

    mujoco.MjModel = MjModel
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)

    params = load_params_module(monkeypatch)

    model = params.get_robot_model("unitree_g1")

    assert model == {"xml_path": str(params.ROBOT_XML_DICT["unitree_g1"])}
    assert loaded_paths == [str(params.ROBOT_XML_DICT["unitree_g1"])]
