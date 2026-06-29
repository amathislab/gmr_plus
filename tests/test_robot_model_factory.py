import importlib.util
import sys
import types
from pathlib import Path


def load_params_module(monkeypatch):
    if "myo_sim" not in sys.modules:
        myo_sim = types.ModuleType("myo_sim")
        compose = types.ModuleType("myo_sim.build.compose")
        build_package = types.ModuleType("myo_sim.build")

        myo_sim.get_xml_path = lambda name: Path(f"/fake/{name}.xml")
        compose.GENERATE_XML_TARGETS = {}
        compose.build_model = lambda name: {"model": name}
        compose.build_generated_model_spec = lambda name: None
        compose.write_spec_xml = lambda spec, output_path: None
        monkeypatch.setitem(sys.modules, "myo_sim", myo_sim)
        monkeypatch.setitem(sys.modules, "myo_sim.build", build_package)
        monkeypatch.setitem(sys.modules, "myo_sim.build.compose", compose)

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
    compose = types.ModuleType("myo_sim.build.compose")
    build_package = types.ModuleType("myo_sim.build")

    def build_model(name):
        built_models.append(name)
        return {"model": name}

    myo_sim.get_xml_path = lambda name: (_ for _ in ()).throw(ValueError(name))
    compose.build_model = build_model
    monkeypatch.setitem(sys.modules, "myo_sim", myo_sim)
    monkeypatch.setitem(sys.modules, "myo_sim.build", build_package)
    monkeypatch.setitem(sys.modules, "myo_sim.build.compose", compose)

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
