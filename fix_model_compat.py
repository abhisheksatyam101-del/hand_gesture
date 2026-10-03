"""
fix_model_compat.py  –  Strip unsupported layer config keys from .keras files.

The model was saved with a newer Keras that added config fields the installed
Keras 3.12.3 doesn't understand.  We strip all known problematic keys.

Strategy: for each layer class, remove any key that is NOT in the known set of
accepted init params for this Keras version.  We use a conservative whitelist
approach only for layers that are actually failing.
"""
import zipfile
import json
import shutil
import os

MODELS = [
    os.path.join("models", "hand_gesture_mobilenetv2.keras"),
    os.path.join("models", "hand_gesture_finetuned.keras"),
]

# class_name -> keys to REMOVE from its "config" dict
STRIP_RULES = {
    "GlorotUniform":       {"input_axes", "output_axes"},
    "BatchNormalization":  {"renorm", "renorm_clipping", "renorm_momentum"},
    "Dense":               {"quantization_config"},
    "Conv2D":              {"quantization_config"},
    "DepthwiseConv2D":     {"quantization_config"},
    "Conv2DTranspose":     {"quantization_config"},
    # Add more as needed – check the error messages for class_name + bad key
}


def strip_bad_keys(obj):
    """Recursively walk the JSON config and strip unsupported per-class keys."""
    if isinstance(obj, dict):
        class_name = obj.get("class_name")
        if class_name and class_name in STRIP_RULES and "config" in obj and isinstance(obj["config"], dict):
            for k in STRIP_RULES[class_name]:
                removed = obj["config"].pop(k, _SENTINEL)
                if removed is not _SENTINEL:
                    print(f"    Stripped {class_name}.{k}")
        return {k: strip_bad_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_bad_keys(i) for i in obj]
    return obj


_SENTINEL = object()


def patch_keras_file(src_path):
    dst_path = src_path + ".fixed"
    with zipfile.ZipFile(src_path, "r") as zin, \
         zipfile.ZipFile(dst_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data.decode("utf-8"))
                cfg = strip_bad_keys(cfg)
                data = json.dumps(cfg, indent=2).encode("utf-8")
            zout.writestr(item, data)

    backup = src_path + ".bak"
    if os.path.exists(backup):
        os.remove(backup)
    shutil.move(src_path, backup)
    shutil.move(dst_path, src_path)
    print(f"  OK  {src_path}  (backup -> {backup})")


if __name__ == "__main__":
    for model_path in MODELS:
        if not os.path.exists(model_path):
            print(f"Skipping (not found): {model_path}")
            continue
        print(f"\nPatching: {model_path}")
        patch_keras_file(model_path)
    print("\nDone.")
