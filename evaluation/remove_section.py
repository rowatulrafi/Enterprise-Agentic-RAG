import json
from pathlib import Path


CLEAN_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "clean_json"
)


for path in CLEAN_DIR.glob("*.json"):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        pages = json.load(file)

    changed = False

    for page in pages:
        if "section" in page:
            del page["section"]
            changed = True

    if changed:

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                pages,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"✅ Removed sections: {path.name}"
        )

    else:
        print(
            f"➖ No sections found: {path.name}"
        )