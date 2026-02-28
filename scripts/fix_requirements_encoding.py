"""
Fix requirements.txt saved as UTF-16 (null bytes between chars) so pip can read it.
Run from project root: python scripts/fix_requirements_encoding.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")


def main() -> None:
    with open(REQUIREMENTS, "rb") as f:
        raw = f.read()

    # UTF-16 LE BOM, or null every other byte (no BOM)
    if raw.startswith(b"\xff\xfe") or (len(raw) > 1 and raw[1:2] == b"\x00"):
        text = raw.decode("utf-16-le")
    elif raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16-be")
    else:
        text = raw.decode("utf-8")

    with open(REQUIREMENTS, "w", encoding="utf-8") as f:
        f.write(text)

    print("requirements.txt converted to UTF-8. Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
