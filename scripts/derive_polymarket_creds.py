"""
Derive Polymarket CLOB API credentials from your wallet private key.
Reads POLYMARKET_PK from .env and prints apiKey, secret, passphrase for .env.
Run once, then paste the three values into .env.
"""
import os
import sys


def _load_pk_from_env() -> str | None:
    """Read POLYMARKET_PK from .env in project root (no dotenv dependency)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if not os.path.isfile(env_path):
        return None
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                if key.strip() == "POLYMARKET_PK":
                    return value.strip().strip('"').strip("'")
    return None


def main() -> None:
    pk = _load_pk_from_env() or os.getenv("POLYMARKET_PK")
    if not pk or pk.startswith("your_"):
        print("POLYMARKET_PK not set or still placeholder in .env. Add your private key first.")
        sys.exit(1)

    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        print("py_clob_client not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # CLOB client for mainnet (chain_id 137 = Polygon)
    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=pk.strip(),
        signature_type=1,
    )
    creds = client.create_or_derive_api_creds()

    # ApiCreds is an object with attributes, not a dict
    api_key = getattr(creds, "api_key", None) or getattr(creds, "apiKey", None)
    secret = getattr(creds, "api_secret", None) or getattr(creds, "secret", None)
    passphrase = getattr(creds, "api_passphrase", None) or getattr(creds, "passphrase", None)

    print("\nAdd these three lines to your .env file (replace the placeholders):\n")
    print(f"POLYMARKET_API_KEY={api_key}")
    print(f"POLYMARKET_API_SECRET={secret}")
    print(f"POLYMARKET_PASSPHRASE={passphrase}")
    print("\nThen save .env and run the bot.\n")


if __name__ == "__main__":
    main()
