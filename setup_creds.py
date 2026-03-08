"""Derive Polymarket CLOB API credentials from your Ethereum private key."""

import sys

try:
    from py_clob_client.client import ClobClient
except ImportError:
    print("ERROR: py-clob-client not installed. Run: pip install py-clob-client==0.34.5")
    sys.exit(1)

from config import cfg


def main():
    if not cfg.creds.private_key or cfg.creds.private_key == "0x...":
        print("ERROR: Set POLY_PRIVATE_KEY in .env before running this script.")
        sys.exit(1)

    print("Deriving Polymarket API credentials...")
    print(f"Chain ID: {cfg.chain_id}")

    client = ClobClient(
        cfg.clob_url,
        key=cfg.creds.private_key,
        chain_id=cfg.chain_id,
    )

    creds = client.derive_api_key()

    print("\n--- Add these to your .env file ---")
    print(f"POLY_API_KEY={creds.api_key}")
    print(f"POLY_API_SECRET={creds.api_secret}")
    print(f"POLY_API_PASSPHRASE={creds.api_passphrase}")
    print("-----------------------------------")

    print(f"\nYour funder address (for POLY_FUNDER_ADDRESS):")
    print(f"POLY_FUNDER_ADDRESS={client.get_address()}")


if __name__ == "__main__":
    main()
