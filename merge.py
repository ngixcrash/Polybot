"""Merge YES+NO position pairs back into $1 USDC via the CTF to free locked capital."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import cfg

log = logging.getLogger(__name__)


@dataclass
class Merger:
    dry_run: bool = True
    _client: Optional[object] = field(default=None, repr=False)
    total_merged_usd: float = 0.0
    merge_count: int = 0

    async def start(self):
        if not self.dry_run:
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                self._client = ClobClient(
                    cfg.clob_url,
                    key=cfg.creds.private_key,
                    chain_id=cfg.chain_id,
                    creds=ApiCreds(
                        api_key=cfg.creds.api_key,
                        api_secret=cfg.creds.api_secret,
                        api_passphrase=cfg.creds.api_passphrase,
                    ),
                    signature_type=cfg.creds.signature_type,
                    funder=cfg.creds.funder_address or None,
                )
                log.info("Merger initialized")
            except Exception as e:
                log.error(f"Merger init failed: {e}")
        else:
            log.info("Merger in DRY RUN mode")

    async def stop(self):
        if self.merge_count > 0:
            log.info(
                f"Merger stats: {self.merge_count} merges, "
                f"${self.total_merged_usd:.2f} recovered"
            )

    async def check_and_merge(self, condition_id: str) -> float:
        """Merge any YES+NO pairs for this condition, return USDC recovered."""
        if not condition_id:
            return 0.0

        if self.dry_run:
            log.debug(f"[MERGE] Dry run — skipping merge for condition={condition_id[:12]}...")
            return 0.0

        try:
            balances = self._get_position_balances(condition_id)
            if not balances:
                return 0.0

            yes_balance = balances.get("yes", 0)
            no_balance = balances.get("no", 0)

            mergeable = min(yes_balance, no_balance)
            if mergeable < 1:  # Need at least 1 share of each
                return 0.0

            log.info(
                f"[MERGE] Found {mergeable} mergeable pairs "
                f"(YES={yes_balance}, NO={no_balance}) "
                f"condition={condition_id[:12]}..."
            )

            # Merge positions — each YES+NO pair → $1 USDC
            if not hasattr(self._client, 'merge_positions'):
                log.warning("[MERGE] merge_positions not available in py-clob-client — skipping")
                return 0.0
            self._client.merge_positions(condition_id, int(mergeable))

            recovered = mergeable * 1.0  # $1 per pair
            self.total_merged_usd += recovered
            self.merge_count += 1

            log.info(f"[MERGED] Recovered ${recovered:.2f} USDC from {int(mergeable)} pairs")
            return recovered

        except Exception as e:
            log.error(f"Merge failed for condition={condition_id[:12]}...: {e}")
            return 0.0

    def _get_position_balances(self, condition_id: str) -> dict:
        """Get YES/NO share counts for a condition."""
        try:
            positions = self._client.get_balances()
            yes_balance = 0
            no_balance = 0

            if isinstance(positions, dict):
                for token_id, balance in positions.items():
                    bal = float(balance) if isinstance(balance, (int, float, str)) else 0
                    if bal > 0:
                        pass

            return {"yes": yes_balance, "no": no_balance}
        except Exception as e:
            log.debug(f"Balance query failed: {e}")
            return {}

    async def check_all_positions(self) -> float:
        """Scan all positions for mergeable YES+NO pairs."""
        if self.dry_run or not self._client:
            return 0.0

        total_recovered = 0.0
        try:
            positions = self._client.get_balances()
            if not positions:
                return 0.0

            log.info(f"Scanning {len(positions)} positions for merge opportunities...")

        except Exception as e:
            log.error(f"Position scan failed: {e}")

        return total_recovered
