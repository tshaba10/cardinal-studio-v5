"""
Credit card domain rules (package ``credit_card``).

Money is exact: every ``money`` input/output is an int64 count of cents and all
percentage maths is done in integer parts-per-million with explicit
half-up rounding, so a balance carried for 36+ cycles reconciles to the cent.

Conventions implemented here (also declared in the contract's ``conventions``
block so that a reader of the contract knows them):

* Interest uses the daily-balance method: sum of end-of-day balances x APR /
  day-count basis (365), rounded half-up to the cent once, when assessed.
* A transaction posted on day *d* of a *D*-day cycle contributes to the
  end-of-day balance for days d..D, i.e. ``amount x (D - d + 1)`` cent-days.
* A payment in cycle N pays the statement of cycle N-1.
* Interest is waived (grace period) when the previous statement was paid in
  full (a zero previous balance counts as paid in full).
"""
from __future__ import annotations

import numpy as np

from ..identifiers import luhn_check_digit
from ..types import div_round
from . import registrar

rule, sampler, group = registrar("credit_card")

_PPM = 1_000_000


def _ppm(x) -> np.ndarray:
    """Float fraction -> integer parts per million (exact for <=6 decimals)."""
    return np.rint(np.asarray(x, dtype=np.float64) * _PPM).astype(np.int64)


# --------------------------------------------------------------------------- #
# Payments
# --------------------------------------------------------------------------- #

@rule("payment.minimum_due", "1.0", inputs={"balance": "money"}, output="money",
      params={"floor": ("money", 25.0), "pct": ("float", 0.02)})
def minimum_due_v1(balance, floor, pct):
    """max(floor, pct x balance), capped at the balance; 0 when balance <= 0."""
    pct_amount = div_round(balance * _ppm(pct), _PPM)
    due = np.minimum(balance, np.maximum(floor, pct_amount))
    return np.where(balance <= 0, 0, due)


@rule("payment.minimum_due", "2.0",
      inputs={"balance": "money", "interest": "money", "fees": "money"}, output="money",
      params={"floor": ("money", 25.0), "pct": ("float", 0.01)})
def minimum_due_v2(balance, interest, fees, floor, pct):
    """max(floor, pct x balance + interest + fees), capped at the balance; 0 when balance <= 0."""
    pct_amount = div_round(balance * _ppm(pct), _PPM) + interest + fees
    due = np.minimum(balance, np.maximum(floor, pct_amount))
    return np.where(balance <= 0, 0, due)


@rule("payment.paid_in_full", "1.0",
      inputs={"payment": "money", "previous_balance": "money"}, output="bool")
def paid_in_full(payment, previous_balance):
    """True when the previous statement balance was fully paid (or was <= 0)."""
    return (previous_balance <= 0) | (payment >= previous_balance)


@sampler("payment.by_behaviour", "1.0",
         inputs={"previous_balance": "money", "previous_minimum_due": "money",
                 "missed": "bool"},
         output="money",
         params={"p_full": "float", "p_minimum": "float",
                 "partial_low": ("float", 0.05), "partial_high": ("float", 0.9)},
         uniforms=2)
def payment_by_behaviour(u, n, previous_balance, previous_minimum_due, missed,
                         p_full, p_minimum, partial_low, partial_high):
    """Payment against last statement: full / minimum / partial; 0 if missed.

    The amount is bounded *inside the draw* to [minimum due, statement balance]
    so nothing downstream ever has to clip it.
    """
    u_choice, u_frac = u(0), u(1)
    frac = partial_low + (partial_high - partial_low) * u_frac
    partial = div_round(previous_balance * _ppm(frac), _PPM)
    amount = np.where(u_choice < p_full, previous_balance,
                      np.where(u_choice < p_full + p_minimum, previous_minimum_due, partial))
    amount = np.clip(amount, previous_minimum_due, np.maximum(previous_balance, 0))
    return np.where(missed | (previous_balance <= 0), 0, amount)


@sampler("payment.posting_day", "1.0",
         inputs={"days_in_cycle": "int", "payment": "money"}, output="int",
         params={"due_offset_days": ("int", 25), "max_days_early": ("int", 10)}, uniforms=1)
def payment_posting_day(u, n, days_in_cycle, payment, due_offset_days, max_days_early):
    """Day of the cycle the payment posts (on or before the due date); NULL if no payment."""
    early = np.floor(u(0) * (max_days_early + 1)).astype(np.int64)
    day = np.clip(due_offset_days - early, 1, days_in_cycle)
    return day, payment <= 0


# --------------------------------------------------------------------------- #
# Interest, fees and the ledger
# --------------------------------------------------------------------------- #

@rule("interest.daily_balance", "1.0",
      inputs={"opening_balance": "money", "days_in_cycle": "int",
              "activity_balance_days": "money", "payment": "money",
              "payment_day": "int", "grace": "bool", "apr": "fixed6"},
      output="money",
      params={"day_count_basis": ("int", 365)},
      nullable_inputs=("payment_day",))
def interest_daily_balance(opening_balance, days_in_cycle, activity_balance_days, payment,
                           payment_day, grace, apr, day_count_basis):
    """Sum of end-of-day balances x APR / basis, half-up to the cent; 0 in grace.

    ``activity_balance_days`` is the pre-aggregated sum over the cycle's posted
    transactions of ``signed_amount x (days_in_cycle - post_day + 1)``.
    """
    pay_days = np.where(payment > 0, days_in_cycle - payment_day + 1, 0)
    balance_days = opening_balance * days_in_cycle + activity_balance_days - payment * pay_days
    balance_days = np.maximum(balance_days, 0)           # no interest on credit balances
    interest = div_round(balance_days * apr, day_count_basis * _PPM)
    return np.where(grace, 0, interest)


@rule("ledger.balance_days_contribution", "1.0",
      inputs={"amount": "money", "post_day": "int", "days_in_cycle": "int"}, output="money")
def balance_days_contribution(amount, post_day, days_in_cycle):
    """amount x number of end-of-day balances it is part of (cent-days)."""
    return amount * (days_in_cycle - post_day + 1)


@rule("ledger.close_statement", "1.0",
      inputs={"opening_balance": "money", "purchases": "money", "refunds": "money",
              "fees": "money", "interest": "money", "payment": "money"},
      output="money")
def close_statement(opening_balance, purchases, refunds, fees, interest, payment):
    """closing = opening + purchases - refunds + fees + interest - payment (exact)."""
    return opening_balance + purchases - refunds + fees + interest - payment


@rule("txn.posted_amount", "1.0", inputs={"amount": "money", "is_declined": "bool"},
      output="money", params={"sign": ("int", 1)})
def posted_amount(amount, is_declined, sign):
    """Signed amount that hits the balance (0 if declined; refunds have sign -1)."""
    return np.where(is_declined, 0, amount * sign)


@rule("fees.late_fee", "1.0",
      inputs={"payment": "money", "previous_minimum_due": "money"}, output="money",
      params={"amount": ("money", 30.0)})
def late_fee(payment, previous_minimum_due, amount):
    """Flat late fee when the minimum was not met, never more than the minimum due."""
    late = (previous_minimum_due > 0) & (payment < previous_minimum_due)
    return np.where(late, np.minimum(amount, previous_minimum_due), 0)


@rule("ratio.utilization", "1.0", inputs={"balance": "money", "credit_limit": "money"},
      output="float")
def utilization(balance, credit_limit):
    """max(balance, 0) / credit_limit; NULL when the limit is 0."""
    zero = credit_limit <= 0
    out = np.maximum(balance, 0) / np.where(zero, 1, credit_limit)
    return np.where(zero, 0.0, out), zero


@rule("credit.available", "1.0", inputs={"credit_limit": "money", "balance": "money"},
      output="money")
def available_credit(credit_limit, balance):
    """credit_limit - balance (negative when over limit)."""
    return credit_limit - balance


# --------------------------------------------------------------------------- #
# Delinquency
# --------------------------------------------------------------------------- #

@rule("delinquency.days_past_due", "1.0",
      inputs={"previous_dpd": "int", "payment": "money", "previous_minimum_due": "money"},
      output="int", params={"step_days": ("int", 30)})
def days_past_due(previous_dpd, payment, previous_minimum_due, step_days):
    """+step_days when the minimum due was not paid, otherwise back to 0."""
    missed = (previous_minimum_due > 0) & (payment < previous_minimum_due)
    return np.where(missed, previous_dpd + step_days, 0)


@rule("delinquency.charged_off", "1.0", inputs={"dpd": "int"}, output="bool",
      params={"threshold": ("int", 180)})
def charged_off(dpd, threshold):
    """Terminal state: days past due reached the charge-off threshold."""
    return dpd >= threshold


@rule("ledger.charge_off_amount", "1.0",
      inputs={"balance": "money", "charged_off": "bool"}, output="money")
def charge_off_amount(balance, charged_off):
    """Balance written off at charge-off (0 otherwise)."""
    return np.where(charged_off, np.maximum(balance, 0), 0)


# --------------------------------------------------------------------------- #
# Transaction-level constraint (group rule)
# --------------------------------------------------------------------------- #

@group("spend.decline_over_limit", "1.0",
       inputs={"amount": "money", "is_debit": "bool", "pre_declined": "bool",
               "post_day": "int", "available_credit": "money"},
       output="bool")
def decline_over_limit(ctx, amount, is_debit, pre_declined, post_day, available_credit):
    """Declined = declined for another reason (risk block) OR the debit would push
    cumulative approved spend in the cycle above the credit available at the start
    of the cycle.

    Attempts are processed in posting order (day, then sequence). Because
    amounts are positive, approved debits form a prefix of the ordered attempts,
    which guarantees ``approved total <= available credit`` - the constraint is
    enforced at the event level, never by clipping the monthly total.
    """
    grp, seq, n_groups = ctx["group"], ctx["seq"], ctx["n_groups"]
    eligible = is_debit & ~pre_declined
    order = np.lexsort((seq, post_day, grp))
    a = np.where(eligible, amount, 0)[order]
    g = grp[order]
    cs = np.cumsum(a)
    starts = np.ones(g.size, dtype=bool)
    starts[1:] = g[1:] != g[:-1]
    base = np.zeros(n_groups, dtype=np.int64)
    base[g[starts]] = (cs - a)[starts]
    within = cs - base[g]
    over_sorted = eligible[order] & (within > available_credit[order])
    over = np.zeros(g.size, dtype=bool)
    over[order] = over_sorted
    return over | pre_declined


# --------------------------------------------------------------------------- #
# Decorative
# --------------------------------------------------------------------------- #

@sampler("id.masked_pan", "1.0", inputs={}, output="any",
         params={"prefixes": ("list", ["4", "51", "52", "53", "54", "55"]),
                 "prefix_weights": ("list", None), "length": ("int", 16)},
         uniforms=2)
def masked_pan(u, n, prefixes, prefix_weights, length):
    """Luhn-valid card number, masked as 'NNNN **** **** NNNN'."""
    from ..distributions import categorical

    weights = prefix_weights or [1.0] * len(prefixes)
    idx = categorical(u(0), weights)
    payload_len = length - 1
    digits = np.zeros((n, payload_len), dtype=np.uint8)
    # fill with keyed digits: 53 bits of randomness give ~15 decimal digits
    rnd = np.floor(u(1) * 1e15).astype(np.int64)
    for p in range(payload_len - 1, -1, -1):
        digits[:, p] = rnd % 10
        rnd //= 10
    for k, pre in enumerate(prefixes):           # overwrite with the brand prefix
        rows = idx == k
        for j, ch in enumerate(pre):
            digits[rows, j] = int(ch)
    check = luhn_check_digit(digits)
    full = np.concatenate([digits, check[:, None].astype(np.uint8)], axis=1) + ord("0")
    first4 = full[:, :4].copy().view("S4").reshape(-1)
    last4 = full[:, -4:].copy().view("S4").reshape(-1)
    return np.char.add(np.char.add(first4, b" **** **** "), last4)
