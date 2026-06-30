import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

"""Simulation of a repetition code under bit-flip noise.

This script encodes a simple message into a repetition code, introduces
independent bit-flip errors with a tunable physical error probability, applies
syndrome-based decoding, and then estimates the logical error rate as a
function of the physical error probability for several repetition-code
lengths. The resulting data are then fitted to the scaling form
p_L = a p^{(d+1)/2}.
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Change these values at the top before running the script.
message = [0]  # Message bits to encode and test.
physical_error_probabilities = np.linspace(0.001, 0.08, 20)  # Values of p to sweep.
code_lengths = [3, 5, 7]  # Repetition-code sizes to test.
num_trials = 100000  # Number of Monte Carlo trials per data point.
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Output locations for saved results and figures.
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
figures_dir = project_root / "figures"
data_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)
data_file = data_dir / "repetition_code_results.csv"
figure_file = figures_dir / "repetition_code_error_rates.png"


def encoding(bit_string, d):
    """Repeat each bit of the input message d times to build a codeword.

    Parameters
    ----------
    bit_string : list[int]
        The message bits to be encoded.
    d : int
        The repetition factor, i.e. how many times each bit is copied.

    Returns
    -------
    list[int]
        The encoded repetition-code word.
    """
    encoded_string = []
    for bit in bit_string:
        encoded_string.extend([bit] * d)
    return encoded_string


def error_sampling(bit_string, p):
    """Apply independent bit-flip noise to each bit with probability p.

    Parameters
    ----------
    bit_string : list[int]
        The bits before noise is introduced.
    p : float
        The probability that each bit flips.

    Returns
    -------
    list[int]
        The noisy bit string after sampling errors.
    """
    after_error = []
    for bit in bit_string:
        if np.random.rand() < p:
            after_error.append(1 - bit)
        else:
            after_error.append(bit)
    return after_error


def syndrome_measurement(block):
    """Compute the syndrome of a block by XORing adjacent bits.

    The syndrome reveals where the parity checks are violated and is used to
    infer the likely error pattern.
    """
    s = []
    for i in range(0, len(block) - 1):
        s.append(block[i] ^ block[i + 1])
    return s


def build_single_error_lookup_table(d):
    """Construct a lookup table for the simplest error patterns.

    This table maps a syndrome to the corresponding single-bit error pattern for
    a repetition code of length d. It is useful for small demonstrations but is
    not the most general decoding strategy.
    """
    table = {}

    no_error = [0] * d
    no_syndrome = tuple(syndrome_measurement(no_error))
    table[no_syndrome] = [0] * d

    for error_position in range(d):
        error = [0] * d
        error[error_position] = 1

        syndrome = tuple(syndrome_measurement(error))
        table[syndrome] = error

    return table


def weight(error):
    """Return the number of ones in an error pattern."""
    return sum(error)


def build_minimum_weight_lookup_table(d):
    """Build a decoder lookup table using minimum-weight matching.

    For each possible error pattern, the syndrome is computed. If the syndrome has
    not yet been seen, the pattern is stored. If the same syndrome appears again,
    the lower-weight pattern is kept, which is the standard minimum-weight
    decoding rule for repetition codes.
    """
    table = {}

    for error in product([0, 1], repeat=d):
        error = list(error)
        syndrome = tuple(syndrome_measurement(error))

        if syndrome not in table:
            table[syndrome] = error
        else:
            if weight(error) < weight(table[syndrome]):
                table[syndrome] = error

    return table


def decoding_list(syndrome, lookup_table):
    """Look up the error pattern associated with a syndrome.

    Parameters
    ----------
    syndrome : list[int] or tuple[int, ...]
        The syndrome measured from a received block.
    lookup_table : dict
        A mapping from syndrome to the corresponding recovery pattern.

    Returns
    -------
    list[int] | None
        The inferred error pattern, or None if the syndrome is unknown.
    """
    syndrome_tuple = tuple(syndrome)
    if syndrome_tuple in lookup_table:
        return lookup_table[syndrome_tuple]


def error_correction_block(received_block, lookup_table):
    """Correct a single block by applying the inferred error pattern.

    The syndrome of the received block is computed and used to choose a recovery
    operation from the lookup table. If no matching syndrome is found, the block
    is left unchanged.
    """
    syndrome = syndrome_measurement(received_block)
    error_pattern = decoding_list(syndrome, lookup_table)
    if error_pattern is not None:
        corrected_block = [(received_block[i] ^ error_pattern[i]) for i in range(len(received_block))]
        return corrected_block
    else:
        return received_block


def error_correction_message(received_message, lookup_table, d):
    """Decode an entire message block-by-block.

    The message is partitioned into chunks of length d, and each chunk is decoded
    independently using the supplied lookup table.
    """
    corrected_message = []
    for i in range(0, len(received_message), d):
        block = received_message[i:i + d]
        corrected_block = error_correction_block(block, lookup_table)
        corrected_message.extend(corrected_block)
    return corrected_message



def trial(lookup_table, d, p):
    """Run one Monte Carlo trial and report whether decoding succeeded."""
    encoded = encoding(message, d)
    noisy = error_sampling(encoded, p)
    corrected = error_correction_message(noisy, lookup_table, d)
    return corrected == encoded


def error_rate(lookup_table, d, p, trials=1000):
    """Estimate the logical error probability for one physical error rate."""
    successes = sum(trial(lookup_table, d, p) for _ in range(trials))
    return 1 - (successes / trials)


# Parameter sweep over several physical error rates and code lengths.
results = []
for d in code_lengths:
    rates = []
    lookup_table = build_minimum_weight_lookup_table(d)
    for p in physical_error_probabilities:
        rates.append(error_rate(lookup_table, d, p, trials=num_trials))

    # Estimate the logical error probability using the scaling form
    # p_L ≈ a * p^((d+1)/2), which is expected for repetition codes.
    exponent = (d + 1) // 2
    x = physical_error_probabilities ** exponent
    y = rates
    a = sum(x * y) / sum(x * x)
    fit_quality = sum((y - a * x) ** 2) / len(y)
    print(f"Estimated a for d={d}: {a}")
    print(f"Quality of fit for d={d}: {fit_quality}")
    plt.plot(physical_error_probabilities, rates, marker="o", label=f"d={d}")
    plt.plot(physical_error_probabilities, a * physical_error_probabilities ** exponent, linestyle="--", label=f"Fit d={d}")

    for p, rate in zip(physical_error_probabilities, rates):
        results.append(
            {
                "distance": d,
                "physical_error_probability": p,
                "logical_error_rate": rate,
                "fit_coefficient": a,
                "fit_quality": fit_quality,
            }
        )

with data_file.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "distance",
            "physical_error_probability",
            "logical_error_rate",
            "fit_coefficient",
            "fit_quality",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

plt.xlabel("physical error probability p")
plt.ylabel("logical error probability")
plt.title("Logical error probability vs physical error probability for repetition code")
plt.legend()
plt.grid(True)
plt.savefig(figure_file, dpi=300, bbox_inches="tight")
plt.show()