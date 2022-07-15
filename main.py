
from fractions import Fraction
import math
import random
import time
from typing import Generator, Tuple, Iterable, Sequence, List, Callable

# Total number of megaminds.
N = 10

# Probability changing step.
# The smaller it is, the more accurate the calculations.
# Not used for newer version `solve_norm`.
STEP = 1


FloatGen = Generator[float, None, None]
Prob = Tuple[float, float]
Probs = List[Prob]
ProbsGen = Generator[Probs, None, None]
ProbsGenT = Callable[..., ProbsGen]


def float_range(start: float, stop: float, step: float) -> FloatGen:
	yield start

	if start == stop:
		return

	delta = step * step * step
	p = start + step
	while p + delta < stop:
		yield p
		p += step

	yield stop


def one_gen(step: float) -> FloatGen:
	for white in float_range(0, 1, step):
		for black in float_range(0, 1 - white, step):
			# Yield probabilities of saying "white" and "black".
			yield white, black


def one_gen_norm(n: int, i: int) -> FloatGen:
	# step = 2 / (min(i + 1, n - i) + 1)
	# step = 1 / min(i + 1, n - i)
	# step = 1 / ((min(i + 1, n - i) << 1) - 1)
	step = 1 / (min(i + 1, n - i) * 3 - 2)

	return one_gen(step)


def one_rand() -> Prob:
	white, black = random.random(), random.random()
	if white + black <= 1:
		return white, black
	else:
		return 1 - white, 1 - black


def all_gen(n: int, step: float) -> ProbsGen:
	# one_gen() must yield at least one value.
	digit_gens = [one_gen(step) for _ in range(n)]
	digits = [next(digit_gens[i]) for i in range(n)]

	while True:
		yield digits

		for i in range(n):
			try:
				digits[i] = next(digit_gens[i])
			except StopIteration:
				digit_gens[i] = one_gen(step)
				digits[i] = next(digit_gens[i])
			else:
				break
		else:
			break


def all_gen_norm(n: int) -> ProbsGen:
	# one_gen_norm() must yield at least one value.
	digit_gens = [one_gen_norm(n, i) for i in range(n)]
	digits = [next(digit_gens[i]) for i in range(n)]

	while True:
		yield digits

		for i in range(n):
			try:
				digits[i] = next(digit_gens[i])
			except StopIteration:
				digit_gens[i] = one_gen_norm(n, i)
				digits[i] = next(digit_gens[i])
			else:
				break
		else:
			break


def all_gen_rand(n: int) -> ProbsGen:
	digits = [one_rand() for _ in range(n)]

	while True:
		yield digits

		for i in range(n):
			digits[i] = one_rand()


def probs_str(probs: Iterable[Prob], precision: int = 2) -> str:
	return ",".join(
		f"({white:.{precision}f},{black:.{precision}f})"
		for white, black in probs
	)


def cond_surviving_prob(whites: int, probs: Sequence[Prob]) -> float:
	"""Surviving probability if there are `whites` white caps"""
	n = len(probs)
	blacks = n - whites

	# Megamind in white cap sees `whites - 1` white caps.
	if whites:
		white_for_white, black_for_white = probs[whites - 1]
	else:
		white_for_white, black_for_white = 0, 0

	# Megamind in black cap sees `whites - 1` white caps.
	if blacks:
		white_for_black, black_for_black = probs[whites]
	else:
		white_for_black, black_for_black = 0, 0

	silence_for_white = 1 - white_for_white - black_for_white
	silence_for_black = 1 - white_for_black - black_for_black

	noone_fails_prob = (1 - black_for_white) ** whites * (1 - white_for_black) ** blacks
	all_silence_prob = silence_for_white ** whites * silence_for_black ** blacks

	#   w b s
	# 0
	# 1
	# 2
	# 3 + - -
	# i - + -

	#      w   b   s
	#  0   0   1   0
	#  1   1   0   0
	#  2   0   0   1
	#  3   0   1   0
	#  4   1   0   0
	#  5   0   0   1
	#  6   0   1   0
	#  7   1   0   0
	#  8   0   0   1
	#  9   0   1   0

	return noone_fails_prob - all_silence_prob


def surviving_prob(probs: Sequence[Prob]) -> float:
	"""Final surviving probability."""
	n = len(probs)

	return sum(
		# Weight of distribution multiplied by surviving probability.
		math.comb(n, whites) * cond_surviving_prob(whites, probs)
		for whites in range(n+1)
	) / 2**n  # Must be normalized to one.


def _solve(_all_gen: ProbsGenT, n: int = N, log: bool = False, **kwargs):
	best_s_prob = 0
	best_probs = None

	for i, probs in enumerate(_all_gen(n, **kwargs)):
		# probs[whites] is the probabilities of saying 'white' and 'black'
		# respectively when you see `whites` white caps.
		s_prob = surviving_prob(probs)
		if s_prob > best_s_prob:
			best_s_prob = s_prob
			best_s_prob_f = Fraction(best_s_prob)
			best_probs = probs
			if log:
				print(f"Try {i}:")
				print(f"{best_s_prob} = {best_s_prob_f.numerator}/{best_s_prob_f.denominator} for:")
				print(probs_str(best_probs))

	return best_s_prob, best_probs


def solve(n: int = N, step: float = STEP, log: bool = False) -> Tuple[float, Probs]:
	return _solve(all_gen, n, log, step=step)


def solve_norm(n: int = N, log: bool = False) -> Tuple[float, Probs]:
	return _solve(all_gen_norm, n, log)


def solve_rand(n: int = N) -> None:
	_solve(all_gen_rand, n, log=True)


def calculate_one(n: int = N, step: float = STEP, log: bool = False) -> None:
	time0 = time.time()

	best_s_prob, best_probs = solve(n, step, log)
	best_s_prob_f = Fraction(best_s_prob)
	print(
		f"Best surviving probability for {n} megaminds:"
		f" {best_s_prob} = {best_s_prob_f.numerator}/{best_s_prob_f.denominator}"
	)
	if best_probs is not None:
		print(probs_str(best_probs))

	time1 = time.time()
	print(f"{time1 - time0:.2f} seconds")


def calculate_all(max_n: int = N, step: float = STEP, log: bool = False) -> None:
	for n in range(1, max_n + 1):
		calculate_one(n, step, log)
		print()


def calculate_one_norm(n: int = N, log: bool = False) -> None:
	time0 = time.time()

	best_s_prob, best_probs = solve_norm(n, log)
	best_s_prob_f = Fraction(best_s_prob)
	print(
		f"Best surviving probability for {n} megaminds:"
		f" {best_s_prob} = {best_s_prob_f.numerator}/{best_s_prob_f.denominator}"
	)
	if best_probs is not None:
		print(probs_str(best_probs))

	time1 = time.time()
	print(f"{time1 - time0:.2f} seconds")


def calculate_all_norm(max_n: int = N, log: bool = False) -> None:
	for n in range(1, max_n + 1):
		calculate_one_norm(n, log)
		print()


def main():
	"""
	000 0
	001 0
	010 1
	011 1
	100 1
	101 1
	110 2
	111 2

	:return:
	"""
	# solve_rand(N)

	calculate_one_norm(N, log=True)

	# 0.6669921875 for:
	# (0.00,1.00) (1.00,0.00) (0.00,0.00) (0.00,1.00) (1.00,0.00) (0.00,0.00) (0.00,1.00) (1.00,0.00) (0.00,0.00) (0.00,1.00)

	# calculate_all_norm(N)

	# for i, probs in enumerate(all_gen_norm(6)):
	# 	print(i, probs_str(probs))

	# for white, black in one_gen_norm(5, 0):
	# 	print(white, black)

	...


if __name__ == '__main__':
	main()

