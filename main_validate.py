from validation.symbol_error_rate_torch import main
from time import perf_counter

t0 = perf_counter()
main()
print(perf_counter() - t0)