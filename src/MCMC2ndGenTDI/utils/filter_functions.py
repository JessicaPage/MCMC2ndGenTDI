import numpy as np
from .. import settings


def delay_polynomials(D):

    """Return delay polynomial portion of the FDI filter."""

    D=D*settings.f_s
    integer_part, d_frac = np.divmod(D,1)
    integer_part = integer_part-settings.p
    d_frac = d_frac+settings.p
    delay_polynomial_array = np.ones((settings.number_n+1,settings.length))
    factors = -1*d_frac+settings.ints
    delay_polynomial_array[1:settings.number_n+1] = np.cumprod(factors,axis=0)
    return delay_polynomial_array,int(integer_part[0])

def trim_data(dataset,filter_array,einsum_path_to_use):

	"""The FDI filter implementation."""

	ein_path = einsum_path_to_use[0]
	dataset=np.roll(dataset,filter_array[1],axis=0)
	dataset[:filter_array[1]] = 0.0
	val = np.einsum('ij,ji->i',dataset,filter_array[0],optimize=ein_path)
	return val

