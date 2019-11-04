include("misc.jl")

function leastSquaresBasis(X,y)
	Z = polyBasis(X)
	v = (Z'Z)\(Z'y)

	function predict(Xhat)
		Z = polyBasis(Xhat)
		return Z*v
	end
	# Return model
	return GenericModel(predict)

end

function polyBasis(X)
	(n) = size(X,1)
	z = Array{Float64}(undef, n, 0)
	z = [z (X.^0)]
	z = [z (X.^1)]
	z = [z (X.^2)]
	z = [z (X.^3)]
	z = [z sin.(X.*5)]
	return z
end
